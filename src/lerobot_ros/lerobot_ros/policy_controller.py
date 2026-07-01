import contextlib
import json
import threading
import time
import types
from collections import deque
from queue import Queue
from typing import Dict, List, Optional, Tuple

import rclpy
import rclpy.executors
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor.pipeline import DataProcessorPipeline
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot_interfaces.action import RunPolicy
from lerobot_interfaces.msg import PolicyStatus
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Empty, String

from .config import PolicyConfig, ROSFeatureConfig, load_toml_dict, parse_config
from .core import RosFeaturePublisher
from .core.frame_assembler import FrameAssembler
from .episode_tracker import EpisodeTracker, stack_progress_window
from .ros_torch_utils import BaseTopic, prepare_frame
from .subscriber import Ros2Feature

register_third_party_plugins()


# Latched, depth-1 profile for the status topic so late joiners get the last state.
STATUS_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)
# Reliable, depth-1 profile for the safety heartbeat.
HEARTBEAT_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
)


class PolicyController:
    """Control robot with output from a pretrained policy.

    Autonomous execution is driven through the ``run_policy`` action server:
    sending a goal runs a policy, cancelling the goal (or losing the safety
    heartbeat) stops it. The latest state is published on a latched
    ``policy_control/status`` topic.
    """

    def __init__(self, node: Node, config: ROSFeatureConfig, subscriber_node: Node):
        self.node = node
        self.config = config

        # Topic setup
        self.convertor = Ros2Feature(
            subscriber_node,
            topics=config.topics,
            fps=config.fps,
            rerun_remote=config.rerun_remote,
            visualize=config.visualize,
        )

        self.observation_queue = Queue(maxsize=100)
        self.action_queue = deque(maxlen=100)

        # Mutable run state, guarded by _state_lock for atomic transitions.
        self._state_lock = threading.Lock()
        self.running = False
        self.collect_frames = False
        self._active_policy = None
        self._current_task = ""
        self._latest_progress = 0.0
        self._task_start_mono = None
        self._goal_active = False

        self.predict_thread = threading.Thread(target=self.predict_loop, daemon=True)

        # When benchmark=True actions are published under "benchmark/<topic>" so
        # the robot never moves; detailed per-inference timing is collected and
        # published on policy_control/metrics as JSON.
        self.benchmark = (
            node.declare_parameter("benchmark", False)
            .get_parameter_value()
            .bool_value
        )

        self.timings: Dict[str, List[float]] = {
            "preprocess": [],
            "inference": [],
            "postprocess": [],
            "blend": [],
            "publish": [],
        }

        self.policies: Dict[
            str, Tuple[DataProcessorPipeline, PreTrainedPolicy, DataProcessorPipeline]
        ] = {}

        # Progress estimators (optional, per policy) and their rolling frame
        # history, loaded in-process here instead of via a separate node/topic
        # so the estimator in use can never go stale relative to the active policy.
        self.progress_models: Dict[str, EpisodeTracker] = {}
        self.progress_windows: Dict[str, deque] = {}

        # Parameters
        self.task_completion_threshold = (
            node.declare_parameter("task_completion_threshold", 0.9)
            .get_parameter_value()
            .double_value
        )
        self.max_episode_length_s = (
            node.declare_parameter("max_episode_length_s", 0.0)
            .get_parameter_value()
            .double_value
        )
        self.heartbeat_timeout_s = (
            node.declare_parameter("heartbeat_timeout_s", 0.5)
            .get_parameter_value()
            .double_value
        )
        # How long to hold a paused trajectory before discarding it as stale.
        # Short pauses resume seamlessly; longer ones re-predict from scratch.
        # 0.0 disables flushing (hold indefinitely).
        self.pause_flush_timeout_s = (
            node.declare_parameter("pause_flush_timeout_s", 5.0)
            .get_parameter_value()
            .double_value
        )
        self._last_heartbeat = 0.0

        # Safety heartbeat: autonomy only publishes while this is refreshed.
        self.heartbeat_sub = node.create_subscription(
            Empty, "policy_control/heartbeat", self.heartbeat_callback, HEARTBEAT_QOS
        )

        # Latched status topic for operators / GUIs.
        self.status_pub = node.create_publisher(
            PolicyStatus, "policy_control/status", STATUS_QOS
        )

        # Per-inference timing metrics (only published when benchmark=True).
        self.metrics_pub = (
            node.create_publisher(String, "policy_control/metrics", 10)
            if self.benchmark
            else None
        )
        if self.benchmark:
            node.get_logger().info(
                "Benchmark mode enabled: actions published under 'benchmark/<topic>', "
                "zero-filled synthetic observations at configured FPS, "
                "timing metrics on 'policy_control/metrics'."
            )

        # Action and observation queue
        self.setup_action_topics(config.topics)

        if self.benchmark:
            # Skip real subscribers; push zero-filled observations at FPS rate instead.
            self._synthetic_thread = threading.Thread(
                target=self._synthetic_observation_loop, daemon=True
            )
        else:
            self._synthetic_thread = None
            self.convertor.register_frame_callback(self.frame_callback)
            self.convertor.setup_subscribers()

        self.publisher_thread = threading.Thread(target=self.publisher_loop, daemon=True)

        # Action server
        self._cb_group = ReentrantCallbackGroup()
        self._run_policy_server = ActionServer(
            self.node,
            RunPolicy,
            "run_policy",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self._cb_group,
        )

        self.load_policies(config.policies)

        # Refresh the latched status (heartbeat liveness / progress) at 2 Hz.
        self.status_timer = node.create_timer(0.5, self.publish_status)

        self.publisher_thread.start()
        self.predict_thread.start()

        if self._synthetic_thread is not None:
            self._synthetic_thread.start()
        else:
            self.convertor.running = True

        self.publish_status()

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _start(self, policy_name: str, task: str):
        """Begin autonomous execution of ``policy_name`` for ``task``."""
        with self._state_lock:
            self.active_policy_name = policy_name  # setter validates + weights
            self._current_task = task
            self._latest_progress = 0.0
            if policy_name in self.progress_windows:
                self.progress_windows[policy_name].clear()
            self.action_queue.clear()
            self.collect_frames = True
            self.running = True
            self._task_start_mono = time.monotonic()
        self.publish_status()
        self.node.get_logger().info(
            f"Started policy '{policy_name}' for task '{task}'"
        )
        if not self._heartbeat_alive():
            self.node.get_logger().warn(
                "No recent heartbeat on 'policy_control/heartbeat'; the policy "
                "will run but NOT move until an operator starts publishing it."
            )

    def _stop(self, reason: str = ""):
        """Stop execution and flush queued actions so resume cannot replay them."""
        with self._state_lock:
            self.running = False
            self.collect_frames = False
            self.action_queue.clear()
            self.active_policy_name = None
            self._current_task = ""
            self._latest_progress = 0.0
            self._task_start_mono = None
        self.publish_status()
        if reason:
            self.node.get_logger().info(f"Stopped policy: {reason}")

    def heartbeat_callback(self, _msg: Empty):
        self._last_heartbeat = time.monotonic()

    def _heartbeat_alive(self) -> bool:
        if self.heartbeat_timeout_s <= 0.0:
            return True
        return (time.monotonic() - self._last_heartbeat) <= self.heartbeat_timeout_s

    def publish_status(self):
        msg = PolicyStatus()
        msg.running = self.running
        msg.active_policy = self.active_policy_name or ""
        msg.task = self._current_task or ""
        msg.available_policies = list(self.policies.keys())
        msg.progress = float(self._latest_progress)
        msg.heartbeat_alive = self._heartbeat_alive()
        self.status_pub.publish(msg)

    # ------------------------------------------------------------------
    # Action server
    # ------------------------------------------------------------------

    def goal_callback(self, _goal_request) -> GoalResponse:
        with self._state_lock:
            if self._goal_active:
                self.node.get_logger().warn(
                    "Rejecting run_policy goal: controller is busy."
                )
                return GoalResponse.REJECT
            self._goal_active = True
        return GoalResponse.ACCEPT

    def cancel_callback(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        request = goal_handle.request
        policy_name = request.policy_name
        task = request.task
        result = RunPolicy.Result()

        if policy_name not in self.policies:
            self._goal_active = False
            goal_handle.abort()
            result.success = False
            result.message = f"unknown policy: {policy_name}"
            return result

        policy_cfg = self.config.policies[policy_name]
        has_regressor = policy_cfg.progress_model is not None
        max_len = policy_cfg.max_episode_length_s
        if max_len is None:
            max_len = self.max_episode_length_s  # node-level fallback (0 = disabled)

        if not has_regressor and max_len <= 0.0:
            self.node.get_logger().warn(
                f"Policy '{policy_name}' has no progress regressor and no "
                "max_episode_length_s; it will only stop on cancel or heartbeat loss."
            )

        self._start(policy_name, task)

        feedback = RunPolicy.Feedback()
        try:
            while True:
                time.sleep(0.05)
                elapsed = time.monotonic() - self._task_start_mono

                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    result.message = "cancelled"
                    return result

                # The heartbeat gates *actuation*, not the goal: the operator
                # owns it (e.g. a joystick deadman). While it is stale the goal
                # stays active but nothing is published, so motion pauses and
                # resumes as the operator engages. publisher_loop logs why.
                actuating = self._heartbeat_alive()

                if has_regressor:
                    progress = self._latest_progress
                    if progress >= self.task_completion_threshold:
                        goal_handle.succeed()
                        result.success = True
                        result.message = "completed"
                        return result
                else:
                    progress = (elapsed / max_len) if max_len > 0 else 0.0

                if max_len > 0.0 and elapsed >= max_len:
                    goal_handle.succeed()
                    result.success = True
                    result.message = "timeout"
                    return result

                feedback.progress = float(progress)
                feedback.elapsed_s = float(elapsed)
                feedback.status = "running" if actuating else "waiting_for_heartbeat"
                goal_handle.publish_feedback(feedback)
        finally:
            self._stop(f"goal ended ({result.message or 'unknown'})")
            self._goal_active = False

    # ------------------------------------------------------------------
    # Policy loading / accessors
    # ------------------------------------------------------------------

    def load_policies(self, config: Dict[str, PolicyConfig]):
        """Load policies based on the provided configuration."""
        for name, policy_config in config.items():
            self.load_policy(name, policy_config)

    def setup_action_topics(self, topics: Dict[str, BaseTopic]):
        self.action_publisher = RosFeaturePublisher(
            self.node, topics, topic_prefix="benchmark" if self.benchmark else ""
        )

    def load_policy(self, task, config: PolicyConfig):
        policy_config = PreTrainedConfig.from_pretrained(
            pretrained_name_or_path=config.pretrained_name_or_path,
        )
        policy_config.pretrained_path = config.pretrained_name_or_path
        self.node.get_logger().info(f"Loaded policy config: {policy_config}")

        for key, value in config.policy_config.items():
            setattr(policy_config, key, value)

        if config.ds_repo_id:
            dataset = LeRobotDataset(config.ds_repo_id, config.ds_root)
            ds_meta = dataset.meta
        else:
            # No dataset configured: derive input/output features from the TOML
            # topic config so the policy can be instantiated for benchmarking
            # without a recorded dataset.
            features = FrameAssembler(self.config.topics).get_feature_description()
            ds_meta = types.SimpleNamespace(features=features, stats=None)
            self.node.get_logger().info(
                f"No ds_repo_id for policy '{task}'; using TOML config features."
            )

        policy = make_policy(
            policy_config,
            ds_meta=ds_meta,
            rename_map=config.rename_map,
        )
        policy.eval()

        if getattr(policy_config, "compile_model", False):
            try:
                # "cudagraphs" avoids Inductor/Triton and works on Jetson without a C compiler.
                policy = torch.compile(policy, backend="cudagraphs")
                self.node.get_logger().info(f"torch.compile (cudagraphs) applied to '{task}'")
            except Exception as e:
                self.node.get_logger().warn(f"torch.compile failed, running eager: {e}")

        processor_kwargs = {}
        postprocessor_kwargs = {}

        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": config.device},
        }

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=policy_config.pretrained_path,
            **processor_kwargs,
            **postprocessor_kwargs,
        )
        preprocessor.reset()
        postprocessor.reset()

        # Move policy and pre/postprocessors to device if they support it
        if hasattr(policy, "to"):
            policy = policy.to(config.device)
        if hasattr(preprocessor, "to"):
            preprocessor = preprocessor.to(config.device)
        if hasattr(postprocessor, "to"):
            postprocessor = postprocessor.to(config.device)

        self.policies[task] = (preprocessor, policy, postprocessor)

        if config.progress_model is not None:
            progress_model = EpisodeTracker.from_pretrained(config.progress_model)
            progress_model.eval()
            progress_model = progress_model.to(config.device)
            self.progress_models[task] = progress_model
            self.progress_windows[task] = deque(maxlen=progress_model.window)
            self.node.get_logger().info(
                f"Loaded progress model '{config.progress_model}' for policy '{task}'"
            )

    # ------------------------------------------------------------------
    # Worker loops
    # ------------------------------------------------------------------

    def _synthetic_observation_loop(self):
        """Push zero-filled observations at FPS rate (benchmark mode only)."""
        assembler = FrameAssembler(self.config.topics)
        delta_t = 1.0 / self.config.fps
        next_tick = time.time() + delta_t
        while True:
            now = time.time()
            # assemble() with an empty dict returns zero tensors for every topic
            observation = assembler.assemble({})
            observation["task"] = self._current_task or ""
            observation["robot_type"] = self.config.robot_type
            if self.has_active_policy and self.collect_frames:
                self.observation_queue.put((observation, now))
            time.sleep(max(0.0, next_tick - time.time()))
            next_tick += delta_t

    def frame_callback(self, observation, t):
        if not self.has_active_policy or not self.collect_frames:
            return

        with torch.inference_mode():
            # Add metadata on the callback thread, but do NOT call prepare_frame here
            # to avoid blocking the subscriber thread with tensor processing/device transfers.
            observation["task"] = self._current_task or ""
            observation["robot_type"] = self.config.robot_type

        self.observation_queue.put((observation, t))

    def predict_loop(self):
        delta_t = 1.0 / self.config.fps
        while True:
            observation, t = self.observation_queue.get()
            self.node.get_logger().debug(f"Predicting action for time {t}")

            self.observation_queue.task_done()
            if not self.has_active_policy:
                continue

            pre, policy, post = self.get_active_policy()
            config = self.active_config

            if not policy or not config:
                continue

            # Batch and preprocess the observation on the target device
            with torch.inference_mode():
                observation = prepare_frame(observation, config.device)

            progress_model = self.progress_models.get(self.active_policy_name)
            if progress_model is not None:
                with torch.inference_mode():
                    progress_frame = {
                        key: observation[key]
                        for key in progress_model.progress_keys
                        if key in observation
                    }
                    window = self.progress_windows[self.active_policy_name]
                    window.append(progress_frame)
                    windowed = stack_progress_window(window, progress_model.window)
                    self._latest_progress = progress_model.predict_progress(windowed).item()

            remaining_actions = len(self.action_queue)

            if remaining_actions > config.action_queue_size:
                continue

            # Populate action queue if below desired size.
            # When benchmarking on CUDA, synchronize before each timing point so
            # that async GPU kernels are fully counted in the measured window.
            is_cuda = config.device.startswith("cuda")

            def _sync():
                if self.benchmark and is_cuda:
                    torch.cuda.synchronize()

            _sync()
            t0 = time.time()

            with torch.inference_mode():
                observation = pre(observation)
            _sync()
            t1 = time.time()

            _autocast = (
                torch.autocast(config.device, dtype=torch.float16)
                if config.autocast and is_cuda
                else contextlib.nullcontext()
            )
            with torch.inference_mode(), _autocast:
                action_chunk = policy.predict_action_chunk(observation)
            _sync()
            t2 = time.time()

            with torch.inference_mode():
                # Postprocess the entire action chunk at once on the target device
                action_chunk = post(action_chunk)
                # Squeeze the batch dimension and move to CPU in a single step
                actions = action_chunk.squeeze(0).to("cpu")
            _sync()
            t3 = time.time()

            passed_actions = remaining_actions - len(self.action_queue)
            old_actions = deque(self.action_queue)
            self.action_queue.clear()

            for i, action in enumerate(actions[passed_actions:]):
                if len(old_actions) > 0:
                    old_action, _ = old_actions.popleft()
                    w = self.action_weights[passed_actions + i]
                    action = action * (1 - w) + old_action * w

                self.action_queue.append((action, t))
                t += delta_t
            t4 = time.time()

            preprocess_ms = (t1 - t0) * 1e3
            inference_ms = (t2 - t1) * 1e3
            postprocess_ms = (t3 - t2) * 1e3
            blend_ms = (t4 - t3) * 1e3

            self.timings["preprocess"].append(preprocess_ms)
            self.timings["inference"].append(inference_ms)
            self.timings["postprocess"].append(postprocess_ms)
            self.timings["blend"].append(blend_ms)

            if self.benchmark and self.metrics_pub is not None:
                msg = String()
                msg.data = json.dumps({
                    "type": "inference",
                    "policy": self.active_policy_name or "",
                    "preprocess_ms": round(preprocess_ms, 3),
                    "inference_ms": round(inference_ms, 3),
                    "postprocess_ms": round(postprocess_ms, 3),
                    "blend_ms": round(blend_ms, 3),
                    "total_ms": round((t4 - t0) * 1e3, 3),
                    "queue_depth": remaining_actions,
                    "chunk_size": len(actions),
                    "timestamp": t0,
                })
                self.metrics_pub.publish(msg)

    def publisher_loop(self):
        delta_t = 1.0 / self.config.fps
        paused_since = None
        flushed_while_paused = False
        while True:
            now = time.time()
            next_iter = now + delta_t

            # Only consume the queue when we are actually going to actuate.
            # Popping while paused would discard the head of the trajectory and,
            # on resume, jump to a later action with the first segment skipped.
            # While paused we leave the queue intact; the position controller
            # holds the arm, and predict_loop refreshes the trajectory.
            if not self.running:
                paused_since = None
                flushed_while_paused = False
                time.sleep(max(0, next_iter - time.time()))
                continue

            # Defense in depth: never actuate unless the safety heartbeat is
            # fresh. The heartbeat is published by the operator (e.g. a joystick
            # deadman) and lives outside this repo, so a running policy that
            # never moves is almost always a missing heartbeat. Say so loudly,
            # but only while there is a trajectory waiting to go out.
            if not self._heartbeat_alive():
                if paused_since is None:
                    paused_since = time.monotonic()
                if len(self.action_queue) > 0:
                    self.node.get_logger().warn(
                        "Policy '%s' is running but no recent heartbeat on "
                        "'policy_control/heartbeat' (timeout %.2fs): actions are "
                        "NOT being published. An operator must publish the "
                        "heartbeat (e.g. hold the joystick deadman) to enable "
                        "actuation."
                        % (self.active_policy_name, self.heartbeat_timeout_s),
                        throttle_duration_sec=2.0,
                    )
                # Hold the trajectory for a short pause (seamless resume), but
                # discard it once the pause gets long so we re-predict from the
                # current observation instead of resuming a stale plan.
                if (
                    not flushed_while_paused
                    and self.pause_flush_timeout_s > 0.0
                    and (time.monotonic() - paused_since) >= self.pause_flush_timeout_s
                ):
                    self.action_queue.clear()
                    flushed_while_paused = True
                    self.node.get_logger().info(
                        "Paused for >%.1fs without heartbeat; flushed stale "
                        "trajectory, will re-predict fresh on resume."
                        % self.pause_flush_timeout_s
                    )
                time.sleep(max(0, next_iter - time.time()))
                continue

            # Actuating again: clear pause bookkeeping.
            paused_since = None
            flushed_while_paused = False

            if len(self.action_queue) == 0:
                time.sleep(max(0, next_iter - time.time()))
                continue

            action, t = self.action_queue.popleft()
            t_pub0 = time.time()
            self.action_publisher.publish(action)
            t_pub1 = time.time()

            if self.benchmark:
                pub_ms = (t_pub1 - t_pub0) * 1e3
                self.timings["publish"].append(pub_ms)
                if self.metrics_pub is not None:
                    msg = String()
                    msg.data = json.dumps({
                        "type": "publish",
                        "policy": self.active_policy_name or "",
                        "publish_ms": round(pub_ms, 3),
                        "timestamp": t_pub0,
                    })
                    self.metrics_pub.publish(msg)

            time.sleep(max(0, next_iter - time.time()))

    # ------------------------------------------------------------------
    # Active-policy helpers
    # ------------------------------------------------------------------

    @property
    def active_policy_name(self) -> Optional[str]:
        return self._active_policy

    @property
    def has_active_policy(self) -> bool:
        return self._active_policy is not None and self._active_policy in self.policies

    def f(self, x, N, beta=1.0):
        return (1 - x / N) ** (2**beta)

    @active_policy_name.setter
    def active_policy_name(self, value):
        if value is not None and value not in self.policies:
            raise ValueError(f"Policy {value} is not loaded.")
        self._active_policy = value

        if value is not None:
            self.calculate_action_weights()

    def calculate_action_weights(self):
        config = self.active_config

        N = config.action_queue_size
        self.action_weights = [
            self.f(x, N, config.action_smoothing_beta) for x in range(N)
        ]

    def get_active_policy(
        self,
    ) -> tuple[DataProcessorPipeline, PreTrainedPolicy, DataProcessorPipeline]:
        """Get the currently active policy components (pre, policy, post)."""
        if not self.active_policy_name:
            raise ValueError("No active policy set.")
        return self.policies[self.active_policy_name]

    @property
    def active_config(self) -> PolicyConfig:
        if self.active_policy_name is None:
            raise ValueError("No active policy set.")
        return self.config.policies[self.active_policy_name]

    def cleanup(self):
        def _stats(values):
            if not values:
                return None
            n = len(values)
            avg = sum(values) / n
            mn = min(values)
            mx = max(values)
            sorted_v = sorted(values)
            p50 = sorted_v[n // 2]
            p95 = sorted_v[int(n * 0.95)]
            p99 = sorted_v[int(n * 0.99)]
            return {"n": n, "avg": avg, "min": mn, "max": mx, "p50": p50, "p95": p95, "p99": p99}

        keys = ["preprocess", "inference", "postprocess", "blend", "publish"]
        for key in keys:
            s = _stats(self.timings.get(key, []))
            if s:
                self.node.get_logger().info(
                    f"[benchmark] {key:>12}: n={s['n']}  avg={s['avg']:.2f}ms  "
                    f"min={s['min']:.2f}ms  p50={s['p50']:.2f}ms  "
                    f"p95={s['p95']:.2f}ms  p99={s['p99']:.2f}ms  max={s['max']:.2f}ms"
                )


def main():
    rclpy.init()
    node = rclpy.create_node("ai_control_node")
    subscriber_node = rclpy.create_node("subscriber_node")

    config = parse_config(
        load_toml_dict(
            node.declare_parameter("config", "config.toml")
            .get_parameter_value()
            .string_value
        )
    )
    controller = PolicyController(node, config, subscriber_node)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(subscriber_node)
    executor.add_node(node)

    try:
        node.get_logger().info("Starting AI control...")
        executor.spin()
    except KeyboardInterrupt:
        pass
    controller.cleanup()
    node.destroy_node()
    subscriber_node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
