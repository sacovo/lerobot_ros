import threading
import time
from collections import deque
from queue import Queue
from typing import Dict, Optional, Tuple

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
from lerobot_interfaces.msg import PolicyStatus, TaskProgress
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Empty

from .config import PolicyConfig, ROSFeatureConfig, load_toml_dict, parse_config
from .core import RosFeaturePublisher
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

        self.timings = {
            "predict": [],
            "blend": [],
        }

        self.policies: Dict[
            str, Tuple[DataProcessorPipeline, PreTrainedPolicy, DataProcessorPipeline]
        ] = {}

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
        self._last_heartbeat = 0.0

        # Safety heartbeat: autonomy only publishes while this is refreshed.
        self.heartbeat_sub = node.create_subscription(
            Empty, "policy_control/heartbeat", self.heartbeat_callback, HEARTBEAT_QOS
        )

        # Progress regressor feedback (optional, per policy).
        self.progress_subscriber = node.create_subscription(
            TaskProgress, "episode_progress", self.progress_callback, 10
        )

        # Latched status topic for operators / GUIs.
        self.status_pub = node.create_publisher(
            PolicyStatus, "policy_control/status", STATUS_QOS
        )

        # Action and observation queue
        self.setup_action_topics(config.topics)

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
            self.action_queue.clear()
            self.collect_frames = True
            self.running = True
            self._task_start_mono = time.monotonic()
        self.publish_status()
        self.node.get_logger().info(
            f"Started policy '{policy_name}' for task '{task}'"
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

    def progress_callback(self, msg: TaskProgress):
        if msg.policy_name != self.active_policy_name:
            return
        self._latest_progress = msg.progress

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

                if not self._heartbeat_alive():
                    goal_handle.abort()
                    result.success = False
                    result.message = "heartbeat lost"
                    return result

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
                feedback.status = "running"
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
        self.action_publisher = RosFeaturePublisher(self.node, topics)

    def load_policy(self, task, config: PolicyConfig):
        policy_config = PreTrainedConfig.from_pretrained(
            pretrained_name_or_path=config.pretrained_name_or_path,
        )
        policy_config.pretrained_path = config.pretrained_name_or_path
        self.node.get_logger().info(f"Loaded policy config: {policy_config}")

        for key, value in config.policy_config.items():
            setattr(policy_config, key, value)

        dataset = LeRobotDataset(config.ds_repo_id, config.ds_root)
        ds_meta = dataset.meta
        policy = make_policy(
            policy_config,
            ds_meta=ds_meta,
            rename_map=config.rename_map,
        )
        policy.eval()

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

    # ------------------------------------------------------------------
    # Worker loops
    # ------------------------------------------------------------------

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

            remaining_actions = len(self.action_queue)

            if remaining_actions > config.action_queue_size:
                continue

            # Populate action queue if below desired size
            t0 = time.time()

            with torch.inference_mode():
                observation = pre(observation)
                action_chunk = policy.predict_action_chunk(observation)
                # Postprocess the entire action chunk at once on the target device
                action_chunk = post(action_chunk)
                # Squeeze the batch dimension and move to CPU in a single step
                actions = action_chunk.squeeze(0).to("cpu")

            t1 = time.time()

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
            t2 = time.time()

            self.timings["predict"].append(t1 - t0)
            self.timings["blend"].append(t2 - t1)

    def publisher_loop(self):
        delta_t = 1.0 / self.config.fps
        while True:
            now = time.time()
            next_iter = now + delta_t

            if len(self.action_queue) == 0:
                time.sleep(max(0, next_iter - time.time()))
                continue

            action, t = self.action_queue.popleft()

            # Defense in depth: never actuate unless running and the safety
            # heartbeat is fresh.
            if not self.running or not self._heartbeat_alive():
                time.sleep(max(0, next_iter - time.time()))
                continue

            self.action_publisher.publish(action)

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
        if self.timings and self.timings["predict"]:
            avg_predict = sum(self.timings["predict"]) / len(self.timings["predict"])
            avg_blend = sum(self.timings["blend"]) / len(self.timings["blend"])
            self.node.get_logger().info(
                f"Average predict time: {avg_predict:.6f} seconds"
            )
            self.node.get_logger().info(f"Average blend time: {avg_blend:.6f} seconds")


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
