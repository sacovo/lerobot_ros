import threading
import time
from collections import deque
from queue import Queue
from typing import Dict, Optional, Tuple

from lerobot_interfaces.srv import SetActivePolicy, Calibrate, ListPolicies
import rclpy
import rclpy.executors
import torch
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import (
    make_policy,
    make_pre_post_processors,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor.pipeline import DataProcessorPipeline
from lerobot.utils.import_utils import register_third_party_plugins

from .config import PolicyConfig, ROSFeatureConfig, load_toml_dict, parse_config
from .ros_torch_utils import BaseTopic, TensorToRosConverter, prepare_frame
from .subscriber import Ros2Feature


register_third_party_plugins()


class PolicyController:
    """Control robot with output from a pretrained policy."""

    def __init__(self, node: Node, config: ROSFeatureConfig, subscriber_node: Node):
        self.node = node
        self.config = config

        # Topic setup
        self.convertor = Ros2Feature(
            subscriber_node,
            topics=config.topics,
            fps=config.fps,
            rerun_remote=config.rerrun_remote,
            visualize=config.visualize,
        )

        self.observation_queue = Queue(maxsize=100)
        self.action_queue = deque(maxlen=100)

        self.qos_profile = 10

        self.task = None
        self._active_policy = None

        self.running = False
        self.collect_frames = False

        self.predict_thread = threading.Thread(
            target=self.predict_loop,
            daemon=True,
        )

        self.timings = {
            "predict": [],
            "blend": [],
        }

        self.calibration = False
        self.calibration_frames = Queue(maxsize=100)
        self.calibration_n_frames = 20
        self.calibration_thread = threading.Thread(
            target=self.calibrate_ttt_thread,
            args=(self.calibration_frames,),
            daemon=True,
        )

        self._predicted_timesteps = set()
        self.publishers = {}

        self.policies: Dict[
            str, Tuple[DataProcessorPipeline, PreTrainedPolicy, DataProcessorPipeline]
        ] = {}

        self.task_subscrber = node.create_subscription(
            String, "/task", self.task_callback, 10
        )

        # Action and observation queue

        self.setup_action_topics(config.topics)

        self.convertor.register_frame_callback(self.frame_callback)
        self.convertor.setup_subscribers()

        self.publisher_thread = threading.Thread(
            target=self.publisher_loop,
            daemon=True,
        )
        self.task = None

        self.node.create_service(ListPolicies, "/list_policies", self.list_policies)
        self.node.create_service(
            SetBool, "/set_policy_running", self.set_policy_running
        )
        self.node.create_service(
            Calibrate, "/calibrate", self.trigger_calibration_service
        )
        self.node.create_service(
            Trigger, "/toggle_policy_running", self.toggle_policy_running_service
        )
        self.node.create_service(
            SetActivePolicy, "/set_active_policy", self.set_policy_service
        )

        self.load_policies(config.policies)
        self.autonomy_modse = "manual"

        self.publisher_thread.start()
        self.predict_thread.start()
        self.calibration_thread.start()

        self.convertor.running = True

    def task_callback(self, msg: String):
        task = msg.data
        self.task = task

    def trigger_calibration_service(
        self, request: Calibrate.Request, response: Calibrate.Response
    ):
        self.calibration_n_frames = request.frames
        self.trigger_calibration()
        response.success = True

        return response

    def trigger_calibration(self):
        self.running = False
        self.calibration_frames.queue.clear()
        self.collect_frames = True
        self.calibration = True

    def _set_running(self, running: bool):
        self.running = running
        if not running:
            self.collect_frames = False
        else:
            self.collect_frames = True

    def toggle_policy_running_service(
        self, request: Trigger.Request, response: Trigger.Response
    ):
        self._set_running(not self.running)

        response.success = True
        response.message = f"Policy running set to {self.running}"
        return response

    def set_policy_running(self, request: SetBool.Request, response: SetBool.Response):
        self._set_running(request.data)

        response.success = True
        response.message = f"Policy running set to {self.running}"
        return response

    def load_policies(self, config: Dict[str, PolicyConfig]):
        """Load policies based on the provided configuration."""
        for name, policy_config in config.items():
            self.load_policy(name, policy_config)

    def list_policies(
        self, request: ListPolicies.Request, response: ListPolicies.Response
    ):
        tasks = list(self.policies.keys())
        response.policy_names = tasks
        return response

    def set_policy_service(
        self, request: SetActivePolicy.Request, response: SetActivePolicy.Response
    ):
        self.set_policy(request.policy_name)
        response.success = True
        return response

    def set_policy(self, policy_name: str):
        if policy_name not in self.policies:
            raise ValueError(f"Policy {policy_name} is not loaded.")
        self.active_policy_name = policy_name
        self._predicted_timesteps.clear()

    def end_task(self):
        self.active_policy_name = None

    def setup_action_topics(self, topics: Dict[str, BaseTopic]):
        self.torch_to_ros = TensorToRosConverter(topics)

        for topic_name, topic in topics.items():
            if topic.is_action:
                self.publishers[topic_name] = self.node.create_publisher(
                    topic.msg_type(), topic_name, topic.qos
                )

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
        # policy.reset()

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

        self.policies[task] = (preprocessor, policy, postprocessor)

    def frame_callback(self, observation, t):
        if not self.has_active_policy or not self.collect_frames:
            return

        with torch.inference_mode():
            prepare_frame(observation, "cpu")

            observation["task"] = self.task or ""
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

            if self.calibration:
                observation = pre(observation)
                self.calibration_frames.put(observation)
                continue

            remaining_actions = len(self.action_queue)

            if remaining_actions > config.action_queue_size:
                continue

            # Populate action queue if below desired size
            t0 = time.time()

            with torch.inference_mode():
                observation = pre(observation)
                action_chunk = policy.predict_action_chunk(observation)
                actions = action_chunk.transpose(0, 1)

            t1 = time.time()

            passed_actions = remaining_actions - len(self.action_queue)
            old_actions = deque(self.action_queue)
            self.action_queue.clear()

            for i, action in enumerate(actions[passed_actions:]):
                action = post(action)
                action = action.squeeze(0).to("cpu")

                if len(old_actions) > 0:
                    old_action, _ = old_actions.popleft()
                    w = self.action_weights[passed_actions + i]
                    action = action * (1 - w) + old_action * w

                self.action_queue.append((action, t))
                t += delta_t
            t2 = time.time()

            self.timings["predict"].append(t1 - t0)
            self.timings["blend"].append(t2 - t1)

    def calibrate_ttt_thread(self, frames):
        frames = []
        while True:
            frame = self.calibration_frames.get()
            frames.append(frame)
            self.calibration_frames.task_done()

            if len(frames) < self.calibration_n_frames:
                continue

            self.collect_frames = False
            self.calibration = False
            self.calibration_frames.queue.clear()

            batch = {}

            self.node.get_logger().info(
                f"Starting calibration with {len(frames)} frames."
            )

            for key, value in frames[0].items():
                if isinstance(value, torch.Tensor):
                    batch[key] = torch.cat([f[key] for f in frames], dim=0)
                else:
                    batch[key] = [f[key] for f in frames]

            policy = self.active_policy
            if hasattr(policy, "test_time_train"):
                policy.test_time_train(batch)
                self.node.get_logger().info("Calibration completed.")
            else:
                self.node.get_logger().info(
                    "Policy does not support test-time training."
                )
            frames = []

    def publisher_loop(self):
        delta_t = 1.0 / self.config.fps
        while True:
            now = time.time()
            next_iter = now + delta_t

            if len(self.action_queue) == 0:
                time.sleep(max(0, next_iter - time.time()))
                continue

            action, t = self.action_queue.popleft()
            msgs = self.torch_to_ros.convert(action)

            if not self.running:
                time.sleep(max(0, next_iter - time.time()))
                continue

            for topic, msg in msgs.items():
                self.publishers[topic].publish(msg)

            time.sleep(max(0, next_iter - time.time()))

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

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(subscriber_node)
    executor.add_node(node)

    try:
        node.get_logger().info("Starting AI control...")
        executor.spin()
    except KeyboardInterrupt:
        pass
    controller.cleanup()
    node.destroy_node()
    # subscriber_node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
