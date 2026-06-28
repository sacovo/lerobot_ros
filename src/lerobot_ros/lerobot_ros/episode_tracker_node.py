import threading

import torch
from lerobot_interfaces.msg import PolicyStatus, TaskProgress
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile

from .config import load_toml_dict, parse_config
from .episode_tracker import EpisodeTracker
from .ros_torch_utils import prepare_frame
from .subscriber import Ros2Feature


class EpisodeTrackerNode:
    def __init__(self, node: Node):
        self.node = node

        config = parse_config(
            load_toml_dict(
                node.declare_parameter("config", "config.toml")
                .get_parameter_value()
                .string_value
            )
        )

        self.models: dict[str, EpisodeTracker] = {}
        self.devices = {}
        self.active_policy = None

        for name, policy in config.policies.items():
            if policy.progress_model is None:
                continue

            self.models[name] = EpisodeTracker.from_pretrained(
                policy.progress_model
            ).to(policy.device)
            self.devices[name] = policy.device

        self.convertor = Ros2Feature(
            node, config.topics, config.fps, rerun_remote=None, visualize=False
        )

        status_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.status_sub = node.create_subscription(
            PolicyStatus,
            "policy_control/status",
            self.status_callback,
            status_qos,
        )

        self.convertor.register_frame_callback(self.frame_callback)
        self.convertor.setup_subscribers()
        self.convertor.running = True

        self.episode_progress_publisher = node.create_publisher(
            TaskProgress, "episode_progress", 10
        )
        self.last_progress = None

        self.current_frame = None
        self.frame_lock = threading.Lock()

        self.timer = node.create_timer(0.5, self.publish_progress)

    def frame_callback(self, frame: dict[str, torch.Tensor], t) -> None:
        with self.frame_lock:
            self.current_frame = frame

    def shutdown(self):
        self.convertor.running = False
        self.timer.cancel()

    def status_callback(self, msg: PolicyStatus) -> None:
        policy_name = msg.active_policy
        if policy_name in self.models:
            self.active_policy = policy_name
        else:
            self.active_policy = None

    def publish_progress(self):
        with self.frame_lock:
            frame = self.current_frame

        if frame is None:
            return

        if self.active_policy is None:
            return

        model = self.models[self.active_policy]
        prepare_frame(frame, self.devices[self.active_policy])

        with torch.no_grad():
            progress = model(frame).item()

        msg = TaskProgress()
        msg.policy_name = self.active_policy
        msg.progress = progress
        self.episode_progress_publisher.publish(msg)
        self.last_progress = progress


def main():
    import rclpy

    rclpy.init()
    node = rclpy.create_node("episode_tracker_node")
    episode_tracker_node = EpisodeTrackerNode(node)

    rclpy.spin(node)
    episode_tracker_node.shutdown()

    rclpy.shutdown()
