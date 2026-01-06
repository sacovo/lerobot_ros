import threading

import torch
from rclpy.node import Node
from std_msgs.msg import Int8

from .config import load_toml_dict, parse_config
from .episode_tracker import EpisodeTracker
from .ros_torch_utils import prepare_frame
from .subscriber import Ros2Feature


class EpisodeTrackerNode:
    def __init__(self, node: Node):
        self.node = node

        self.model = EpisodeTracker.from_pretrained(
            node.declare_parameter(
                "repo_id", "fhnwrover/so101-ros-red-ring-episode-tracker"
            )
            .get_parameter_value()
            .string_value
        )

        config = parse_config(
            load_toml_dict(
                node.declare_parameter("config", "config.toml")
                .get_parameter_value()
                .string_value
            )
        )

        self.convertor = Ros2Feature(
            node, config.topics, config.fps, rerun_remote=None, visualize=False
        )

        self.convertor.register_frame_callback(self.frame_callback)
        self.convertor.setup_subscribers()
        self.convertor.running = True

        self.episode_progress_publisher = node.create_publisher(
            Int8, "episode_progress", 10
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

    def publish_progress(self):
        with self.frame_lock:
            frame = self.current_frame

        if frame is None:
            return

        prepare_frame(frame, "cpu")

        progress = int(self.model(frame).item() * 100)

        if progress == self.last_progress:
            return

        msg = Int8()
        msg.data = progress
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
