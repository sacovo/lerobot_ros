from typing import Dict, Any
import torch
from rclpy.node import Node
from lerobot_ros.convert.base import BaseTopic
from lerobot_ros.ros_torch_utils import TensorToRosConverter

class RosFeaturePublisher:
    def __init__(self, node: Node, topics: Dict[str, BaseTopic]):
        self.node = node
        self.topics = topics
        # Only publish topics tagged as 'action' or 'meta'
        self.publishers = {
            name: node.create_publisher(topic.msg_type(), name, 10)
            for name, topic in topics.items() if topic.is_action or topic.is_meta
        }
        self.converter = TensorToRosConverter(topics)

    def publish(self, action_tensor: torch.Tensor):
        """Converts PyTorch tensor to messages and publishes them on matching topics."""
        msgs = self.converter.convert(action_tensor)
        for topic_name, msg in msgs.items():
            if topic_name in self.publishers:
                self.publishers[topic_name].publish(msg)
