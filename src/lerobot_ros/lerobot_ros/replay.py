import os
import threading
import time

import rclpy

from ares_reach.config import load_toml_dict, parse_config
from ares_reach.ros_torch_utils import TensorToRosConverter
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import precise_sleep

# import pyo3_example



def my_callback(x: int) -> int:
    print(f"Callback called with {x}")
    return x + 1


def main():
    rclpy.init()
    node = rclpy.create_node("replay_node")
    root = node.declare_parameter("root", "").get_parameter_value().string_value
    ds = LeRobotDataset(
        node.declare_parameter("repo_id", "").get_parameter_value().string_value,
        tolerance_s=0.001,
        root=root if root else None,
    )
    node.get_logger().info(f"Loaded dataset: {ds}")
    config_path = (
        node.declare_parameter(
            "config", os.getenv("CONFIG_PATH", "config/hufi_arm.toml")
        )
        .get_parameter_value()
        .string_value
    )
    config = parse_config(load_toml_dict(config_path))
    node.get_logger().info(f"Loaded config from {config_path}")
    node.get_logger().info(f"Config: {config}")

    convertor = TensorToRosConverter(config.topics)

    actions = ds.hf_dataset.select_columns("action")

    observations = ds.hf_dataset.select_columns("observation.state")

    images = ds.features.get("observation.image", None)

    publishers = {}

    for topic_name, topic in config.topics.items():
        publishers[topic_name] = node.create_publisher(topic.msg_type(), topic_name, 10)

    def ros_thread_func():
        rclpy.spin(node)

    ros_thread = threading.Thread(target=ros_thread_func, daemon=True)
    ros_thread.start()
    iterations = (
        node.declare_parameter("iterations", 1).get_parameter_value().integer_value
    )

    for _ in range(iterations):
        precise_sleep(1 / ds.fps)
        for i, data in enumerate(ds.hf_dataset):
            print(data)
            action = data["action"]
            observation = data["observation.state"]
            start_episode_t = time.perf_counter()

            node.get_logger().info(f"Step {i}: {action}")
            action_array = actions[i]["action"]

            msgs = convertor.convert(action)
            for topic, msg in msgs.items():
                publishers[topic].publish(msg)

            dt_s = time.perf_counter() - start_episode_t
            precise_sleep(1 / ds.fps - dt_s)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
