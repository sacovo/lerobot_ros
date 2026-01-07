import os
import threading
import time
from typing import List

import rclpy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import precise_sleep

from .config import load_toml_dict, parse_config
from .ros_torch_utils import TensorToRosConverter


def main():
    rclpy.init()
    node = rclpy.create_node("replay_node")
    logger = node.get_logger()

    # Dataset parameters
    root = node.declare_parameter("root", "").get_parameter_value().string_value
    repo_id = node.declare_parameter("repo_id", "").get_parameter_value().string_value

    ds = LeRobotDataset(repo_id, tolerance_s=0.001, root=root if root else None)
    logger.info(f"Loaded dataset: {ds}")

    # Config
    config_path = (
        node.declare_parameter(
            "config", os.getenv("CONFIG_PATH", "config/hufi_arm.toml")
        )
        .get_parameter_value()
        .string_value
    )
    config = parse_config(load_toml_dict(config_path))
    logger.info(f"Loaded config from {config_path}")

    # Replay parameters
    repetitions = (
        node.declare_parameter("repetitions", 1).get_parameter_value().integer_value
    )

    all_episodes = list(ds.meta.episodes["episode_index"])

    episodes_param = (
        node.declare_parameter("episodes", []).get_parameter_value().integer_array_value
    )
    episodes: List[int] = list(episodes_param) if episodes_param else all_episodes

    logger.info(f"Replaying episodes {episodes} with {repetitions} repetition(s)")

    # Setup publishers
    converter = TensorToRosConverter(config.topics)
    publishers = {
        topic_name: node.create_publisher(topic.msg_type(), topic_name, 10)
        for topic_name, topic in config.topics.items()
    }

    # Spin ROS in background thread
    ros_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    ros_thread.start()

    frame_duration = 1.0 / ds.fps

    for rep in range(repetitions):
        logger.info(f"Repetition {rep + 1}/{repetitions}")

        for episode_idx in episodes:
            episode_data = ds.hf_dataset.filter(
                lambda x: x["episode_index"] == episode_idx
            )
            logger.info(f"Playing episode {episode_idx} ({len(episode_data)} frames)")

            precise_sleep(frame_duration)

            for frame_idx, frame in enumerate(episode_data):
                start_t = time.perf_counter()

                action = frame["action"]
                msgs = converter.convert(action)

                for topic, msg in msgs.items():
                    publishers[topic].publish(msg)

                elapsed = time.perf_counter() - start_t
                precise_sleep(frame_duration - elapsed)

    logger.info("Replay complete")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
