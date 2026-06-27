import os
import shutil
import tempfile
import pytest
import sys
from unittest.mock import patch

import rosbag2_py
from rclpy.serialization import serialize_message
from std_msgs.msg import Float32, String
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_ros.bag_to_dataset import main as bag_to_dataset_main


def create_test_bag(bag_path, storage_id="mcap"):
    writer = rosbag2_py.SequentialWriter()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
    writer.open(storage_options, converter_options)

    # Create topic metadata
    float_topic = rosbag2_py.TopicMetadata(
        id=0,
        name="/test/float",
        type="std_msgs/msg/Float32",
        serialization_format="cdr"
    )
    writer.create_topic(float_topic)

    task_topic = rosbag2_py.TopicMetadata(
        id=1,
        name="/task",
        type="std_msgs/msg/String",
        serialization_format="cdr"
    )
    writer.create_topic(task_topic)

    # t=0: idle
    msg = String()
    msg.data = "idle"
    writer.write("/task", serialize_message(msg), 0)

    # t=1.0s: start task_1
    msg = String()
    msg.data = "task_1"
    writer.write("/task", serialize_message(msg), 1_000_000_000)

    # t=1.2s: float message
    msg = Float32()
    msg.data = 10.0
    writer.write("/test/float", serialize_message(msg), 1_200_000_000)

    # t=1.5s: float message
    msg = Float32()
    msg.data = 20.0
    writer.write("/test/float", serialize_message(msg), 1_500_000_000)

    # t=2.0s: idle / reset
    msg = String()
    msg.data = "idle"
    writer.write("/task", serialize_message(msg), 2_000_000_000)

    # t=2.2s: float message (during reset, should be ignored)
    msg = Float32()
    msg.data = 30.0
    writer.write("/test/float", serialize_message(msg), 2_200_000_000)

    # t=3.0s: start task_2
    msg = String()
    msg.data = "task_2"
    writer.write("/task", serialize_message(msg), 3_000_000_000)

    # t=3.3s: float message
    msg = Float32()
    msg.data = 40.0
    writer.write("/test/float", serialize_message(msg), 3_300_000_000)

    # t=4.0s: end
    msg = String()
    msg.data = ""
    writer.write("/task", serialize_message(msg), 4_000_000_000)


def test_bag_to_dataset_conversion():
    # Setup temporary directory for test outputs
    temp_dir = tempfile.mkdtemp()
    
    try:
        bag_path = os.path.join(temp_dir, "test_bag")
        config_path = os.path.join(temp_dir, "test_config.toml")
        dataset_root = os.path.join(temp_dir, "datasets")
        
        # Write config TOML
        config_content = f"""
        fps = 10
        dataset_root = "{dataset_root}"

        [topics]
        "/test/float" = {{ msg_type = "Float32" }}
        """
        with open(config_path, "w") as f:
            f.write(config_content)
            
        # Create mock bag
        create_test_bag(bag_path, storage_id="mcap")
        
        # Run conversion programmatically
        test_args = [
            "bag_to_dataset",
            "--bag-path", bag_path,
            "--config", config_path,
            "--repo-id", "test_mcap_dataset",
            "--task-topic", "/task"
        ]
        
        with patch.object(sys, 'argv', test_args):
            bag_to_dataset_main()
            
        # Verify dataset was created and contains correct data
        dataset_path = os.path.join(dataset_root, "test_mcap_dataset")
        assert os.path.exists(dataset_path)
        
        # Load dataset
        dataset = LeRobotDataset("test_mcap_dataset", root=dataset_path)
        
        # We expect two episodes (task_1 and task_2)
        assert dataset.num_episodes == 2
        
        # Verify episode tasks
        assert dataset.meta.episodes[0]['tasks'][0] == "task_1"
        assert dataset.meta.episodes[1]['tasks'][0] == "task_2"
        
        # Let's inspect the frames for Episode 0 (task_1)
        from_idx_0 = dataset.meta.episodes[0]["dataset_from_index"]
        to_idx_0 = dataset.meta.episodes[0]["dataset_to_index"]
        len_0 = to_idx_0 - from_idx_0
        assert len_0 > 0
        
        # Check task name in frames
        for idx in range(from_idx_0, to_idx_0):
            frame = dataset[idx]
            assert frame["task"] == "task_1"

        # Let's inspect the frames for Episode 1 (task_2)
        from_idx_1 = dataset.meta.episodes[1]["dataset_from_index"]
        to_idx_1 = dataset.meta.episodes[1]["dataset_to_index"]
        len_1 = to_idx_1 - from_idx_1
        assert len_1 > 0
        
        for idx in range(from_idx_1, to_idx_1):
            frame = dataset[idx]
            assert frame["task"] == "task_2"

    finally:
        shutil.rmtree(temp_dir)
