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


def test_float64_action_is_coerced_to_float32():
    """A Float64MultiArray action (e.g. /position_controller/commands) must not
    make every frame fail with 'feature action of dtype float64 is not float32'.

    FrameAssembler casts the stacked action/observation.state to float32 to match
    the schema, so the whole episode builds instead of producing an empty buffer
    that then wedges video encoding.
    """
    import torch
    from lerobot_ros.core import DatasetWriter, FrameAssembler
    from lerobot_ros.convert.std import Float64MultiArrayTopic, Float32Topic
    from lerobot_ros.config import ROSFeatureConfig

    temp_dir = tempfile.mkdtemp()
    try:
        pos = Float64MultiArrayTopic(
            names=[f"j{i}" for i in range(6)],
            topic_name="/position_controller/commands",
            tag="action", qos={},
        )
        grip = Float32Topic(
            topic_name="/wecant/GRAB/Grip_Man/Set", tag="action", key="grip_man", qos={}
        )
        cfg = ROSFeatureConfig(
            topics={"/position_controller/commands": pos, "/wecant/GRAB/Grip_Man/Set": grip},
            fps=20, dataset_root=temp_dir,
        )
        writer = DatasetWriter("float64_action_ds", cfg, resume=False)
        assembler = FrameAssembler(cfg.topics)

        frames = []
        for i in range(20):
            latest = {
                "/position_controller/commands": torch.tensor([0.1 * i] * 6, dtype=torch.float64),
                "/wecant/GRAB/Grip_Man/Set": torch.tensor([1.0], dtype=torch.float32),
            }
            frames.append((assembler.assemble(latest), "pick", i / 20.0))

        # Must not raise (previously: every frame rejected -> empty episode).
        writer.save_episode(frames, "pick", success=True)
        writer.finalize()

        ds = LeRobotDataset("float64_action_ds", root=os.path.join(temp_dir, "float64_action_ds"))
        assert ds.meta.total_episodes == 1
        assert ds.meta.total_frames == 20
    finally:
        shutil.rmtree(temp_dir)


def test_schema_mismatch_aborts_episode_loudly():
    """A genuine schema mismatch (wrong action shape) must raise a clear error
    and leave no pending buffer, instead of silently saving a broken/empty
    episode that hangs ffmpeg during video encoding."""
    import torch
    from lerobot_ros.core import DatasetWriter
    from lerobot_ros.convert.std import Float32Topic
    from lerobot_ros.config import ROSFeatureConfig

    temp_dir = tempfile.mkdtemp()
    try:
        grip = Float32Topic(
            topic_name="/wecant/GRAB/Grip_Man/Set", tag="action", key="grip_man", qos={}
        )
        cfg = ROSFeatureConfig(
            topics={"/wecant/GRAB/Grip_Man/Set": grip}, fps=20, dataset_root=temp_dir
        )
        writer = DatasetWriter("bad_shape_ds", cfg, resume=False)

        # Schema expects a 1-element action; feed 2 elements so every frame is rejected.
        frames = [
            ({"action": torch.tensor([1.0, 2.0], dtype=torch.float32)}, "pick", i / 20.0)
            for i in range(20)
        ]
        with pytest.raises(RuntimeError, match="rejected by the dataset writer"):
            writer.save_episode(frames, "pick", success=True)

        # Buffer discarded -> writer is clean, not stuck mid-episode.
        assert not writer.dataset.has_pending_frames()
    finally:
        shutil.rmtree(temp_dir)
