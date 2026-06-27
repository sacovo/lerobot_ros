import os
import shutil
import tempfile
import pytest
import torch
from rclpy.qos import QoSProfile

from std_msgs.msg import Float32, Float32MultiArray
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_ros.convert.std import Float32Topic, Float32MultiArrayTopic
from lerobot_ros.ros_torch_utils import TensorToRosConverter
from lerobot_ros.config import ROSFeatureConfig, PolicyConfig
from lerobot_ros.core import FrameAssembler, DatasetWriter


def test_round_to_nearest_scalar():
    """Test that round_to_nearest correctly rounds scalar float topics in TensorToRosConverter."""
    qos = QoSProfile(depth=10)

    # 1. Round to nearest enabled
    topic_round = Float32Topic(
        topic_name="/test/gripper", qos=qos, tag="action", round_to_nearest=True
    )
    assert topic_round.round_to_nearest is True

    converter_round = TensorToRosConverter({"/test/gripper": topic_round})

    # Test conversion of various float values
    test_cases = [
        (0.6, 1.0),
        (-0.6, -1.0),
        (0.4, 0.0),
        (-0.2, 0.0),
        (1.8, 2.0),
        (-1.9, -2.0),
    ]

    for val, expected in test_cases:
        t = torch.tensor([val], dtype=torch.float32)
        msgs = converter_round.convert(t)
        assert "/test/gripper" in msgs
        msg = msgs["/test/gripper"]
        assert isinstance(msg, Float32)
        assert msg.data == expected

    # 2. Round to nearest disabled
    topic_no_round = Float32Topic(
        topic_name="/test/gripper_raw", qos=qos, tag="action", round_to_nearest=False
    )
    assert topic_no_round.round_to_nearest is False

    converter_no_round = TensorToRosConverter({"/test/gripper_raw": topic_no_round})

    for val, _ in test_cases:
        t = torch.tensor([val], dtype=torch.float32)
        msgs = converter_no_round.convert(t)
        msg = msgs["/test/gripper_raw"]
        assert msg.data == pytest.approx(val)


def test_round_to_nearest_array():
    """Test that round_to_nearest correctly rounds elements in float array topics."""
    qos = QoSProfile(depth=10)

    topic = Float32MultiArrayTopic(
        names=["joint_a", "joint_b"],
        topic_name="/test/joints",
        qos=qos,
        tag="action",
        round_to_nearest=True,
    )
    assert topic.round_to_nearest is True

    converter = TensorToRosConverter({"/test/joints": topic})

    t = torch.tensor([0.6, -1.8], dtype=torch.float32)
    msgs = converter.convert(t)
    assert "/test/joints" in msgs
    msg = msgs["/test/joints"]
    assert isinstance(msg, Float32MultiArray)
    # The output should be rounded to [1.0, -2.0]
    assert list(msg.data) == [1.0, -2.0]


def test_dataset_writer_store_retrieval():
    """Integration test verifying DatasetWriter correctly stores values in LeRobotDataset."""
    temp_dir = tempfile.mkdtemp()
    try:
        qos = QoSProfile(depth=10)

        # Define some topics
        topics = {
            "/test/scalar": Float32Topic(
                topic_name="/test/scalar", qos=qos, tag="observation"
            ),
            "/test/action_float": Float32Topic(
                topic_name="/test/action_float", qos=qos, tag="action", round_to_nearest=True
            ),
            "/test/action_array": Float32MultiArrayTopic(
                names=["x", "y"],
                topic_name="/test/action_array",
                qos=qos,
                tag="action",
                round_to_nearest=False,
            ),
        }

        config = ROSFeatureConfig(
            topics=topics,
            fps=10,
            dataset_root=temp_dir,
            tolerance_s=0.1,
        )

        writer = DatasetWriter("test_dataset", config, resume=False)
        assembler = FrameAssembler(topics)

        # Assemble and write a few frames for an episode
        episode_frames = []
        for i in range(5):
            t_sec = float(i) * 0.1
            latest_msgs = {
                "/test/scalar": torch.tensor([1.5 + float(i)], dtype=torch.float32),
                "/test/action_float": torch.tensor([0.6], dtype=torch.float32),  # if rounded is enabled, does DatasetWriter round?
                # Wait, DatasetWriter stores what is assembled from latest_msgs.
                # Since latest_msgs has standard raw values (before TensorToRosConverter / Policy publish rounding),
                # they are stored with full float precision in the dataset.
                # Rounding is only for policy-to-ROS publishing (policy control output).
                # But let's verify both are stored correctly.
                "/test/action_array": torch.tensor([2.5, -3.5], dtype=torch.float32),
            }
            frame = assembler.assemble(latest_msgs)
            episode_frames.append((frame, "test_task", t_sec))

        writer.save_episode(episode_frames, "test_task")
        writer.finalize()

        # Load back the dataset directly from disk to verify contents
        dataset_path = os.path.join(temp_dir, "test_dataset")
        assert os.path.exists(dataset_path)

        dataset = LeRobotDataset("test_dataset", root=dataset_path)
        assert dataset.num_episodes == 1
        assert dataset.num_frames == 5

        # Check features
        features = dataset.features
        assert "observation.state" in features
        assert "action" in features

        # State size: size of /test/scalar = 1
        assert features["observation.state"]["shape"] == (1,)
        # Action size: size of /test/action_float (1) + /test/action_array (2) = 3
        assert features["action"]["shape"] == (3,)

        # Inspect values of the stored frames
        for idx in range(5):
            frame = dataset[idx]
            assert frame["task"] == "test_task"
            # Compare state
            assert frame["observation.state"].item() == pytest.approx(1.5 + float(idx))
            # Compare action values
            # Index 0: /test/action_float, Index 1-2: /test/action_array
            assert frame["action"][0].item() == pytest.approx(0.6)
            assert frame["action"][1].item() == pytest.approx(2.5)
            assert frame["action"][2].item() == pytest.approx(-3.5)

    finally:
        shutil.rmtree(temp_dir)


def test_limits_and_rounding():
    """Test our custom limits and round_values features on scalar and vector topics."""
    qos = QoSProfile(depth=10)

    # 1. Test scalar topic with limits and round_values
    topic_scalar = Float32Topic(
        topic_name="/test/scalar_action",
        qos=qos,
        tag="action",
        limits=[-1.0, 1.0],
        round_values=[-1.0, 1.0],
    )
    converter_scalar = TensorToRosConverter({"/test/scalar_action": topic_scalar})

    # Test cases: (input, expected)
    test_cases_scalar = [
        (0.6, 1.0),    # rounds to 1.0
        (-0.8, -1.0),  # rounds to -1.0
        (1.5, 1.0),    # limited to 1.0, then rounded to 1.0
        (-2.5, -1.0),  # limited to -1.0, then rounded to -1.0
        (0.1, 1.0),    # closest to 1.0
    ]
    for val, expected in test_cases_scalar:
        t = torch.tensor([val], dtype=torch.float32)
        msgs = converter_scalar.convert(t)
        assert msgs["/test/scalar_action"].data == expected

    # 2. Test array topic with limits and round_values as lists/tuples (per-index)
    topic_array = Float32MultiArrayTopic(
        names=["joint_a", "joint_b"],
        topic_name="/test/joints",
        qos=qos,
        tag="action",
        limits=[[-1.0, 1.0], [-2.0, 2.0]],
        round_values=[[-0.5, 0.5], [1.0, -1.0, 0.0]],
    )
    converter_array = TensorToRosConverter({"/test/joints": topic_array})

    # Case A: within limits, round to nearest allowed value
    t = torch.tensor([0.4, 0.8], dtype=torch.float32)
    msgs = converter_array.convert(t)
    # 0.4 -> closest to 0.5 in [-0.5, 0.5]
    # 0.8 -> closest to 1.0 in [1.0, -1.0, 0.0]
    assert list(msgs["/test/joints"].data) == [0.5, 1.0]

    # Case B: exceeds limits, gets clamped, then rounded
    t = torch.tensor([1.5, -3.0], dtype=torch.float32)
    msgs = converter_array.convert(t)
    # 1.5 -> clamped to 1.0 -> closest to 0.5 in [-0.5, 0.5]
    # -3.0 -> clamped to -2.0 -> closest to -1.0 in [1.0, -1.0, 0.0]
    assert list(msgs["/test/joints"].data) == [0.5, -1.0]

    # 3. Test JointStateTopic with dict limits and round_values
    from lerobot_ros.convert.sensor import JointStateTopic
    topic_js = JointStateTopic(
        joints=["shoulder", "wrist", "gripper"],
        position=True,
        velocity=False,
        effort=False,
        topic_name="/test/js",
        qos=qos,
        tag="action",
        limits={"shoulder": [-1.57, 1.57], "gripper": [-1.0, 1.0]},
        round_values={"gripper": [-1.0, 1.0]},
    )
    converter_js = TensorToRosConverter({"/test/js": topic_js})

    # shoulder.position (index 0) limits [-1.57, 1.57], no rounding
    # wrist.position (index 1) no limits, no rounding
    # gripper.position (index 2) limits [-1.0, 1.0], rounding to [-1.0, 1.0]
    t = torch.tensor([2.0, 5.0, 0.6], dtype=torch.float32)
    msgs = converter_js.convert(t)
    msg = msgs["/test/js"]
    # 2.0 -> clamped to 1.57
    assert msg.position[0] == pytest.approx(1.57)
    # 5.0 -> unchanged (no limit)
    assert msg.position[1] == pytest.approx(5.0)
    # 0.6 -> rounded to 1.0
    assert msg.position[2] == pytest.approx(1.0)

