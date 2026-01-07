"""
Tests for ROS message to tensor conversions.

Tests both directions:
- to_tensor: ROS message -> PyTorch tensor
- from_tensor: PyTorch tensor -> ROS message
"""

import pytest
import torch

# geometry_msgs
from geometry_msgs.msg import (
    Accel,
    Point,
    Point32,
    Pose,
    Pose2D,
    PoseStamped,
    Quaternion,
    Transform,
    Twist,
    TwistStamped,
    Vector3,
    Wrench,
)
from rclpy.qos import QoSProfile

# sensor_msgs
from sensor_msgs.msg import (
    FluidPressure,
    Illuminance,
    Imu,
    MagneticField,
    NavSatStatus,
    Range,
    RelativeHumidity,
    Temperature,
)

# std_msgs
from std_msgs.msg import (
    Bool,
    Float32,
    Float32MultiArray,
    Float64,
    Float64MultiArray,
    Int8,
    Int16,
    Int32,
    Int32MultiArray,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
)

from lerobot_ros.convert import geometry, sensor, std

# Default QoS for testing
DEFAULT_QOS = QoSProfile(depth=10)


# =============================================================================
# std_msgs scalar tests
# =============================================================================


class TestStdMsgsScalar:
    """Test scalar std_msgs conversions."""

    @pytest.mark.parametrize(
        "topic_cls,msg_cls,value,dtype",
        [
            (std.BoolTopic, Bool, True, torch.bool),
            (std.BoolTopic, Bool, False, torch.bool),
            (std.Int8Topic, Int8, -128, torch.int8),
            (std.Int8Topic, Int8, 127, torch.int8),
            (std.Int16Topic, Int16, -32768, torch.int16),
            (std.Int16Topic, Int16, 32767, torch.int16),
            (std.Int32Topic, Int32, -2147483648, torch.int32),
            (std.Int32Topic, Int32, 2147483647, torch.int32),
            (std.Int64Topic, Int64, 0, torch.int64),
            (std.UInt8Topic, UInt8, 0, torch.uint8),
            (std.UInt8Topic, UInt8, 255, torch.uint8),
            (std.UInt16Topic, UInt16, 0, torch.int32),  # torch has no uint16
            (std.UInt16Topic, UInt16, 65535, torch.int32),
            (std.UInt32Topic, UInt32, 0, torch.int64),  # torch has no uint32
            (std.UInt64Topic, UInt64, 0, torch.int64),  # torch has no uint64
            (std.Float32Topic, Float32, 3.14, torch.float32),
            (std.Float32Topic, Float32, -1.0e10, torch.float32),
            (std.Float64Topic, Float64, 3.141592653589793, torch.float64),
            (std.Float64Topic, Float64, -1.0e100, torch.float64),
        ],
    )
    def test_to_tensor(self, topic_cls, msg_cls, value, dtype):
        """Test converting ROS message to tensor."""
        topic = topic_cls(topic_name="/test/scalar", qos=DEFAULT_QOS)

        msg = msg_cls()
        msg.data = value

        tensor = topic.to_tensor(msg)

        assert tensor.dtype == dtype
        assert tensor.shape == (1,)
        # Use approx for floating point comparisons
        if dtype in (torch.float32, torch.float64):
            assert tensor[0].item() == pytest.approx(value, rel=1e-6)
        else:
            assert tensor[0].item() == value

    @pytest.mark.parametrize(
        "topic_cls,msg_cls,value",
        [
            (std.Int32Topic, Int32, 42),
            (std.Float32Topic, Float32, 3.14),
            (std.Float64Topic, Float64, 2.718281828),
        ],
    )
    def test_roundtrip(self, topic_cls, msg_cls, value):
        """Test roundtrip: msg -> tensor -> msg."""
        topic = topic_cls(topic_name="/test/roundtrip", qos=DEFAULT_QOS)

        # Create original message
        original_msg = msg_cls()
        original_msg.data = value

        # Convert to tensor
        tensor = topic.to_tensor(original_msg)

        # Convert back to message
        recovered_msg = topic.from_tensor(tensor.squeeze())

        assert recovered_msg.data == pytest.approx(value, rel=1e-6)


# =============================================================================
# std_msgs MultiArray tests
# =============================================================================


class TestStdMsgsMultiArray:
    """Test MultiArray std_msgs conversions."""

    def test_float64_multi_array_to_tensor(self):
        """Test Float64MultiArray to tensor."""
        topic = std.Float64MultiArrayTopic(
            names=["a", "b", "c"],
            topic_name="/test/array",
            qos=DEFAULT_QOS,
        )

        msg = Float64MultiArray()
        msg.data = [1.0, 2.0, 3.0]

        tensor = topic.to_tensor(msg)

        assert tensor.dtype == torch.float64  # Uses native float64 dtype
        assert tensor.shape == (3,)
        assert tensor.tolist() == pytest.approx([1.0, 2.0, 3.0])

    def test_float64_multi_array_roundtrip(self):
        """Test Float64MultiArray roundtrip."""
        topic = std.Float64MultiArrayTopic(
            names=["x", "y", "z"],
            topic_name="/test/array",
            qos=DEFAULT_QOS,
        )

        original_msg = Float64MultiArray()
        original_msg.data = [1.5, -2.5, 3.5]

        tensor = topic.to_tensor(original_msg)
        recovered_msg = topic.from_tensor(tensor)

        assert list(recovered_msg.data) == pytest.approx([1.5, -2.5, 3.5])

    def test_float32_multi_array_roundtrip(self):
        """Test Float32MultiArray roundtrip."""
        topic = std.Float32MultiArrayTopic(
            names=["a", "b"],
            topic_name="/test/f32array",
            qos=DEFAULT_QOS,
        )

        original_msg = Float32MultiArray()
        original_msg.data = [1.0, 2.0]

        tensor = topic.to_tensor(original_msg)
        recovered_msg = topic.from_tensor(tensor)

        assert list(recovered_msg.data) == pytest.approx([1.0, 2.0])

    def test_int32_multi_array_roundtrip(self):
        """Test Int32MultiArray roundtrip."""
        topic = std.Int32MultiArrayTopic(
            names=["i", "j", "k"],
            topic_name="/test/i32array",
            qos=DEFAULT_QOS,
        )

        original_msg = Int32MultiArray()
        original_msg.data = [10, 20, 30]

        tensor = topic.to_tensor(original_msg)
        recovered_msg = topic.from_tensor(tensor)

        assert list(recovered_msg.data) == [10, 20, 30]

    def test_feature_description(self):
        """Test feature description includes correct names."""
        topic = std.Float64MultiArrayTopic(
            names=["joint1", "joint2", "joint3"],
            topic_name="/robot/joints",
            qos=DEFAULT_QOS,
        )

        desc = topic.feature_description()

        assert desc["dtype"] == "float64"
        assert desc["shape"] == (3,)
        assert desc["names"] == [
            "robot.joints.joint1",
            "robot.joints.joint2",
            "robot.joints.joint3",
        ]


# =============================================================================
# geometry_msgs tests
# =============================================================================


class TestGeometryMsgs:
    """Test geometry_msgs conversions."""

    def test_point_to_tensor(self):
        """Test Point to tensor."""
        topic = geometry.PointTopic(topic_name="/test/point", qos=DEFAULT_QOS)

        msg = Point()
        msg.x = 1.0
        msg.y = 2.0
        msg.z = 3.0

        tensor = topic.to_tensor(msg)

        assert tensor.dtype == torch.float32
        assert tensor.shape == (3,)
        assert tensor.tolist() == pytest.approx([1.0, 2.0, 3.0])

    def test_point_roundtrip(self):
        """Test Point roundtrip."""
        topic = geometry.PointTopic(topic_name="/test/point", qos=DEFAULT_QOS)

        original_msg = Point()
        original_msg.x = 1.5
        original_msg.y = -2.5
        original_msg.z = 3.5

        tensor = topic.to_tensor(original_msg)
        recovered_msg = topic.from_tensor(tensor)

        assert recovered_msg.x == pytest.approx(1.5)
        assert recovered_msg.y == pytest.approx(-2.5)
        assert recovered_msg.z == pytest.approx(3.5)

    def test_point32_roundtrip(self):
        """Test Point32 roundtrip."""
        topic = geometry.Point32Topic(topic_name="/test/point32", qos=DEFAULT_QOS)

        original_msg = Point32()
        original_msg.x = 1.0
        original_msg.y = 2.0
        original_msg.z = 3.0

        tensor = topic.to_tensor(original_msg)
        recovered_msg = topic.from_tensor(tensor)

        assert recovered_msg.x == pytest.approx(1.0)
        assert recovered_msg.y == pytest.approx(2.0)
        assert recovered_msg.z == pytest.approx(3.0)

    def test_vector3_roundtrip(self):
        """Test Vector3 roundtrip."""
        topic = geometry.Vector3Topic(topic_name="/test/vector3", qos=DEFAULT_QOS)

        original_msg = Vector3()
        original_msg.x = 0.1
        original_msg.y = 0.2
        original_msg.z = 0.3

        tensor = topic.to_tensor(original_msg)
        recovered_msg = topic.from_tensor(tensor)

        assert recovered_msg.x == pytest.approx(0.1)
        assert recovered_msg.y == pytest.approx(0.2)
        assert recovered_msg.z == pytest.approx(0.3)

    def test_quaternion_roundtrip(self):
        """Test Quaternion roundtrip."""
        topic = geometry.QuaternionTopic(topic_name="/test/quat", qos=DEFAULT_QOS)

        original_msg = Quaternion()
        original_msg.x = 0.0
        original_msg.y = 0.0
        original_msg.z = 0.707
        original_msg.w = 0.707

        tensor = topic.to_tensor(original_msg)
        recovered_msg = topic.from_tensor(tensor)

        assert recovered_msg.x == pytest.approx(0.0)
        assert recovered_msg.y == pytest.approx(0.0)
        assert recovered_msg.z == pytest.approx(0.707)
        assert recovered_msg.w == pytest.approx(0.707)

    def test_pose2d_roundtrip(self):
        """Test Pose2D roundtrip."""
        topic = geometry.Pose2DTopic(topic_name="/test/pose2d", qos=DEFAULT_QOS)

        original_msg = Pose2D()
        original_msg.x = 1.0
        original_msg.y = 2.0
        original_msg.theta = 1.57

        tensor = topic.to_tensor(original_msg)
        recovered_msg = topic.from_tensor(tensor)

        assert recovered_msg.x == pytest.approx(1.0)
        assert recovered_msg.y == pytest.approx(2.0)
        assert recovered_msg.theta == pytest.approx(1.57)

    def test_pose_roundtrip(self):
        """Test Pose (with nested fields) roundtrip."""
        topic = geometry.PoseTopic(topic_name="/test/pose", qos=DEFAULT_QOS)

        original_msg = Pose()
        original_msg.position.x = 1.0
        original_msg.position.y = 2.0
        original_msg.position.z = 3.0
        original_msg.orientation.x = 0.0
        original_msg.orientation.y = 0.0
        original_msg.orientation.z = 0.0
        original_msg.orientation.w = 1.0

        tensor = topic.to_tensor(original_msg)

        assert tensor.shape == (7,)

        recovered_msg = topic.from_tensor(tensor)

        assert recovered_msg.position.x == pytest.approx(1.0)
        assert recovered_msg.position.y == pytest.approx(2.0)
        assert recovered_msg.position.z == pytest.approx(3.0)
        assert recovered_msg.orientation.w == pytest.approx(1.0)

    def test_transform_roundtrip(self):
        """Test Transform roundtrip."""
        topic = geometry.TransformTopic(topic_name="/test/tf", qos=DEFAULT_QOS)

        original_msg = Transform()
        original_msg.translation.x = 1.0
        original_msg.translation.y = 2.0
        original_msg.translation.z = 3.0
        original_msg.rotation.x = 0.0
        original_msg.rotation.y = 0.0
        original_msg.rotation.z = 0.0
        original_msg.rotation.w = 1.0

        tensor = topic.to_tensor(original_msg)
        recovered_msg = topic.from_tensor(tensor)

        assert recovered_msg.translation.x == pytest.approx(1.0)
        assert recovered_msg.translation.y == pytest.approx(2.0)
        assert recovered_msg.translation.z == pytest.approx(3.0)
        assert recovered_msg.rotation.w == pytest.approx(1.0)

    def test_twist_roundtrip(self):
        """Test Twist roundtrip."""
        topic = geometry.TwistTopic(topic_name="/cmd_vel", qos=DEFAULT_QOS)

        original_msg = Twist()
        original_msg.linear.x = 1.0
        original_msg.linear.y = 0.0
        original_msg.linear.z = 0.0
        original_msg.angular.x = 0.0
        original_msg.angular.y = 0.0
        original_msg.angular.z = 0.5

        tensor = topic.to_tensor(original_msg)

        assert tensor.shape == (6,)

        recovered_msg = topic.from_tensor(tensor)

        assert recovered_msg.linear.x == pytest.approx(1.0)
        assert recovered_msg.angular.z == pytest.approx(0.5)

    def test_accel_roundtrip(self):
        """Test Accel roundtrip."""
        topic = geometry.AccelTopic(topic_name="/test/accel", qos=DEFAULT_QOS)

        original_msg = Accel()
        original_msg.linear.x = 9.8
        original_msg.linear.y = 0.0
        original_msg.linear.z = 0.0
        original_msg.angular.x = 0.1
        original_msg.angular.y = 0.2
        original_msg.angular.z = 0.3

        tensor = topic.to_tensor(original_msg)
        recovered_msg = topic.from_tensor(tensor)

        assert recovered_msg.linear.x == pytest.approx(9.8)
        assert recovered_msg.angular.z == pytest.approx(0.3)

    def test_wrench_roundtrip(self):
        """Test Wrench roundtrip."""
        topic = geometry.WrenchTopic(topic_name="/test/wrench", qos=DEFAULT_QOS)

        original_msg = Wrench()
        original_msg.force.x = 10.0
        original_msg.force.y = 20.0
        original_msg.force.z = 30.0
        original_msg.torque.x = 1.0
        original_msg.torque.y = 2.0
        original_msg.torque.z = 3.0

        tensor = topic.to_tensor(original_msg)
        recovered_msg = topic.from_tensor(tensor)

        assert recovered_msg.force.x == pytest.approx(10.0)
        assert recovered_msg.force.y == pytest.approx(20.0)
        assert recovered_msg.force.z == pytest.approx(30.0)
        assert recovered_msg.torque.x == pytest.approx(1.0)
        assert recovered_msg.torque.y == pytest.approx(2.0)
        assert recovered_msg.torque.z == pytest.approx(3.0)

    def test_pose_stamped_roundtrip(self):
        """Test PoseStamped roundtrip (header ignored)."""
        topic = geometry.PoseStampedTopic(
            topic_name="/test/pose_stamped", qos=DEFAULT_QOS
        )

        original_msg = PoseStamped()
        original_msg.pose.position.x = 1.0
        original_msg.pose.position.y = 2.0
        original_msg.pose.position.z = 3.0
        original_msg.pose.orientation.w = 1.0

        tensor = topic.to_tensor(original_msg)
        recovered_msg = topic.from_tensor(tensor)

        assert recovered_msg.pose.position.x == pytest.approx(1.0)
        assert recovered_msg.pose.orientation.w == pytest.approx(1.0)

    def test_twist_stamped_roundtrip(self):
        """Test TwistStamped roundtrip (header ignored)."""
        topic = geometry.TwistStampedTopic(
            topic_name="/test/twist_stamped", qos=DEFAULT_QOS
        )

        original_msg = TwistStamped()
        original_msg.twist.linear.x = 1.0
        original_msg.twist.angular.z = 0.5

        tensor = topic.to_tensor(original_msg)
        recovered_msg = topic.from_tensor(tensor)

        assert recovered_msg.twist.linear.x == pytest.approx(1.0)
        assert recovered_msg.twist.angular.z == pytest.approx(0.5)

    def test_feature_description(self):
        """Test feature description for geometry msgs."""
        topic = geometry.TwistTopic(topic_name="/cmd_vel", qos=DEFAULT_QOS)

        desc = topic.feature_description()

        assert desc["dtype"] == "float32"
        assert desc["shape"] == (6,)
        assert "cmd_vel.linear.x" in desc["names"]
        assert "cmd_vel.angular.z" in desc["names"]


# =============================================================================
# sensor_msgs tests
# =============================================================================


class TestSensorMsgs:
    """Test sensor_msgs conversions."""

    def test_temperature_roundtrip(self):
        """Test Temperature roundtrip."""
        topic = sensor.TemperatureTopic(topic_name="/test/temp", qos=DEFAULT_QOS)

        original_msg = Temperature()
        original_msg.temperature = 25.5
        original_msg.variance = 0.1

        tensor = topic.to_tensor(original_msg)

        assert tensor.shape == (2,)

        recovered_msg = topic.from_tensor(tensor)

        assert recovered_msg.temperature == pytest.approx(25.5)
        assert recovered_msg.variance == pytest.approx(0.1)

    def test_relative_humidity_roundtrip(self):
        """Test RelativeHumidity roundtrip."""
        topic = sensor.RelativeHumidityTopic(
            topic_name="/test/humidity", qos=DEFAULT_QOS
        )

        original_msg = RelativeHumidity()
        original_msg.relative_humidity = 0.65
        original_msg.variance = 0.01

        tensor = topic.to_tensor(original_msg)
        recovered_msg = topic.from_tensor(tensor)

        assert recovered_msg.relative_humidity == pytest.approx(0.65)
        assert recovered_msg.variance == pytest.approx(0.01)

    def test_fluid_pressure_roundtrip(self):
        """Test FluidPressure roundtrip."""
        topic = sensor.FluidPressureTopic(topic_name="/test/pressure", qos=DEFAULT_QOS)

        original_msg = FluidPressure()
        original_msg.fluid_pressure = 101325.0  # 1 atm in Pa
        original_msg.variance = 100.0

        tensor = topic.to_tensor(original_msg)
        recovered_msg = topic.from_tensor(tensor)

        assert recovered_msg.fluid_pressure == pytest.approx(101325.0)
        assert recovered_msg.variance == pytest.approx(100.0)

    def test_illuminance_roundtrip(self):
        """Test Illuminance roundtrip."""
        topic = sensor.IlluminanceTopic(topic_name="/test/light", qos=DEFAULT_QOS)

        original_msg = Illuminance()
        original_msg.illuminance = 500.0  # lux
        original_msg.variance = 10.0

        tensor = topic.to_tensor(original_msg)
        recovered_msg = topic.from_tensor(tensor)

        assert recovered_msg.illuminance == pytest.approx(500.0)
        assert recovered_msg.variance == pytest.approx(10.0)

    def test_range_roundtrip(self):
        """Test Range roundtrip."""
        topic = sensor.RangeTopic(topic_name="/test/range", qos=DEFAULT_QOS)

        original_msg = Range()
        original_msg.radiation_type = 0  # ULTRASOUND
        original_msg.field_of_view = 0.5
        original_msg.min_range = 0.1
        original_msg.max_range = 10.0
        original_msg.range = 2.5

        tensor = topic.to_tensor(original_msg)

        assert tensor.shape == (5,)

        recovered_msg = topic.from_tensor(tensor)

        assert recovered_msg.radiation_type == 0
        assert recovered_msg.field_of_view == pytest.approx(0.5)
        assert recovered_msg.min_range == pytest.approx(0.1)
        assert recovered_msg.max_range == pytest.approx(10.0)
        assert recovered_msg.range == pytest.approx(2.5)

    def test_magnetic_field_roundtrip(self):
        """Test MagneticField roundtrip."""
        topic = sensor.MagneticFieldTopic(topic_name="/test/mag", qos=DEFAULT_QOS)

        original_msg = MagneticField()
        original_msg.magnetic_field.x = 0.00002
        original_msg.magnetic_field.y = 0.00001
        original_msg.magnetic_field.z = 0.00005

        tensor = topic.to_tensor(original_msg)

        # 3 field components + 9 covariance
        assert tensor.shape == (12,)

        recovered_msg = topic.from_tensor(tensor)

        assert recovered_msg.magnetic_field.x == pytest.approx(0.00002, rel=1e-4)
        assert recovered_msg.magnetic_field.y == pytest.approx(0.00001, rel=1e-4)
        assert recovered_msg.magnetic_field.z == pytest.approx(0.00005, rel=1e-4)

    def test_nav_sat_status_roundtrip(self):
        """Test NavSatStatus roundtrip."""
        topic = sensor.NavSatStatusTopic(topic_name="/test/gps_status", qos=DEFAULT_QOS)

        original_msg = NavSatStatus()
        original_msg.status = 0  # STATUS_FIX
        original_msg.service = 1  # SERVICE_GPS

        tensor = topic.to_tensor(original_msg)

        assert tensor.shape == (2,)

        recovered_msg = topic.from_tensor(tensor)

        assert recovered_msg.status == 0
        assert recovered_msg.service == 1

    def test_imu_roundtrip(self):
        """Test IMU roundtrip."""
        topic = sensor.ImuTopic(topic_name="/test/imu", qos=DEFAULT_QOS)

        original_msg = Imu()
        # Orientation (quaternion)
        original_msg.orientation.x = 0.0
        original_msg.orientation.y = 0.0
        original_msg.orientation.z = 0.0
        original_msg.orientation.w = 1.0
        # Angular velocity
        original_msg.angular_velocity.x = 0.1
        original_msg.angular_velocity.y = 0.2
        original_msg.angular_velocity.z = 0.3
        # Linear acceleration
        original_msg.linear_acceleration.x = 0.0
        original_msg.linear_acceleration.y = 0.0
        original_msg.linear_acceleration.z = 9.8

        tensor = topic.to_tensor(original_msg)

        # 4 orientation + 3 angular_vel + 3 linear_accel + 3*9 covariances = 37
        assert tensor.shape == (37,)

        recovered_msg = topic.from_tensor(tensor)

        assert recovered_msg.orientation.w == pytest.approx(1.0)
        assert recovered_msg.angular_velocity.x == pytest.approx(0.1)
        assert recovered_msg.angular_velocity.y == pytest.approx(0.2)
        assert recovered_msg.angular_velocity.z == pytest.approx(0.3)
        assert recovered_msg.linear_acceleration.z == pytest.approx(9.8)


# =============================================================================
# Error handling tests
# =============================================================================


class TestErrorHandling:
    """Test error handling in conversions."""

    def test_wrong_tensor_shape_geometry(self):
        """Test that wrong tensor shape raises ValueError."""
        topic = geometry.PointTopic(topic_name="/test/point", qos=DEFAULT_QOS)

        wrong_tensor = torch.tensor([1.0, 2.0])  # Shape (2,) instead of (3,)

        with pytest.raises(ValueError, match="shape"):
            topic.from_tensor(wrong_tensor)

    def test_wrong_tensor_shape_multi_array(self):
        """Test that wrong tensor shape raises ValueError for MultiArray."""
        topic = std.Float64MultiArrayTopic(
            names=["a", "b", "c"],
            topic_name="/test/array",
            qos=DEFAULT_QOS,
        )

        wrong_tensor = torch.tensor([1.0, 2.0])  # Shape (2,) instead of (3,)

        with pytest.raises(ValueError, match="shape"):
            topic.from_tensor(wrong_tensor)


# =============================================================================
# Topic name and feature tests
# =============================================================================


class TestTopicNaming:
    """Test topic name handling."""

    def test_clean_topic_name_leading_slash(self):
        """Test that leading slash is removed."""
        topic = geometry.PointTopic(topic_name="/robot/position", qos=DEFAULT_QOS)

        assert topic.topic_name == "robot.position"

    def test_clean_topic_name_nested(self):
        """Test that nested slashes are converted to dots."""
        topic = geometry.TwistTopic(topic_name="/robot/arm/cmd_vel", qos=DEFAULT_QOS)

        assert topic.topic_name == "robot.arm.cmd_vel"

    def test_size_calculation(self):
        """Test size() method."""
        topic = geometry.PoseTopic(topic_name="/test/pose", qos=DEFAULT_QOS)

        assert topic.size() == 7  # 3 position + 4 orientation

    def test_is_image_false(self):
        """Test that non-image topics return False for is_image()."""
        topic = geometry.PointTopic(topic_name="/test/point", qos=DEFAULT_QOS)

        assert topic.is_image() is False


# =============================================================================
# Registration tests
# =============================================================================


class TestRegistration:
    """Test that topic classes are registered in BaseTopic.MAPPINGS."""

    def test_geometry_msgs_registered(self):
        """Test that geometry_msgs topic classes are registered."""
        from lerobot_ros.convert.base import BaseTopic

        # Point, Vector3, Quaternion, etc. should be registered
        assert "Point" in BaseTopic.MAPPINGS
        assert "Vector3" in BaseTopic.MAPPINGS
        assert "Quaternion" in BaseTopic.MAPPINGS
        assert "Pose" in BaseTopic.MAPPINGS
        assert "Twist" in BaseTopic.MAPPINGS

    def test_std_msgs_registered(self):
        """Test that std_msgs topic classes are registered."""
        from lerobot_ros.convert.base import BaseTopic

        assert "Int32" in BaseTopic.MAPPINGS
        assert "Float32" in BaseTopic.MAPPINGS
        assert "Float64" in BaseTopic.MAPPINGS
        assert "Bool" in BaseTopic.MAPPINGS

    def test_sensor_msgs_registered(self):
        """Test that sensor_msgs topic classes are registered."""
        from lerobot_ros.convert.base import BaseTopic

        assert "Imu" in BaseTopic.MAPPINGS
        assert "Temperature" in BaseTopic.MAPPINGS
        assert "Range" in BaseTopic.MAPPINGS


# =============================================================================
# Image conversion tests
# =============================================================================


class TestImageConversion:
    """Test image message conversions."""

    def _create_ros_image(self, width, height, encoding="rgb8", fill_value=None):
        """Helper to create a ROS Image message with test data."""
        import numpy as np
        from sensor_msgs.msg import Image

        msg = Image()
        msg.width = width
        msg.height = height
        msg.encoding = encoding

        if encoding == "rgb8":
            channels = 3
            bytes_per_pixel = 3
            if fill_value is None:
                # Create a gradient pattern for testing
                data = np.zeros((height, width, channels), dtype=np.uint8)
                data[:, :, 0] = np.arange(width) % 256  # Red gradient
                data[:, :, 1] = np.arange(height).reshape(-1, 1) % 256  # Green gradient
                data[:, :, 2] = 128  # Blue constant
            else:
                data = np.full((height, width, channels), fill_value, dtype=np.uint8)
            msg.step = width * bytes_per_pixel
            msg.data = data.tobytes()

        elif encoding == "bgr8":
            channels = 3
            bytes_per_pixel = 3
            if fill_value is None:
                data = np.zeros((height, width, channels), dtype=np.uint8)
                data[:, :, 2] = np.arange(width) % 256  # Red (BGR order)
                data[:, :, 1] = np.arange(height).reshape(-1, 1) % 256  # Green
                data[:, :, 0] = 128  # Blue
            else:
                data = np.full((height, width, channels), fill_value, dtype=np.uint8)
            msg.step = width * bytes_per_pixel
            msg.data = data.tobytes()

        elif encoding == "mono8":
            if fill_value is None:
                data = np.zeros((height, width), dtype=np.uint8)
                data[:, :] = np.arange(width) % 256
            else:
                data = np.full((height, width), fill_value, dtype=np.uint8)
            msg.step = width
            msg.data = data.tobytes()

        elif encoding == "rgba8":
            channels = 4
            bytes_per_pixel = 4
            if fill_value is None:
                data = np.zeros((height, width, channels), dtype=np.uint8)
                data[:, :, 0] = 100  # Red
                data[:, :, 1] = 150  # Green
                data[:, :, 2] = 200  # Blue
                data[:, :, 3] = 255  # Alpha
            else:
                data = np.full((height, width, channels), fill_value, dtype=np.uint8)
            msg.step = width * bytes_per_pixel
            msg.data = data.tobytes()

        else:
            raise ValueError(f"Unsupported encoding in test helper: {encoding}")

        msg.is_bigendian = 0
        return msg

    def _create_compressed_image(self, width, height, format="jpeg"):
        """Helper to create a ROS CompressedImage message."""
        from io import BytesIO

        import numpy as np
        from PIL import Image as PILImage
        from sensor_msgs.msg import CompressedImage

        # Create a test image
        data = np.zeros((height, width, 3), dtype=np.uint8)
        data[:, :, 0] = 100  # Red
        data[:, :, 1] = 150  # Green
        data[:, :, 2] = 200  # Blue

        pil_img = PILImage.fromarray(data, "RGB")

        # Compress to bytes
        buffer = BytesIO()
        if format == "jpeg":
            pil_img.save(buffer, format="JPEG", quality=95)
        elif format == "png":
            pil_img.save(buffer, format="PNG")
        else:
            raise ValueError(f"Unsupported format: {format}")

        msg = CompressedImage()
        msg.format = format
        msg.data = buffer.getvalue()

        return msg, data

    def test_image_topic_to_tensor_rgb8(self):
        """Test ImageTopic converts RGB8 ROS Image to tensor."""
        from lerobot_ros.convert import image

        width, height = 64, 48
        topic = image.ImageTopic(
            height=height,
            width=width,
            channels=3,
            topic_name="/camera/image",
            qos=DEFAULT_QOS,
        )

        ros_msg = self._create_ros_image(width, height, encoding="rgb8")
        tensor = topic.to_tensor(ros_msg)

        # Verify tensor properties
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.uint8
        assert tensor.shape == (height, width, 3)  # HxWxC

    def test_image_topic_to_tensor_bgr8(self):
        """Test ImageTopic converts BGR8 ROS Image to tensor (with BGR->RGB conversion)."""
        from lerobot_ros.convert import image

        width, height = 32, 32
        topic = image.ImageTopic(
            height=height,
            width=width,
            channels=3,
            topic_name="/camera/image",
            qos=DEFAULT_QOS,
        )

        ros_msg = self._create_ros_image(width, height, encoding="bgr8")
        tensor = topic.to_tensor(ros_msg)

        assert tensor.dtype == torch.uint8
        assert tensor.shape == (height, width, 3)

    def test_image_topic_to_tensor_mono8(self):
        """Test ImageTopic converts mono8 grayscale ROS Image to RGB tensor."""
        from lerobot_ros.convert import image

        width, height = 64, 64
        topic = image.ImageTopic(
            height=height,
            width=width,
            channels=3,
            topic_name="/camera/mono",
            qos=DEFAULT_QOS,
        )

        ros_msg = self._create_ros_image(width, height, encoding="mono8")
        tensor = topic.to_tensor(ros_msg)

        assert tensor.dtype == torch.uint8
        assert tensor.shape == (height, width, 3)  # Mono is converted to RGB

    def test_image_topic_resize(self):
        """Test ImageTopic resizes image to expected dimensions."""
        from lerobot_ros.convert import image

        # Create image with different size than expected
        src_width, src_height = 128, 96
        target_width, target_height = 64, 48

        topic = image.ImageTopic(
            height=target_height,
            width=target_width,
            channels=3,
            topic_name="/camera/image",
            qos=DEFAULT_QOS,
        )

        ros_msg = self._create_ros_image(src_width, src_height, encoding="rgb8")
        tensor = topic.to_tensor(ros_msg)

        # Should be resized to target dimensions
        assert tensor.shape == (target_height, target_width, 3)

    def test_image_topic_rotation(self):
        """Test ImageTopic rotates image correctly."""
        from lerobot_ros.convert import image

        width, height = 64, 48

        # Create topic with 90 degree rotation
        topic = image.ImageTopic(
            height=width,  # After 90° rotation, dimensions swap
            width=height,
            channels=3,
            rotate=90,
            topic_name="/camera/image",
            qos=DEFAULT_QOS,
        )

        ros_msg = self._create_ros_image(width, height, encoding="rgb8")
        tensor = topic.to_tensor(ros_msg)

        # After 90° rotation, width and height should be swapped
        assert tensor.shape == (width, height, 3)

    def test_image_topic_feature_description(self):
        """Test ImageTopic feature description."""
        from lerobot_ros.convert import image

        topic = image.ImageTopic(
            height=480,
            width=640,
            channels=3,
            topic_name="/camera/rgb",
            qos=DEFAULT_QOS,
        )

        desc = topic.feature_description()

        assert desc["dtype"] == "video"
        assert desc["shape"] == (480, 640, 3)
        assert "camera.rgb.height" in desc["names"]
        assert "camera.rgb.width" in desc["names"]
        assert "camera.rgb.channels" in desc["names"]

    def test_image_topic_is_image(self):
        """Test that ImageTopic returns True for is_image()."""
        from lerobot_ros.convert import image

        topic = image.ImageTopic(
            height=480,
            width=640,
            channels=3,
            topic_name="/camera/image",
            qos=DEFAULT_QOS,
        )

        assert topic.is_image() is True

    def test_compressed_image_to_tensor_jpeg(self):
        """Test ImageCompressedTopic converts JPEG compressed image to tensor."""
        from lerobot_ros.convert import image

        width, height = 64, 48

        topic = image.ImageCompressedTopic(
            height=height,
            width=width,
            channels=3,
            topic_name="/camera/compressed",
            qos=DEFAULT_QOS,
        )

        ros_msg, original_data = self._create_compressed_image(
            width, height, format="jpeg"
        )
        tensor = topic.to_tensor(ros_msg)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.uint8
        assert tensor.shape == (height, width, 3)

        # JPEG is lossy, so we can't do exact comparison, but values should be close
        # Check that the general color is preserved (within JPEG tolerance)
        mean_r = tensor[:, :, 0].float().mean().item()
        mean_g = tensor[:, :, 1].float().mean().item()
        mean_b = tensor[:, :, 2].float().mean().item()

        # Original colors were R=100, G=150, B=200
        assert abs(mean_r - 100) < 20  # JPEG tolerance
        assert abs(mean_g - 150) < 20
        assert abs(mean_b - 200) < 20

    def test_compressed_image_to_tensor_png(self):
        """Test ImageCompressedTopic converts PNG compressed image to tensor."""
        import numpy as np

        from lerobot_ros.convert import image

        width, height = 32, 32

        topic = image.ImageCompressedTopic(
            height=height,
            width=width,
            channels=3,
            topic_name="/camera/compressed",
            qos=DEFAULT_QOS,
        )

        ros_msg, original_data = self._create_compressed_image(
            width, height, format="png"
        )
        tensor = topic.to_tensor(ros_msg)

        assert tensor.dtype == torch.uint8
        assert tensor.shape == (height, width, 3)

        # PNG is lossless, so values should match exactly
        np.testing.assert_array_equal(tensor.numpy(), original_data)

    def test_compressed_image_resize(self):
        """Test ImageCompressedTopic resizes image to expected dimensions."""
        from lerobot_ros.convert import image

        src_width, src_height = 128, 96
        target_width, target_height = 64, 48

        topic = image.ImageCompressedTopic(
            height=target_height,
            width=target_width,
            channels=3,
            topic_name="/camera/compressed",
            qos=DEFAULT_QOS,
        )

        ros_msg, _ = self._create_compressed_image(src_width, src_height, format="png")
        tensor = topic.to_tensor(ros_msg)

        assert tensor.shape == (target_height, target_width, 3)

    def test_compressed_image_rotation(self):
        """Test ImageCompressedTopic rotates image correctly."""
        from lerobot_ros.convert import image

        width, height = 64, 48

        topic = image.ImageCompressedTopic(
            height=width,  # After 90° rotation
            width=height,
            channels=3,
            rotate=90,
            topic_name="/camera/compressed",
            qos=DEFAULT_QOS,
        )

        ros_msg, _ = self._create_compressed_image(width, height, format="png")
        tensor = topic.to_tensor(ros_msg)

        assert tensor.shape == (width, height, 3)

    def test_image_topic_registered(self):
        """Test that image topic classes are registered."""
        from lerobot_ros.convert.base import BaseTopic

        assert "Image" in BaseTopic.MAPPINGS
        assert "CompressedImage" in BaseTopic.MAPPINGS

    @pytest.mark.parametrize(
        "width,height,channels",
        [
            (640, 480, 3),
            (1920, 1080, 3),
            (320, 240, 3),
            (64, 64, 3),
        ],
    )
    def test_various_image_sizes(self, width, height, channels):
        """Test ImageTopic with various image sizes."""
        from lerobot_ros.convert import image

        topic = image.ImageTopic(
            height=height,
            width=width,
            channels=channels,
            topic_name="/camera/image",
            qos=DEFAULT_QOS,
        )

        ros_msg = self._create_ros_image(width, height, encoding="rgb8")
        tensor = topic.to_tensor(ros_msg)

        assert tensor.shape == (height, width, channels)
        assert tensor.dtype == torch.uint8

    def test_image_with_row_padding(self):
        """Test ImageTopic handles images with row padding (step > width * bytes_per_pixel)."""
        import numpy as np
        from sensor_msgs.msg import Image

        from lerobot_ros.convert import image

        width, height = 63, 48  # Odd width to likely cause padding
        channels = 3

        topic = image.ImageTopic(
            height=height,
            width=width,
            channels=channels,
            topic_name="/camera/image",
            qos=DEFAULT_QOS,
        )

        # Create image with padding (step aligned to 4 bytes)
        bytes_per_pixel = 3
        step = ((width * bytes_per_pixel + 3) // 4) * 4  # Align to 4 bytes

        msg = Image()
        msg.width = width
        msg.height = height
        msg.encoding = "rgb8"
        msg.step = step
        msg.is_bigendian = 0

        # Create padded data
        padded_data = np.zeros((height, step), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                padded_data[y, x * 3 : (x + 1) * 3] = [100, 150, 200]  # RGB values

        msg.data = padded_data.tobytes()

        tensor = topic.to_tensor(msg)

        assert tensor.shape == (height, width, channels)

        # Verify pixel values are correct (not corrupted by padding)
        assert tensor[0, 0, 0].item() == 100  # R
        assert tensor[0, 0, 1].item() == 150  # G
        assert tensor[0, 0, 2].item() == 200  # B
