"""
Converters for sensor_msgs types.

Sensor messages are diverse - some have fixed layouts, others have variable-length
arrays. This module handles the common fixed-layout messages.
"""

from __future__ import annotations

import numpy as np
import sensor_msgs.msg as sensor_msgs
import torch

from .base import BaseTopic, generate_topic_classes_from_layouts

# Field layouts for sensor messages
# Each entry: message_class -> list of field paths
# Note: Messages with variable-length arrays (PointCloud2, LaserScan, etc.)
# need custom implementations and are not included here.
_SENSOR_LAYOUTS: dict[str, list[str]] = {
    # IMU
    "Imu": [
        "orientation.x",
        "orientation.y",
        "orientation.z",
        "orientation.w",
        "angular_velocity.x",
        "angular_velocity.y",
        "angular_velocity.z",
        "linear_acceleration.x",
        "linear_acceleration.y",
        "linear_acceleration.z",
        # Covariances (9 elements each)
        *[f"orientation_covariance.{i}" for i in range(9)],
        *[f"angular_velocity_covariance.{i}" for i in range(9)],
        *[f"linear_acceleration_covariance.{i}" for i in range(9)],
    ],
    # Temperature
    "Temperature": ["temperature", "variance"],
    # Relative Humidity
    "RelativeHumidity": ["relative_humidity", "variance"],
    # Fluid Pressure
    "FluidPressure": ["fluid_pressure", "variance"],
    # Illuminance
    "Illuminance": ["illuminance", "variance"],
    # Range (ultrasonic, IR, etc.)
    "Range": [
        "radiation_type",
        "field_of_view",
        "min_range",
        "max_range",
        "range",
    ],
    # Magnetic Field
    "MagneticField": [
        "magnetic_field.x",
        "magnetic_field.y",
        "magnetic_field.z",
        *[f"magnetic_field_covariance.{i}" for i in range(9)],
    ],
    # Battery State
    "BatteryState": [
        "voltage",
        "temperature",
        "current",
        "charge",
        "capacity",
        "design_capacity",
        "percentage",
        "power_supply_status",
        "power_supply_health",
        "power_supply_technology",
        "present",
    ],
    # Joy (Joystick)
    # Note: axes and buttons are variable length, but we can define common layouts
    # For now, skip Joy as it needs special handling
    # NavSatFix (GPS)
    "NavSatFix": [
        "status.status",
        "status.service",
        "latitude",
        "longitude",
        "altitude",
        *[f"position_covariance.{i}" for i in range(9)],
        "position_covariance_type",
    ],
    # NavSatStatus
    "NavSatStatus": ["status", "service"],
    # TimeReference
    "TimeReference": [
        "time_ref.sec",
        "time_ref.nanosec",
    ],
}


class JointStateTopic(BaseTopic):
    def __init__(
        self, joints: list[str], effort: bool, velocity: bool, position: bool, **kwargs
    ):
        super().__init__(**kwargs)

        self.joints = joints
        self.has_effort = effort
        self.has_velocity = velocity
        self.has_position = position

        self.field_count = effort + velocity + position
        self.names = []
        if self.has_position:
            self.names += [f"{joint}.position" for joint in joints]
        if self.has_velocity:
            self.names += [f"{joint}.velocity" for joint in joints]
        if self.has_effort:
            self.names += [f"{joint}.effort" for joint in joints]

    @staticmethod
    def msg_type():
        return sensor_msgs.JointState

    def feature_description(self):
        return {
            "dtype": "float32",
            "shape": (self.field_count * len(self.joints),),
            "names": self.names,
        }

    def to_tensor(self, joint_state: sensor_msgs.JointState) -> torch.Tensor:
        """Convert a ROS JointState message to a PyTorch tensor."""
        values = []
        if self.has_position:
            values.append(np.array(joint_state.position, dtype=np.float32))
        if self.has_velocity:
            values.append(np.array(joint_state.velocity, dtype=np.float32))
        if self.has_effort:
            values.append(np.array(joint_state.effort, dtype=np.float32))
        return torch.tensor(
            np.stack(values, axis=0),
            dtype=torch.float32,
        ).flatten()

    def from_tensor(self, tensor: torch.Tensor) -> sensor_msgs.JointState:
        """Convert a PyTorch tensor to a ROS JointState message."""
        joint_state = sensor_msgs.JointState()
        joint_state.name = self.joints
        tensor = tensor.reshape(self.field_count, len(self.joints))

        i = 0
        if self.has_position:
            joint_state.position = tensor[i, :].tolist()
            i += 1
        if self.has_velocity:
            joint_state.velocity = tensor[i, :].tolist()
            i += 1
        if self.has_effort:
            joint_state.effort = tensor[i, :].tolist()
            i += 1

        return joint_state


# Generate all topic classes and inject into module namespace
_generated_classes = generate_topic_classes_from_layouts(sensor_msgs, _SENSOR_LAYOUTS)
globals().update(_generated_classes)

# Export all generated class names
__all__ = list(_generated_classes.keys())
