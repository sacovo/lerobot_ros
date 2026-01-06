"""
Converters for geometry_msgs types.

Geometry messages are more diverse than std_msgs, so we define explicit field
layouts for each message type. The factory function handles the boilerplate.
"""

from __future__ import annotations

import geometry_msgs.msg as geometry_msgs

from .base import generate_topic_classes_from_layouts


# Field layouts for geometry messages
# Each entry: message_class -> list of field paths
# field_path can be nested like "linear.x" or "pose.position.x"
_GEOMETRY_LAYOUTS: dict[str, list[str]] = {
    # Basic primitives
    "Vector3": ["x", "y", "z"],
    "Point": ["x", "y", "z"],
    "Point32": ["x", "y", "z"],
    "Quaternion": ["x", "y", "z", "w"],
    # Compound types
    "Pose2D": ["x", "y", "theta"],
    "Pose": [
        "position.x",
        "position.y",
        "position.z",
        "orientation.x",
        "orientation.y",
        "orientation.z",
        "orientation.w",
    ],
    "Transform": [
        "translation.x",
        "translation.y",
        "translation.z",
        "rotation.x",
        "rotation.y",
        "rotation.z",
        "rotation.w",
    ],
    "Twist": [
        "linear.x",
        "linear.y",
        "linear.z",
        "angular.x",
        "angular.y",
        "angular.z",
    ],
    "Accel": [
        "linear.x",
        "linear.y",
        "linear.z",
        "angular.x",
        "angular.y",
        "angular.z",
    ],
    "Wrench": ["force.x", "force.y", "force.z", "torque.x", "torque.y", "torque.z"],
    "Inertia": [
        "m",
        "com.x",
        "com.y",
        "com.z",
        "ixx",
        "ixy",
        "ixz",
        "iyy",
        "iyz",
        "izz",
    ],
    # Stamped versions (we ignore the header, only convert the data)
    "PointStamped": ["point.x", "point.y", "point.z"],
    "Vector3Stamped": ["vector.x", "vector.y", "vector.z"],
    "QuaternionStamped": [
        "quaternion.x",
        "quaternion.y",
        "quaternion.z",
        "quaternion.w",
    ],
    "PoseStamped": [
        "pose.position.x",
        "pose.position.y",
        "pose.position.z",
        "pose.orientation.x",
        "pose.orientation.y",
        "pose.orientation.z",
        "pose.orientation.w",
    ],
    "TransformStamped": [
        "transform.translation.x",
        "transform.translation.y",
        "transform.translation.z",
        "transform.rotation.x",
        "transform.rotation.y",
        "transform.rotation.z",
        "transform.rotation.w",
    ],
    "TwistStamped": [
        "twist.linear.x",
        "twist.linear.y",
        "twist.linear.z",
        "twist.angular.x",
        "twist.angular.y",
        "twist.angular.z",
    ],
    "AccelStamped": [
        "accel.linear.x",
        "accel.linear.y",
        "accel.linear.z",
        "accel.angular.x",
        "accel.angular.y",
        "accel.angular.z",
    ],
    "WrenchStamped": [
        "wrench.force.x",
        "wrench.force.y",
        "wrench.force.z",
        "wrench.torque.x",
        "wrench.torque.y",
        "wrench.torque.z",
    ],
    "InertiaStamped": [
        "inertia.m",
        "inertia.com.x",
        "inertia.com.y",
        "inertia.com.z",
        "inertia.ixx",
        "inertia.ixy",
        "inertia.ixz",
        "inertia.iyy",
        "inertia.iyz",
        "inertia.izz",
    ],
    # With covariance
    "PoseWithCovariance": [
        "pose.position.x",
        "pose.position.y",
        "pose.position.z",
        "pose.orientation.x",
        "pose.orientation.y",
        "pose.orientation.z",
        "pose.orientation.w",
        *[f"covariance.{i}" for i in range(36)],
    ],
    "TwistWithCovariance": [
        "twist.linear.x",
        "twist.linear.y",
        "twist.linear.z",
        "twist.angular.x",
        "twist.angular.y",
        "twist.angular.z",
        *[f"covariance.{i}" for i in range(36)],
    ],
    "AccelWithCovariance": [
        "accel.linear.x",
        "accel.linear.y",
        "accel.linear.z",
        "accel.angular.x",
        "accel.angular.y",
        "accel.angular.z",
        *[f"covariance.{i}" for i in range(36)],
    ],
    # Stamped with covariance
    "PoseWithCovarianceStamped": [
        "pose.pose.position.x",
        "pose.pose.position.y",
        "pose.pose.position.z",
        "pose.pose.orientation.x",
        "pose.pose.orientation.y",
        "pose.pose.orientation.z",
        "pose.pose.orientation.w",
        *[f"pose.covariance.{i}" for i in range(36)],
    ],
    "TwistWithCovarianceStamped": [
        "twist.twist.linear.x",
        "twist.twist.linear.y",
        "twist.twist.linear.z",
        "twist.twist.angular.x",
        "twist.twist.angular.y",
        "twist.twist.angular.z",
        *[f"twist.covariance.{i}" for i in range(36)],
    ],
    "AccelWithCovarianceStamped": [
        "accel.accel.linear.x",
        "accel.accel.linear.y",
        "accel.accel.linear.z",
        "accel.accel.angular.x",
        "accel.accel.angular.y",
        "accel.accel.angular.z",
        *[f"accel.covariance.{i}" for i in range(36)],
    ],
}


# Generate all topic classes and inject into module namespace
_generated_classes = generate_topic_classes_from_layouts(
    geometry_msgs, _GEOMETRY_LAYOUTS
)
globals().update(_generated_classes)

# Export all generated class names
__all__ = list(_generated_classes.keys())
