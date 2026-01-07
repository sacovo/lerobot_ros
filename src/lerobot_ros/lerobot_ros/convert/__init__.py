"""
ROS message to tensor conversion utilities.

This module provides converters for various ROS message types to PyTorch tensors
and back. All topic classes are registered in BaseTopic.MAPPINGS for easy lookup.
"""

from torch.distributed.algorithms.join import Join

from .base import (
    BaseTopic,
    clean_topic_name,
    generate_topic_classes_from_layouts,
    get_nested_attr,
    make_layout_topic,
    prefix_names,
    set_nested_attr,
)

# Import all topic modules to register them in BaseTopic.MAPPINGS
# Use try/except to handle missing ROS message packages gracefully

# std_msgs - usually always available
try:
    from . import std
except ImportError:
    std = None

# geometry_msgs - usually always available
try:
    from . import geometry
except ImportError:
    geometry = None

# sensor_msgs - usually always available
try:
    from . import sensor
except ImportError:
    sensor = None

from .sensor import JointStateTopic

# Image topics (sensor_msgs.Image, sensor_msgs.CompressedImage)
try:
    from . import image
except ImportError:
    image = None


try:
    from . import fhnw
except ImportError:
    fhnw = None


__all__ = [
    # Base classes and utilities
    "BaseTopic",
    "JointStateTopic",
    "clean_topic_name",
    "prefix_names",
    "get_nested_attr",
    "set_nested_attr",
    "make_layout_topic",
    "generate_topic_classes_from_layouts",
    # Submodules (may be None if import failed)
    "std",
    "geometry",
    "sensor",
    "image",
]

print(sorted(BaseTopic.MAPPINGS.keys()))  # Debug: print registered topic classes
