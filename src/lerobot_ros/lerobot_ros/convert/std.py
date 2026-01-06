from __future__ import annotations

import re
from std_msgs import msg as std_msgs
import torch
from .base import BaseTopic, prefix_names


# Mapping from type name patterns to torch dtypes
_TORCH_DTYPE_MAP: dict[str, torch.dtype] = {
    "bool": torch.bool,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "uint16": torch.int32,  # torch has no uint16
    "uint32": torch.int64,  # torch has no uint32
    "uint64": torch.int64,  # torch has no uint64
    "float32": torch.float32,
    "float64": torch.float64,
}

# Regex to extract dtype from message class name (e.g., "Int32" -> "int32", "Float64MultiArray" -> "float64")
_DTYPE_PATTERN = re.compile(r"^(Bool|U?Int\d+|Float\d+)", re.IGNORECASE)


def _get_dtype_info(msg_name: str) -> tuple[str, torch.dtype] | None:
    """Extract dtype string and torch dtype from a message class name."""
    match = _DTYPE_PATTERN.match(msg_name)
    if not match:
        return None
    dtype_str = match.group(1).lower()
    if dtype_str not in _TORCH_DTYPE_MAP:
        return None
    return dtype_str, _TORCH_DTYPE_MAP[dtype_str]


def _make_scalar_topic(msg_cls, dtype: str, torch_dtype: torch.dtype):
    """Factory to create a scalar topic class for a given ROS message type."""

    class ScalarTopic(BaseTopic):
        _msg_cls = msg_cls
        _dtype = dtype
        _torch_dtype = torch_dtype

        @staticmethod
        def msg_type():
            return msg_cls

        def feature_description(self):
            return {"dtype": self._dtype, "shape": (1,), "names": [self.topic_name]}

        def to_tensor(self, msg) -> torch.Tensor:
            return torch.tensor([msg.data], dtype=self._torch_dtype)

        def from_tensor(self, tensor: torch.Tensor):
            if tensor.shape != ():
                raise ValueError(
                    f"Tensor must be scalar for {msg_cls.__name__} conversion."
                )
            msg = self._msg_cls()
            msg.data = tensor.item()
            return msg

    ScalarTopic.__name__ = f"{msg_cls.__name__}Topic"
    ScalarTopic.__qualname__ = f"{msg_cls.__name__}Topic"
    return ScalarTopic


def _make_multi_array_topic(msg_cls, dtype: str, torch_dtype: torch.dtype):
    """Factory to create a multi-array topic class for a given ROS message type."""

    class MultiArrayTopic(BaseTopic):
        _msg_cls = msg_cls
        _dtype = dtype
        _torch_dtype = torch_dtype

        def __init__(self, names, **kwargs):
            super().__init__(**kwargs)
            self.names = names

        @staticmethod
        def msg_type():
            return msg_cls

        def feature_description(self):
            return {
                "dtype": self._dtype,
                "shape": (len(self.names),),
                "names": prefix_names(self.names, self.topic_name),
            }

        def to_tensor(self, msg) -> torch.Tensor:
            return torch.tensor(msg.data, dtype=self._torch_dtype)

        def from_tensor(self, tensor: torch.Tensor):
            if tensor.shape != (len(self.names),):
                raise ValueError(
                    f"Tensor must have the same shape as the names for {msg_cls.__name__} conversion. "
                    f"{self.names} ({len(self.names)} != {tensor.shape})"
                )
            msg = self._msg_cls()
            msg.data = tensor.tolist()
            return msg

    MultiArrayTopic.__name__ = f"{msg_cls.__name__}Topic"
    MultiArrayTopic.__qualname__ = f"{msg_cls.__name__}Topic"
    return MultiArrayTopic


def _generate_topic_classes() -> dict[str, type]:
    """Dynamically generate topic classes for all compatible std_msgs types."""
    generated = {}

    for name in dir(std_msgs):
        if name.startswith("_"):
            continue

        msg_cls = getattr(std_msgs, name)
        if not isinstance(msg_cls, type):
            continue

        # Check if this message has a 'data' field (required for our converters)
        if not hasattr(msg_cls, "get_fields_and_field_types"):
            continue
        fields = msg_cls.get_fields_and_field_types()
        if "data" not in fields:
            continue

        dtype_info = _get_dtype_info(name)
        if dtype_info is None:
            continue

        dtype_str, torch_dtype = dtype_info
        topic_name = f"{name}Topic"

        if name.endswith("MultiArray"):
            generated[topic_name] = _make_multi_array_topic(
                msg_cls, dtype_str, torch_dtype
            )
        else:
            generated[topic_name] = _make_scalar_topic(msg_cls, dtype_str, torch_dtype)

    return generated


# Generate all topic classes and inject into module namespace
_generated_classes = _generate_topic_classes()
globals().update(_generated_classes)

# Export all generated class names
__all__ = list(_generated_classes.keys())
