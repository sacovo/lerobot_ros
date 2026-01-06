from __future__ import annotations

import math
from types import ModuleType

import torch


def clean_topic_name(topic_name: str) -> str:
    if topic_name.startswith("/"):
        topic_name = topic_name[1:]
    return topic_name.replace("/", ".")


class BaseTopic:
    MAPPINGS = {}

    def __init__(self, topic_name, qos, is_action=False):
        self.is_action = is_action
        self.topic_name = clean_topic_name(topic_name)
        self.qos = qos

    def __init_subclass__(cls):
        """Register the subclass in the mappings."""
        print(f"Registering topic class: {cls.__name__}")
        super().__init_subclass__()
        if cls not in BaseTopic.MAPPINGS:
            BaseTopic.MAPPINGS[cls.msg_type().__name__] = cls

    def is_image(self):
        return False

    def feature_description(self):
        """Return the feature description for this topic."""
        raise NotImplementedError("Subclasses should implement this method.")

    def to_tensor(self, msg):
        """Convert the ROS message to a tensor."""
        raise NotImplementedError("Subclasses should implement this method.")

    def from_tensor(self, tensor):
        """Convert a tensor back to a ROS message."""
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def msg_type():
        raise NotImplementedError("Subclasses should implement this method.")

    def size(self) -> int:
        shape = self.feature_description()["shape"]
        return math.prod(shape)


def prefix_names(names: list[str], prefix: str) -> list[str]:
    return [f"{prefix}.{name}" for name in names]


def get_nested_attr(obj, path: str):
    """Get a nested attribute using dot notation, supporting array indices.

    Examples:
        get_nested_attr(msg, "position.x")
        get_nested_attr(msg, "covariance.5")
    """
    for part in path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def set_nested_attr(obj, path: str, value):
    """Set a nested attribute using dot notation, supporting array indices.

    Examples:
        set_nested_attr(msg, "position.x", 1.0)
        set_nested_attr(msg, "covariance.5", 0.1)
    """
    parts = path.split(".")
    for part in parts[:-1]:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = value
    else:
        setattr(obj, last, value)


def make_layout_topic(
    msg_cls: type,
    fields: list[str],
    dtype: str = "float32",
    torch_dtype: torch.dtype = torch.float32,
) -> type:
    """Factory to create a topic class for a ROS message with a defined field layout.

    Args:
        msg_cls: The ROS message class
        fields: List of field paths (e.g., ["x", "y", "z"] or ["position.x", "position.y"])
        dtype: The dtype string for the feature description
        torch_dtype: The torch dtype to use for tensors

    Returns:
        A new BaseTopic subclass for the given message type
    """

    class LayoutTopic(BaseTopic):
        _msg_cls = msg_cls
        _fields = fields
        _dtype = dtype
        _torch_dtype = torch_dtype

        @staticmethod
        def msg_type():
            return msg_cls

        def feature_description(self):
            return {
                "dtype": self._dtype,
                "shape": (len(self._fields),),
                "names": prefix_names(self._fields, self.topic_name),
            }

        def to_tensor(self, msg) -> torch.Tensor:
            values = [float(get_nested_attr(msg, field)) for field in self._fields]
            return torch.tensor(values, dtype=self._torch_dtype)

        def from_tensor(self, tensor: torch.Tensor):
            if tensor.shape != (len(self._fields),):
                raise ValueError(
                    f"Tensor must have shape ({len(self._fields)},) for {msg_cls.__name__} conversion, "
                    f"got {tensor.shape}"
                )
            msg = self._msg_cls()
            for i, field in enumerate(self._fields):
                set_nested_attr(msg, field, tensor[i].item())
            return msg

    LayoutTopic.__name__ = f"{msg_cls.__name__}Topic"
    LayoutTopic.__qualname__ = f"{msg_cls.__name__}Topic"
    return LayoutTopic


def generate_topic_classes_from_layouts(
    msg_module: ModuleType,
    layouts: dict[str, list[str]],
    dtype: str = "float32",
    torch_dtype: torch.dtype = torch.float32,
) -> dict[str, type]:
    """Generate topic classes for all message types defined in a layout dictionary.

    Args:
        msg_module: The ROS message module (e.g., geometry_msgs.msg)
        layouts: Dictionary mapping message class names to field lists
        dtype: The dtype string for feature descriptions
        torch_dtype: The torch dtype to use for tensors

    Returns:
        Dictionary mapping topic class names to generated classes
    """
    generated = {}

    for msg_name, fields in layouts.items():
        if not hasattr(msg_module, msg_name):
            continue
        msg_cls = getattr(msg_module, msg_name)
        topic_name = f"{msg_name}Topic"
        generated[topic_name] = make_layout_topic(msg_cls, fields, dtype, torch_dtype)

    return generated
