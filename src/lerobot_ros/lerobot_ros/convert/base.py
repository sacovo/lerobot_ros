from __future__ import annotations

import math
from types import ModuleType

import torch


def clean_topic_name(topic_name: str) -> str:
    if topic_name.startswith("/"):
        topic_name = topic_name[1:]
    return topic_name.replace("/", ".")


def _get_config_for_name(config_dict, name):
    """Resolve a per-element config value for a (possibly dotted) feature name.

    Looks up the full name, then each dotted part, then any key that the name
    ends/starts with. Returns None if nothing matches or config_dict is not a dict.
    """
    if not isinstance(config_dict, dict):
        return None
    if name in config_dict:
        return config_dict[name]
    parts = name.split(".")
    for part in parts:
        if part in config_dict:
            return config_dict[part]
    for key, val in config_dict.items():
        if name.endswith("." + key) or name.startswith(key + "."):
            return val
    return None


# --- Input transforms ----------------------------------------------------------
# Elementwise, monotonic reshapings applied to *observation* values as they are
# converted ROS -> tensor. Unlike an affine sign-flip (which normalization in the
# training pipeline absorbs for free), these nonlinear maps change how resolution
# is allocated across the value range -- e.g. handing most of the dynamic range to
# small ToF distances so the policy gets larger gradients near contact.
#
# Each entry maps a spec `type` to `fn(tensor, params) -> tensor`. All return
# float tensors regardless of the input dtype.


def _t_exp_decay(t: torch.Tensor, p: dict) -> torch.Tensor:
    """closeness = exp(-value / scale). Bounded in (0, 1]; 0 -> 1, large -> ~0."""
    scale = float(p.get("scale", 1.0))
    return torch.exp(-t.to(torch.float32) / scale)


def _t_reciprocal(t: torch.Tensor, p: dict) -> torch.Tensor:
    """closeness = scale / (value + eps). Unbounded near 0; eps guards div-by-zero."""
    scale = float(p.get("scale", 1.0))
    eps = float(p.get("eps", 1e-6))
    return scale / (t.to(torch.float32) + eps)


_INPUT_TRANSFORMS = {
    "exp_decay": _t_exp_decay,
    "reciprocal": _t_reciprocal,
}


class BaseTopic:
    MAPPINGS = {}

    def __init__(self, topic_name, qos, tag="observation", key=None, **kwargs):
        self.is_action = tag == "action"
        self.is_meta = tag == "meta"
        self.topic_name = clean_topic_name(topic_name)
        self.qos = qos
        self.key = key if key is not None else self.topic_name
        self.round_to_nearest = kwargs.get("round_to_nearest", False)
        self.round_values = kwargs.get("round_values", None)
        self.limits = kwargs.get("limits", None)
        self.transform = kwargs.get("transform", None)

        if self.transform is not None and self.is_action:
            raise ValueError(
                f"'transform' is only supported on observation topics, but topic "
                f"'{self.topic_name}' is tagged as an action. Input transforms are "
                f"nonlinear and have no exact inverse, so they cannot be undone when "
                f"publishing actions back to ROS."
            )

    def __init_subclass__(cls):
        """Register the subclass and wrap its converters for input transforms.

        `to_tensor` is wrapped so that any configured input transform is applied in
        *every* ROS->tensor path (live subscriber, frame assembler, bag recording,
        annotation) without each converter having to opt in -- this guarantees the
        recorded dataset and the inference-time observations use the identical map.
        `feature_description` is wrapped so a transformed topic reports float32
        (transforms always produce floats, even from integer distance messages).
        """
        super().__init_subclass__()
        if cls not in BaseTopic.MAPPINGS:
            BaseTopic.MAPPINGS[cls.msg_type().__name__] = cls

        if "to_tensor" in cls.__dict__:
            raw_to_tensor = cls.__dict__["to_tensor"]

            def to_tensor(self, msg, _raw=raw_to_tensor):
                return self.apply_input_transform(_raw(self, msg))

            cls.to_tensor = to_tensor

        if "feature_description" in cls.__dict__:
            raw_feature_description = cls.__dict__["feature_description"]

            def feature_description(self, _raw=raw_feature_description):
                fd = _raw(self)
                if getattr(self, "transform", None) is not None:
                    fd = {**fd, "dtype": "float32"}
                return fd

            cls.feature_description = feature_description

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

    def apply_input_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the configured input transform to an observation tensor.

        `transform` may be either a single spec dict (``{"type": ..., ...}``)
        applied to every element, or a mapping of feature name -> spec for
        per-element transforms (same name-resolution rules as `limits`).
        """
        transform = getattr(self, "transform", None)
        if transform is None:
            return tensor

        # Single spec: apply to the whole tensor at once (vectorized).
        if isinstance(transform, dict) and "type" in transform:
            return self._apply_transform_spec(tensor, transform)

        # Per-name mapping: apply each element's spec individually.
        if isinstance(transform, dict):
            tensor = tensor.to(torch.float32)
            names = self.feature_description().get("names", [])
            for i in range(len(tensor)):
                name = names[i] if i < len(names) else ""
                spec = _get_config_for_name(transform, name)
                if isinstance(spec, dict) and "type" in spec:
                    tensor[i] = self._apply_transform_spec(tensor[i], spec)
            return tensor

        raise ValueError(
            f"Invalid 'transform' config for topic '{self.topic_name}': expected a "
            f"spec dict with a 'type' key or a name->spec mapping, got {transform!r}."
        )

    def _apply_transform_spec(self, tensor: torch.Tensor, spec: dict) -> torch.Tensor:
        fn = _INPUT_TRANSFORMS.get(spec["type"])
        if fn is None:
            raise ValueError(
                f"Unknown input transform type {spec['type']!r} for topic "
                f"'{self.topic_name}'. Available: {sorted(_INPUT_TRANSFORMS)}."
            )
        return fn(tensor, spec)

    def apply_limits_and_rounding(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply limits and round_values constraints to the tensor."""
        limits = getattr(self, "limits", None)
        round_values = getattr(self, "round_values", None)

        if limits is None and round_values is None:
            return tensor

        tensor = tensor.clone()
        names = self.feature_description().get("names", [])

        def is_limit_range(val):
            return (
                isinstance(val, (list, tuple))
                and len(val) == 2
                and all(x is None or isinstance(x, (int, float)) for x in val)
            )

        def is_list_of_numbers(lst):
            return isinstance(lst, (list, tuple)) and all(isinstance(x, (int, float)) for x in lst)

        for i in range(len(tensor)):
            val = tensor[i].item()
            name = names[i] if i < len(names) else ""

            # 1. Apply limits
            elem_limit = None
            if limits is not None:
                if is_limit_range(limits):
                    elem_limit = limits
                elif isinstance(limits, (list, tuple)):
                    if i < len(limits):
                        elem_limit = limits[i]
                elif isinstance(limits, dict):
                    elem_limit = _get_config_for_name(limits, name)

            if elem_limit is not None and is_limit_range(elem_limit):
                min_val, max_val = elem_limit[0], elem_limit[1]
                if min_val is not None:
                    val = max(min_val, val)
                if max_val is not None:
                    val = min(max_val, val)

            # 2. Apply rounding to specific set of values
            elem_round = None
            if round_values is not None:
                if is_list_of_numbers(round_values):
                    elem_round = round_values
                elif isinstance(round_values, (list, tuple)):
                    if i < len(round_values):
                        elem_round = round_values[i]
                elif isinstance(round_values, dict):
                    elem_round = _get_config_for_name(round_values, name)

            if elem_round is not None and is_list_of_numbers(elem_round):
                val = min(elem_round, key=lambda x: abs(x - val))

            tensor[i] = val

        return tensor



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
