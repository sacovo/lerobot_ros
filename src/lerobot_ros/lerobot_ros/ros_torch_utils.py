from typing import Any, Dict

import torch

from .convert.base import BaseTopic


def prepare_frame(observation, device="cpu"):
    """Batch and preprocess the observation frame for model input."""
    for name in list(observation.keys()):
        tensor = observation[name]
        if not isinstance(tensor, torch.Tensor):
            continue

        # Move to target device first to run subsequent operations on the GPU/accelerator
        t = tensor.to(device)
        if "image" in name:
            t = t.to(torch.float32) / 255.0
            t = t.permute(2, 0, 1).contiguous()
        t = t.unsqueeze(0)
        observation[name] = t
    return observation


class TensorToRosConverter:
    def __init__(self, topics: Dict[str, BaseTopic]):
        self.topics = {
            name: topic for (name, topic) in topics.items() if topic.is_action
        }
        self.sizes = {
            topic: topic.size() for topic in topics.values() if topic.is_action
        }

    def convert(self, tensor) -> dict[str, Any]:
        pos = 0
        msgs = {}
        for name, topic in self.topics.items():
            size = self.sizes[topic]
            sub_tensor = tensor[pos : pos + size]
            if not isinstance(sub_tensor, torch.Tensor):
                sub_tensor = torch.tensor(sub_tensor)
            if getattr(topic, "round_to_nearest", False):
                sub_tensor = torch.round(sub_tensor)
            sub_tensor = topic.apply_limits_and_rounding(sub_tensor)
            msg = topic.from_tensor(sub_tensor)
            msgs[name] = msg
            pos += size

        return msgs
