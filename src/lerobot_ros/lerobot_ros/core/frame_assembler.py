import torch
from typing import Dict, Any
from lerobot_ros.convert.base import BaseTopic
from lerobot_ros.convert.image import ImageTopic, ImageCompressedTopic

def key_for_topic(topic: BaseTopic) -> str:
    if isinstance(topic, (ImageTopic, ImageCompressedTopic)):
        return f"observation.images.{topic.key}"
    elif topic.is_meta:
        return f"meta.{topic.key}"
    elif topic.is_action:
        return "action"
    else:
        return "observation.state"

class FrameAssembler:
    def __init__(self, topics: Dict[str, BaseTopic]):
        self.topics = topics

    def get_feature_description(self) -> Dict[str, Any]:
        feature_description = {
            "action": {"dtype": "float32", "shape": (0,), "names": []},
            "observation.state": {"dtype": "float32", "shape": (0,), "names": []},
        }
        for topic in self.topics.values():
            key = key_for_topic(topic)
            if isinstance(topic, (ImageTopic, ImageCompressedTopic)):
                feature_description[key] = {
                    "dtype": "video",
                    "shape": (topic.height, topic.width, topic.channels),
                    "names": ["height", "width", "channels"],
                }
                continue
            if topic.is_meta:
                feature_description[key] = topic.feature_description()
                continue

            combined_feature = feature_description[key]
            feature = topic.feature_description()
            size = combined_feature.get("shape", (0,))[0]
            size += feature.get("shape", (0,))[0]

            combined_feature["shape"] = (size,)
            combined_feature["names"] = combined_feature["names"] + feature["names"]

        if feature_description["observation.state"]["shape"] == (0,):
            del feature_description["observation.state"]
        if feature_description["action"]["shape"] == (0,):
            del feature_description["action"]

        return feature_description

    def assemble(self, latest_msgs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        out_frame = {
            "action": [],
            "observation.state": [],
        }
        ds_frame = {}
        for topic_name, topic in self.topics.items():
            key = key_for_topic(topic)
            
            # Fetch message/tensor
            msg_or_tensor = latest_msgs.get(topic_name)
            
            # Convert list of tensors to last tensor if list is given (subscriber loop passes lists of tensors)
            if isinstance(msg_or_tensor, list):
                if len(msg_or_tensor) == 0:
                    msg_or_tensor = None
                else:
                    msg_or_tensor = msg_or_tensor[-1]
            
            if msg_or_tensor is None:
                # Padding/default tensor
                dtype = topic.feature_description().get("dtype", "float32")
                if dtype in ["video", "image"]:
                    dtype = "uint8"
                tensor = torch.zeros(
                    topic.feature_description().get("shape", (1,)),
                    dtype=getattr(torch, dtype)
                )
            elif isinstance(msg_or_tensor, torch.Tensor):
                tensor = msg_or_tensor
            else:
                # Convert raw ROS2 msg to tensor
                tensor = topic.to_tensor(msg_or_tensor)
                
            if key in ["action", "observation.state"]:
                out_frame[key].append(tensor)
            else:
                ds_frame[key] = tensor

        if len(out_frame["action"]) != 0:
            ds_frame["action"] = torch.cat(out_frame["action"], dim=0)

        if len(out_frame["observation.state"]) != 0:
            ds_frame["observation.state"] = torch.cat(
                out_frame["observation.state"], dim=0
            )
        return ds_frame
