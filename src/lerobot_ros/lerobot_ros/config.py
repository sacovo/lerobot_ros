from dataclasses import dataclass, field
from typing import Dict, List, Optional

import toml
import torch
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from .convert import BaseTopic

QOS_SENSOR_DATA = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
)


class PolicyConfig:
    def __init__(
        self,
        pretrained_name_or_path: str,
        device: str,
        ds_repo_id: str,
        ds_root: Optional[str] = None,
        progress_model: str | None = None,
        rename_map: Optional[Dict[str, str]] = None,
        policy_config: Optional[dict] = None,
        action_queue_size: int = 30,
        action_smoothing_beta: float = 1.0,
    ):
        self.pretrained_name_or_path = pretrained_name_or_path
        self.device = device
        self.ds_repo_id = ds_repo_id
        self.ds_root = ds_root
        self.rename_map = rename_map
        self.policy_config = policy_config or {}
        self.action_queue_size = action_queue_size
        self.action_smoothing_beta = action_smoothing_beta
        self.progress_model = progress_model


def load_qos(params):
    history = QoSHistoryPolicy.get_from_short_key(params.get("history", "keep_last"))
    depth = params.get("depth", 10)
    reliability = QoSReliabilityPolicy.get_from_short_key(
        params.get("reliability", "best_effort")
    )
    durability = QoSDurabilityPolicy.get_from_short_key(
        params.get("durability", "transient_local")
    )

    return QoSProfile(
        history=history,
        depth=depth,
        reliability=reliability,
        durability=durability,
    )


def empty_frame(topics: Dict[str, BaseTopic]) -> Dict[str, List[torch.Tensor]]:
    """Create an empty frame with the configured topics."""
    return {topic_name: [] for topic_name in topics.keys()}


@dataclass
class ROSFeatureConfig:
    topics: Dict[str, BaseTopic]

    qos: QoSProfile = field(default_factory=lambda: QOS_SENSOR_DATA)
    visualize: bool = False
    rerrun_remote: Optional[str] = None
    fps: int = 20
    tolerance_s: float = 0.01
    dataset_root: str = "./datasets"
    policies: Dict[str, PolicyConfig] = field(default_factory=dict)

    robot_type: str = ""

    def empty_frame(self) -> Dict[str, List[torch.Tensor]]:
        """Create an empty frame with the configured topics."""
        return empty_frame(self.topics)


def load_toml_dict(file_path: str) -> dict:
    """
    Load a TOML configuration file and return its contents as a dictionary.

    Args:
        file_path (str): The path to the TOML configuration file.

    Returns:
        dict: The contents of the TOML file as a dictionary.
    """
    try:
        with open(file_path, "r") as file:
            config = toml.load(file)
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration from {file_path}: {e}")


def parse_config(config: dict) -> ROSFeatureConfig:
    """
    Parse the configuration dictionary to extract relevant settings.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing parsed configuration settings.
    """
    fps = config.get("fps", 20)
    tolerance_s = config.get("tolerance_s", 0.01)
    dataset_root = config.get("dataset_root", "./datasets")

    config_topics = config.get("topics", {})

    topics = {}

    for topic, settings in config_topics.items():
        topic_type = settings.pop("msg_type")

        if "topic_name" not in settings:
            settings["topic_name"] = topic

        settings["qos"] = load_qos(settings.get("qos", {}))

        topics[topic] = BaseTopic.MAPPINGS[topic_type](**settings)

    policies = {}

    for policy, settings in config.get("policies", {}).items():
        if "pretrained_name_or_path" not in settings:
            raise ValueError(
                f"Policy {policy} must have a 'pretrained_name_or_path' field."
            )
        policies[policy] = PolicyConfig(
            **settings,
        )

    return ROSFeatureConfig(
        topics=topics,
        policies=policies,
        fps=fps,
        tolerance_s=tolerance_s,
        dataset_root=dataset_root,
        visualize=config.get("visualize", False),
        rerrun_remote=config.get("rerun_remote", None),
        robot_type=config.get("robot_type", ""),
    )
