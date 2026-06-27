import os
from typing import Dict, Any, List, Tuple
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import DEFAULT_FEATURES
from lerobot_ros.core.frame_assembler import FrameAssembler
from lerobot_ros.config import ROSFeatureConfig

class DatasetWriter:
    def __init__(self, repo_id: str, config: ROSFeatureConfig, resume: bool = False):
        self.repo_id = repo_id
        self.config = config
        self.assembler = FrameAssembler(config.topics)
        self.dataset = self._init_dataset(resume)

    def _init_dataset(self, resume: bool) -> LeRobotDataset:
        path = os.path.join(self.config.dataset_root, self.repo_id)
        features = self.assembler.get_feature_description()
        
        if resume:
            return LeRobotDataset.resume(
                repo_id=self.repo_id,
                root=path,
                tolerance_s=self.config.tolerance_s,
            )
        elif os.path.exists(path):
            dataset = LeRobotDataset(
                self.repo_id, root=path, tolerance_s=self.config.tolerance_s
            )
            ds_features = set(dataset.features) - set(DEFAULT_FEATURES)
            expected_features = set(features)
            if ds_features != expected_features:
                raise ValueError("Dataset already exists with different features.")
            return dataset
        else:
            return LeRobotDataset.create(
                self.repo_id,
                fps=self.config.fps,
                features=features,
                root=os.path.abspath(path),
                tolerance_s=self.config.tolerance_s,
            )

    def save_episode(self, episode_frames: List[Tuple[Dict[str, Any], str, float]], task: str):
        """Write a list of frames to the dataset and save the episode."""
        print(f"Writing episode to LeRobot dataset with {len(episode_frames)} frames...")
        for i, (frame, _, _) in enumerate(episode_frames):
            frame["task"] = task
            try:
                self.dataset.add_frame(frame)
            except Exception as e:
                print(f"Failed to add frame {i}: {e}")
        try:
            self.dataset.save_episode()
            print("Episode saved successfully.")
        except Exception as e:
            print(f"Failed to save episode: {e}")

    def finalize(self):
        """Finalize the dataset (Hugging Face files)."""
        self.dataset.finalize()
