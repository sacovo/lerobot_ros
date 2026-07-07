import os
from typing import Any, Dict, List, Tuple

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import DEFAULT_FEATURES
from lerobot_ros.core.frame_assembler import FrameAssembler
from lerobot_ros.config import ROSFeatureConfig

INTERVENTION_FEATURE = {
    "intervention": {"dtype": "float32", "shape": (1,), "names": ["intervention"]}
}


class DatasetWriter:
    def __init__(self, repo_id: str, config: ROSFeatureConfig, resume: bool = False):
        self.repo_id = repo_id
        self.config = config
        self.assembler = FrameAssembler(config.topics)
        self.track_intervention = bool(config.human_intervention_topic)
        self.dataset = self._init_dataset(resume)

    def _init_dataset(self, resume: bool) -> LeRobotDataset:
        path = os.path.join(self.config.dataset_root, self.repo_id)
        features = self.assembler.get_feature_description()

        if self.track_intervention:
            features.update(INTERVENTION_FEATURE)

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

    def save_episode(
        self,
        episode_frames: List[Tuple[Dict[str, Any], str, float]],
        task: str,
        success: bool = True,
    ):
        """Write a list of frames to the dataset and save the episode.

        Args:
            episode_frames: List of (frame_dict, task_str, timestamp) tuples.
                            frame_dict may contain "intervention" (float tensor [1])
                            when human_intervention_topic is configured.
            task: Task description string for the episode.
            success: Whether this episode was successful. Stored as episode-level
                     metadata consumed by the critic and RECAP training scripts.
        """
        print(f"Writing episode to LeRobot dataset with {len(episode_frames)} frames...")
        for i, (frame, _, _) in enumerate(episode_frames):
            frame["task"] = task
            if self.track_intervention and "intervention" not in frame:
                frame["intervention"] = torch.tensor([0.0])
            try:
                self.dataset.add_frame(frame)
            except Exception as e:
                print(f"Failed to add frame {i}: {e}")

        # Inject success into episode-level metadata by temporarily patching
        # LeRobotDatasetMetadata.save_episode so that success is stored in the
        # episodes table (read by train_critic.py and ds_utils.load_success_flags).
        meta = self.dataset.writer._meta
        original_meta_save = meta.save_episode

        def _patched_meta_save(
            episode_index, episode_length, episode_tasks, episode_stats, episode_metadata
        ):
            episode_metadata["success"] = success
            return original_meta_save(
                episode_index, episode_length, episode_tasks, episode_stats, episode_metadata
            )

        meta.save_episode = _patched_meta_save
        try:
            self.dataset.save_episode()
            print("Episode saved successfully.")
        except Exception as e:
            print(f"Failed to save episode: {e}")
        finally:
            meta.save_episode = original_meta_save

    def finalize(self):
        """Finalize the dataset (Hugging Face files)."""
        self.dataset.finalize()
