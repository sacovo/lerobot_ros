"""
Differential test for the EpisodeTracker TensorRT export pipeline: exports a
small, randomly-initialized model to ONNX, builds a real TensorRT engine, and
asserts the engine's predicted progress matches the eager PyTorch model.

Needs no real checkpoint or dataset -- only the export/engine-build/runtime
plumbing is under test here, not model accuracy -- but does need a real CUDA
device with tensorrt/onnx installed, so it only runs where that's available
(e.g. on the Jetson CI runner), and is skipped everywhere else.
"""

import pytest
import torch

# No module-level pytest.importorskip here: launch_testing's collection hook
# imports every test module during collection, and a Skipped raised there
# aborts collection of the ENTIRE test directory (see test_annotation_server).
try:
    import tensorrt  # noqa: F401
    import onnx  # noqa: F401
    _have_trt = True
except ImportError:
    _have_trt = False

pytestmark = pytest.mark.skipif(
    not (_have_trt and torch.cuda.is_available()),
    reason="TensorRT engine building requires tensorrt/onnx and a CUDA device",
)

if _have_trt:
    from lerobot_ros.episode_tracker import EpisodeTracker
    from lerobot_ros.trt.engine import build_trt_engine
    from lerobot_ros.trt.exporter import export_episode_tracker
    from lerobot_ros.trt.policy import EpisodeTrackerTRTPolicy
    from lerobot_ros.trt.validate import validate_episode_tracker_accuracy


def _make_dummy_model():
    torch.manual_seed(0)
    return (
        EpisodeTracker(
            n_robot_state_inputs=6,
            n_actions=3,
            image_features=["observation.images.cam0", "observation.images.cam1"],
            proj_dim=8,
            gru_hidden_dim=8,
            n_bins=15,
            window=4,
        )
        .eval()
        .to("cuda")
    )


class _SyntheticWindowedDataset:
    """Stand-in for episode_tracker.WindowedProgressDataset with the same
    __getitem__ contract (unbatched, shape (K, ...) per key) but backed by
    synthetic data instead of a real LeRobotDataset."""

    def __init__(self, model, size=8):
        self.model = model
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        torch.manual_seed(idx)
        K = self.model.window
        sample = {key: torch.rand(K, 3, 32, 32) for key in self.model.image_features}
        sample["observation.state"] = torch.randn(K, 6)
        sample["action"] = torch.randn(K, 3)
        return sample


def test_episode_tracker_trt_matches_eager(tmp_path):
    """EpisodeTrackerTRTPolicy must reproduce the eager EpisodeTracker's
    predicted progress closely enough to be a safe drop-in at inference time
    (see policy_controller.py's progress_model_use_trt path)."""
    model = _make_dummy_model()
    dataset = _SyntheticWindowedDataset(model)
    sample_batch = {k: v.unsqueeze(0).to("cuda") for k, v in dataset[0].items()}

    onnx_path = str(tmp_path / "episode_tracker.onnx")
    engine_path = str(tmp_path / "episode_tracker.trt")

    export_episode_tracker(model, sample_batch, onnx_path)
    build_trt_engine(onnx_path, engine_path, fp16=False)

    trt_policy = EpisodeTrackerTRTPolicy(engine_path, model)

    max_abs, mean_abs = validate_episode_tracker_accuracy(model, trt_policy, dataset, num_samples=8)

    assert max_abs < 5e-3, f"TRT progress diverged from eager by {max_abs:.6f}"
