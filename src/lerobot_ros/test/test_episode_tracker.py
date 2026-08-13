"""
Unit tests for EpisodeTracker's normalization buffers and the graph
fingerprint that keeps a stale TensorRT engine from being loaded silently.

Pure torch -- no dataset, no CUDA, no tensorrt -- so these run anywhere the
package imports.
"""

import json

import pytest
import torch

# No module-level pytest.importorskip: launch_testing's collection hook imports
# every test module during collection, and a Skipped raised there aborts
# collection of the ENTIRE directory (see test_annotation_server).
try:
    from lerobot_ros.episode_tracker import EpisodeTracker, _stat_buffer

    _have_tracker = True
except ImportError:
    _have_tracker = False

pytestmark = pytest.mark.skipif(
    not _have_tracker, reason="lerobot_ros.episode_tracker requires lerobot"
)


def _tracker(**kwargs):
    """A tiny image-free tracker: keeps these tests off MobileNet and CUDA."""
    return EpisodeTracker(
        n_robot_state_inputs=3, n_actions=2, image_features=[], window=2, **kwargs
    )


class TestStatBuffer:
    def test_unset_values_are_identity(self):
        assert torch.equal(_stat_buffer(None, 3, 0.0), torch.zeros(3))
        assert torch.equal(_stat_buffer(None, 3, 1.0), torch.ones(3))

    def test_wrong_length_is_rejected(self):
        with pytest.raises(ValueError, match="expected 3 normalization values"):
            _stat_buffer([1.0, 2.0], 3, 1.0)

    def test_degenerate_std_becomes_one_not_epsilon(self):
        """A constant channel must pass through unscaled.

        Clamping to a small epsilon looks safe on in-distribution data (the
        numerator vanishes with the denominator) but turns the channel into a
        1/eps amplifier as soon as a value deviates at inference.
        """
        std = _stat_buffer([0.0, 1e-9, 2.0], 3, 1.0, is_std=True)
        assert std.tolist() == [1.0, 1.0, 2.0]

    def test_means_are_not_clamped(self):
        mean = _stat_buffer([0.0, -5.0, 2.0], 3, 0.0)
        assert mean.tolist() == [0.0, -5.0, 2.0]


class TestNormalization:
    def test_defaults_are_identity(self):
        model = _tracker()
        assert torch.equal(model.state_mean, torch.zeros(3))
        assert torch.equal(model.state_std, torch.ones(3))

    def test_constant_channel_does_not_explode(self):
        """The failure this guards: a TOF zone that never varies in training
        (masked or dead) and then deviates at inference."""
        model = _tracker(
            state_mean=[0.0, 0.0, 0.0], state_std=[0.0, 1.0, 1.0]
        ).eval()

        batch = {
            "observation.state": torch.zeros(1, 2, 3),
            "action": torch.zeros(1, 2, 2),
        }
        # Channel 0 was constant in training; at inference it moves.
        batch["observation.state"][:, :, 0] = 0.5

        normalized = (batch["observation.state"] - model.state_mean) / model.state_std
        assert normalized.abs().max().item() == pytest.approx(0.5)
        assert torch.isfinite(model.predict_progress(batch)).all()

    def test_progress_is_a_probability(self):
        model = _tracker().eval()
        batch = {
            "observation.state": torch.randn(2, 2, 3),
            "action": torch.randn(2, 2, 2),
        }
        progress = model.predict_progress(batch)
        assert progress.shape == (2,)
        assert ((progress >= 0.0) & (progress <= 1.0)).all()


class TestGraphFingerprint:
    def test_is_stable_across_instances(self):
        a = _tracker(state_std=[1.0, 2.0, 3.0]).graph_fingerprint()
        b = _tracker(state_std=[1.0, 2.0, 3.0]).graph_fingerprint()
        assert a["sha256"] == b["sha256"]

    def test_changes_when_normalization_changes(self):
        """The whole point: these constants are traced into the ONNX graph, so
        two models that differ only here produce engines that differ."""
        a = _tracker(state_std=[1.0, 2.0, 3.0]).graph_fingerprint()
        b = _tracker(state_std=[1.0, 2.0, 4.0]).graph_fingerprint()
        assert a["sha256"] != b["sha256"]

    def test_changes_when_window_changes(self):
        a = EpisodeTracker(n_robot_state_inputs=3, n_actions=2, image_features=[], window=2)
        b = EpisodeTracker(n_robot_state_inputs=3, n_actions=2, image_features=[], window=4)
        assert a.graph_fingerprint()["sha256"] != b.graph_fingerprint()["sha256"]

    def test_is_json_serializable(self):
        json.dumps(_tracker().graph_fingerprint())


class TestEngineFingerprintCheck:
    """_check_engine_fingerprint lives in trt.policy but needs no tensorrt:
    it is pure file/dict comparison and runs before the engine is opened."""

    @staticmethod
    def _check():
        pytest.importorskip("lerobot_ros.trt.policy")
        from lerobot_ros.trt.policy import _check_engine_fingerprint

        return _check_engine_fingerprint

    def test_missing_sidecar_warns_but_allows(self, tmp_path, capsys):
        engine = tmp_path / "episode_tracker.trt"
        engine.write_bytes(b"not-a-real-engine")
        self._check()(str(engine), _tracker())
        assert "no fingerprint" in capsys.readouterr().out

    def test_matching_sidecar_passes(self, tmp_path):
        model = _tracker(state_std=[1.0, 2.0, 3.0])
        engine = tmp_path / "episode_tracker.trt"
        engine.write_bytes(b"not-a-real-engine")
        (tmp_path / "episode_tracker.fingerprint.json").write_text(
            json.dumps(model.graph_fingerprint())
        )
        self._check()(str(engine), model)

    def test_mismatched_sidecar_raises(self, tmp_path):
        exported = _tracker(state_std=[1.0, 2.0, 3.0])
        loaded = _tracker(state_std=[1.0, 2.0, 4.0])
        engine = tmp_path / "episode_tracker.trt"
        engine.write_bytes(b"not-a-real-engine")
        (tmp_path / "episode_tracker.fingerprint.json").write_text(
            json.dumps(exported.graph_fingerprint())
        )
        with pytest.raises(ValueError, match="exported from a different model"):
            self._check()(str(engine), loaded)
