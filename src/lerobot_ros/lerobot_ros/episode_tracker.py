import datetime
import hashlib
import json
import logging
import math
import random
from typing import Optional

import click
import torch
from huggingface_hub import PyTorchModelHubMixin
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import ACTION, OBS_STATE
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MOBILENET_FEATURE_DIM = 576


def hl_gauss_bins(n_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Bin edges/centers for HL-Gauss progress classification over [0, 1]."""
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, edges


def hl_gauss_target(scalar: torch.Tensor, edges: torch.Tensor, sigma: float) -> torch.Tensor:
    """Convert scalar progress targets (B,) into HL-Gauss target distributions (B, n_bins).

    Each label becomes a Gaussian centered on its scalar value, discretized by
    integrating the Gaussian CDF over each bin (Farebrother et al., "Stop
    Regressing: Training Value Functions via Classification", ICML 2024).
    """
    z = (edges.unsqueeze(0) - scalar.unsqueeze(1)) / (sigma * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + torch.erf(z))
    probs = cdf[:, 1:] - cdf[:, :-1]
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)


_DEGENERATE_STD = 1e-6


def _stat_buffer(
    values: Optional[list[float]], dim: int, fill: float, is_std: bool = False
) -> torch.Tensor:
    """Build a (dim,) normalization buffer, falling back to ``fill`` when unset.

    A channel that never varies in the dataset has std 0, which would divide to
    inf/NaN. Such channels are given a divisor of 1.0 -- i.e. left unnormalized
    -- rather than clamped to a small epsilon. Clamping looks safe because the
    numerator vanishes with the denominator on in-distribution data, but it
    turns the channel into a ~1/eps amplifier the moment a value deviates at
    inference: 64 of the 83 state dims here are TOF zones, and a masked or dead
    zone (see the TOF_MASK topic) is exactly the constant channel that bites.
    A channel with no variance carries no information, so passing it through
    unscaled loses nothing.
    """
    if values is None:
        return torch.full((dim,), fill, dtype=torch.float32)
    tensor = torch.as_tensor(list(values), dtype=torch.float32)
    if tensor.numel() != dim:
        raise ValueError(f"expected {dim} normalization values, got {tensor.numel()}")
    if is_std:
        tensor = torch.where(
            tensor > _DEGENERATE_STD, tensor, torch.ones_like(tensor)
        )
    return tensor


class EpisodeTracker(PyTorchModelHubMixin, nn.Module):
    """Track episode progress from a short window of images, robot state and actions.

    Predicts a categorical distribution over discretized progress bins (HL-Gauss)
    rather than a scalar: progress is not guaranteed to be monotonic (a grasped
    object can be dropped mid-episode), and a distribution represents that kind
    of uncertain/bimodal situation far better than a single regressed scalar.
    A short causal window (via a GRU) lets the model use recent motion to both
    estimate progress and notice a regression, instead of judging a single frame.
    """

    def __init__(
        self,
        n_robot_state_inputs: int = 0,
        n_actions: int = 0,
        image_features: Optional[list[str]] = None,
        proj_dim: int = 64,
        gru_hidden_dim: int = 64,
        n_bins: int = 101,
        window: int = 4,
        hl_gauss_sigma_ratio: float = 0.75,
        state_mean: Optional[list[float]] = None,
        state_std: Optional[list[float]] = None,
        action_mean: Optional[list[float]] = None,
        action_std: Optional[list[float]] = None,
    ) -> None:
        super().__init__()

        if image_features is None:
            image_features = []

        logger.info(
            f"Initializing EpisodeTracker with {n_robot_state_inputs} state inputs, "
            f"{n_actions} actions, image_features={image_features}, proj_dim={proj_dim}, "
            f"gru_hidden_dim={gru_hidden_dim}, n_bins={n_bins}, window={window}"
        )

        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # Keep the conv feature extractor + global pooling, drop the classifier head.
        self.backbone = nn.Sequential(backbone.features, backbone.avgpool)
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Loaded and froze MobileNetV3-Small backbone")

        self.image_features = image_features
        self.progress_keys = [*image_features, OBS_STATE, ACTION]
        self.window = window
        self.n_bins = n_bins

        # One shared projection (not one per camera): all cameras view the same
        # scene from different angles, and sharing keeps the fused input from
        # being dominated by image features the way the raw 512-d-per-camera
        # concatenation used to.
        self.image_proj = (
            nn.Sequential(nn.Linear(MOBILENET_FEATURE_DIM, proj_dim), nn.ReLU())
            if image_features
            else None
        )

        fusion_dim = proj_dim * len(image_features) + n_robot_state_inputs + n_actions
        self.gru = nn.GRU(fusion_dim, gru_hidden_dim, batch_first=True)
        self.head = nn.Linear(gru_hidden_dim, n_bins)

        bin_centers, bin_edges = hl_gauss_bins(n_bins)
        self.register_buffer("bin_centers", bin_centers)
        self.register_buffer("bin_edges", bin_edges)
        self.hl_gauss_sigma = hl_gauss_sigma_ratio / n_bins

        # State and action feed the GRU directly, so their raw scale matters.
        # On the rover dataset the state alone spans about five orders of
        # magnitude -- joint efforts (std ~1900) and gripper (std ~135) against
        # TOF channels (std ~0.04) -- so without this the GRU is driven almost
        # entirely by the effort channels. The image half of the fusion input is
        # not covered here: image_proj is a learned projection and scales itself.
        #
        # These are buffers, not preprocessing, so save_pretrained carries them
        # and inference (policy_controller -> predict_progress) normalizes
        # identically without knowing the dataset. Defaults are identity, which
        # keeps the module constructible with no stats (tests, TRT tracing).
        self.register_buffer(
            "state_mean", _stat_buffer(state_mean, n_robot_state_inputs, 0.0)
        )
        self.register_buffer(
            "state_std",
            _stat_buffer(state_std, n_robot_state_inputs, 1.0, is_std=True),
        )
        self.register_buffer("action_mean", _stat_buffer(action_mean, n_actions, 0.0))
        self.register_buffer(
            "action_std", _stat_buffer(action_std, n_actions, 1.0, is_std=True)
        )

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model initialized with {total_params:,} trainable parameters")

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict per-bin logits from a window of observations.

        Every tensor in ``batch`` is shaped ``(B, K, ...)`` for a window of K
        timesteps ending at the current frame (use :func:`stack_progress_window`
        to build this from a rolling buffer of single frames).
        """
        window_key = self.image_features[0] if self.image_features else OBS_STATE
        batch_size, window = batch[window_key].shape[:2]

        parts = []
        if self.image_features:
            with torch.no_grad():
                for key in self.image_features:
                    img = batch[key]  # (B, K, C, H, W)
                    img = img.reshape(batch_size * window, *img.shape[2:])
                    feat = self.backbone(img).flatten(1)  # (B*K, 576)
                    feat = feat.view(batch_size, window, MOBILENET_FEATURE_DIM)
                    parts.append(self.image_proj(feat))  # (B, K, proj_dim)

        # Normalized with the buffers above so no single channel's raw scale
        # dominates the GRU input. Identity when the model was built without stats.
        parts.append((batch[OBS_STATE] - self.state_mean) / self.state_std)
        parts.append((batch[ACTION] - self.action_mean) / self.action_std)

        sequence = torch.cat(parts, dim=-1)  # (B, K, fusion_dim)
        _, h_n = self.gru(sequence)
        return self.head(h_n[-1])  # (B, n_bins) logits

    def predict_progress(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Expected progress in [0, 1], computed from the predicted bin distribution."""
        logits = self.forward(batch)
        probs = torch.softmax(logits, dim=-1)
        return (probs * self.bin_centers.unsqueeze(0)).sum(dim=-1)

    def graph_fingerprint(self) -> dict:
        """Identity of everything an ONNX export freezes into the graph.

        The normalization buffers are traced as constants, so an engine built
        from one checkpoint and loaded against another computes different
        numbers with no outward sign -- the engine still has the right input
        names and shapes, and EpisodeTrackerTRTPolicy takes bin_centers and
        window from the eager model, so nothing disagrees visibly. Exports
        write this next to the ONNX and the TRT policy refuses a mismatch.
        """
        payload = {
            "progress_keys": list(self.progress_keys),
            "image_features": list(self.image_features),
            "window": int(self.window),
            "n_bins": int(self.n_bins),
            "state_mean": [round(float(x), 6) for x in self.state_mean],
            "state_std": [round(float(x), 6) for x in self.state_std],
            "action_mean": [round(float(x), 6) for x in self.action_mean],
            "action_std": [round(float(x), 6) for x in self.action_std],
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        return {"sha256": digest, **payload}


def stack_progress_window(
    frames: list[dict[str, torch.Tensor]], window: int
) -> dict[str, torch.Tensor]:
    """Stack a rolling buffer of single, already-batched frames (each value shaped
    ``(B, ...)``) into a windowed batch shaped ``(B, K, ...)``.

    If fewer than ``window`` frames are available yet (e.g. right after a policy
    starts), the earliest frame is repeated to left-pad the window instead of
    reaching into a previous episode/policy run.
    """
    frames = list(frames)
    if len(frames) < window:
        frames = [frames[0]] * (window - len(frames)) + frames
    return {
        key: torch.stack([frame[key] for frame in frames[-window:]], dim=1)
        for key in frames[0]
    }


class WindowedProgressDataset(Dataset):
    """Wraps a per-episode-contiguous dataset to return a window of ``window``
    consecutive frames ending at each index, clamped to the episode start (early
    frames repeat the first frame of the episode instead of reaching into the
    previous one).

    Assumes the wrapped dataset stores frames in increasing ``frame_index`` order
    within each episode (true for a plain ``LeRobotDataset``; not safe to wrap a
    shuffled ``Subset``).
    """

    def __init__(self, dataset, window: int):
        self.dataset = dataset
        self.window = window

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        episode_start = idx - int(self.dataset[idx]["frame_index"].item())
        indices = [
            max(episode_start, idx - (self.window - 1 - j)) for j in range(self.window)
        ]
        frames = [self.dataset[i] for i in indices]

        windowed = {}
        for key, value in frames[0].items():
            if isinstance(value, torch.Tensor):
                windowed[key] = torch.stack([f[key] for f in frames], dim=0)  # (K, ...)
            else:
                windowed[key] = value
        return windowed


def progress_targets(batch, episode_lengths, device):
    """Scalar progress label for the frame at the end of each window."""
    frame_index = batch["frame_index"][:, -1].float()
    episode_index = batch["episode_index"][:, -1]
    return frame_index / episode_lengths[episode_index]


def evaluate(model, dataloader, episode_lengths, device):
    """Evaluate model on validation set: HL-Gauss cross-entropy and an
    expectation-based MSE for comparability with the previous scalar-regression model.
    """
    model.eval()
    total_ce = 0.0
    total_mse = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            episode_progress = progress_targets(batch, episode_lengths, device)

            logits = model(batch)
            target_dist = hl_gauss_target(episode_progress, model.bin_edges, model.hl_gauss_sigma)
            ce_loss = -(target_dist * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

            pred_progress = (torch.softmax(logits, dim=-1) * model.bin_centers).sum(dim=-1)
            mse = nn.functional.mse_loss(pred_progress, episode_progress)

            total_ce += ce_loss.item()
            total_mse += mse.item()
            n_batches += 1

    model.train()
    if n_batches == 0:
        return 0.0, 0.0
    return total_ce / n_batches, total_mse / n_batches


def train_episode_tracker(
    model: EpisodeTracker,
    dataset: LeRobotDataset,
    dataset_val: LeRobotDataset,
    steps: int = 50_000,
    batch_size: int = 32,
    val_interval: int = 500,
    num_workers: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Train the episode tracker."""

    logger.info(f"Starting training for {steps} steps with batch_size={batch_size}")
    logger.info(f"Using device: {device}")

    train_dataset = WindowedProgressDataset(dataset, model.window)
    val_dataset = WindowedProgressDataset(dataset_val, model.window)

    model = model.to(device)
    episode_lengths = torch.tensor(dataset.meta.episodes["length"], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()

    # Each sample decodes window x n_cameras video frames (4 x 3 = 12 here, so
    # 384 per batch of 32), which makes this loop decode-bound rather than
    # GPU-bound. Workers matter far more than the GPU does.
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=False,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=False,
    )

    logger.info(f"Train dataloader: {len(train_dataloader)} batches per epoch")
    logger.info(f"Val dataloader: {len(val_dataloader)} batches")

    step = 0
    epoch = 0
    start_time = datetime.datetime.now()
    best_val_loss = float("inf")
    best_weights = None

    while step < steps:
        epoch += 1
        logger.info(f"Starting epoch {epoch}")

        for batch in train_dataloader:
            if step >= steps:
                break

            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            episode_progress = progress_targets(batch, episode_lengths, device)

            logits = model(batch)
            target_dist = hl_gauss_target(episode_progress, model.bin_edges, model.hl_gauss_sigma)
            loss = -(target_dist * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if step % 100 == 0:
                elapsed = datetime.datetime.now() - start_time
                steps_per_sec = (step + 1) / elapsed.total_seconds() if step > 0 else 0
                eta = (
                    datetime.timedelta(seconds=int((steps - step) / steps_per_sec))
                    if steps_per_sec > 0
                    else "unknown"
                )

                logger.info(
                    f"Step {step}/{steps} - Train Loss: {loss.item():.4f} - "
                    f"Speed: {steps_per_sec:.2f} steps/s - ETA: {eta}"
                )

            # Validation
            if step % val_interval == 0 and step > 0:
                logger.info(f"Running validation at step {step}...")
                val_ce, val_mse = evaluate(model, val_dataloader, episode_lengths, device)
                logger.info(
                    f"Step {step} - Validation CE: {val_ce:.4f} - Validation MSE: {val_mse:.4f}"
                )

                if val_ce < best_val_loss:
                    best_val_loss = val_ce
                    # Clone: state_dict() hands back references to the live
                    # parameters, so keeping it as-is means the optimizer goes on
                    # mutating "best_weights" in place. The restore at the end
                    # then silently loads the FINAL weights rather than the best
                    # ones, making this whole early-stopping path a no-op.
                    best_weights = {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    }
                    logger.info(f"New best validation CE: {best_val_loss:.4f}")

            step += 1

            # Explicitly delete batch to free memory immediately
            del batch, episode_progress, logits, target_dist, loss

        logger.info(f"Completed epoch {epoch}")

    # Final validation
    logger.info("Running final validation...")
    final_val_ce, final_val_mse = evaluate(model, val_dataloader, episode_lengths, device)
    logger.info(f"Final Validation CE: {final_val_ce:.4f} - MSE: {final_val_mse:.4f}")
    logger.info(f"Best Validation CE: {best_val_loss:.4f}")

    total_time = datetime.datetime.now() - start_time
    logger.info(f"Training completed in {total_time}")
    logger.info(f"Average speed: {steps / total_time.total_seconds():.2f} steps/s")

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model


@click.command()
@click.option("--repo_id", type=str, required=True)
@click.option("--output_dir", type=str, default=None)
@click.option("--steps", type=int, default=10000)
@click.option(
    "--model_repo_id", type=str, default="fhnwrover/so101-ros-red-ring-episode-tracker"
)
@click.option("--push_to_hub/--no_push_to_hub", default=False)
@click.option("--batch_size", type=int, default=32)
@click.option(
    "--val_split",
    type=float,
    default=0.2,
    help="Fraction of episodes to use for validation",
)
@click.option("--val_interval", type=int, default=500, help="Validate every N steps")
@click.option("--window", type=int, default=4, help="Number of frames in the temporal window")
@click.option("--n_bins", type=int, default=101, help="Number of HL-Gauss progress bins")
@click.option(
    "--num_workers",
    type=int,
    default=4,
    help="Dataloader workers. Each sample decodes window x n_cameras video "
    "frames, so this dominates throughput; 0 decodes serially in-process.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Seed for the train/val episode split, so runs stay comparable.",
)
@click.option("--log_file", type=str, default=None, help="Optional log file path")
def main(
    repo_id,
    output_dir,
    steps,
    model_repo_id,
    push_to_hub,
    batch_size,
    val_split,
    val_interval,
    window,
    n_bins,
    num_workers,
    seed,
    log_file,
):
    # Add file handler if log file specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    logger.info("=" * 80)
    logger.info("Starting Episode Tracker Training")
    logger.info("=" * 80)
    logger.info(f"Repository ID: {repo_id}")
    logger.info(f"Training steps: {steps}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Validation split: {val_split * 100:.1f}%")
    logger.info(f"Validation interval: {val_interval} steps")
    logger.info(f"Window: {window} frames, bins: {n_bins}")
    logger.info(f"Dataloader workers: {num_workers}")
    logger.info(f"Split seed: {seed}")
    logger.info(f"Model will be saved to: {model_repo_id}")
    logger.info(f"Push to hub: {push_to_hub}")

    logger.info("Loading dataset...")

    metadata = LeRobotDatasetMetadata(repo_id)

    all_episodes = list(metadata.episodes["episode_index"])

    logger.info(f"Total episodes in dataset: {len(all_episodes)}")

    n_train = int(len(all_episodes) * (1 - val_split))

    # Seeded so the split is reproducible: without it every run validates on a
    # different set of episodes and the val numbers aren't comparable run to run.
    random.seed(seed)
    episodes_train = random.sample(all_episodes, n_train)
    episodes_val = list(set(all_episodes) - set(episodes_train))

    ds = LeRobotDataset(repo_id, episodes=episodes_train)
    ds_val = LeRobotDataset(repo_id, episodes=episodes_val)
    logger.info("Dataset loaded successfully")

    if output_dir is None:
        output_dir = f"outputs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_episode_tracker"

    logger.info(f"Output directory: {output_dir}")

    image_features = [f for f in ds.meta.features if "image" in f]
    n_states = ds.meta.features[OBS_STATE]["shape"][0]
    n_actions = ds.meta.features[ACTION]["shape"][0]

    logger.info(f"Found {len(image_features)} image features")
    logger.info(f"State dimensions: {n_states}")
    logger.info(f"Action dimensions: {n_actions}")

    # Read off `metadata`, the unfiltered LeRobotDatasetMetadata built above,
    # rather than ds.meta: ds was constructed with episodes=episodes_train, and
    # whether .meta.stats then covers the whole dataset or only that subset has
    # varied across lerobot versions. Taking them from the unfiltered handle
    # makes the scope explicit instead of dependent on the pin.
    #
    # Baked into the model as buffers so inference applies the same transform.
    # Plain lists keep them JSON-serializable for the PyTorchModelHubMixin
    # config written by save_pretrained.
    def _stat(feature, key):
        value = metadata.stats[feature][key]
        return [float(x) for x in value]

    logger.info("Normalization: state/action standardized with dataset mean/std")

    model = EpisodeTracker(
        n_robot_state_inputs=n_states,
        n_actions=n_actions,
        image_features=image_features,
        window=window,
        n_bins=n_bins,
        state_mean=_stat(OBS_STATE, "mean"),
        state_std=_stat(OBS_STATE, "std"),
        action_mean=_stat(ACTION, "mean"),
        action_std=_stat(ACTION, "std"),
    )

    model = train_episode_tracker(
        model,
        ds,
        ds_val,
        steps=steps,
        batch_size=batch_size,
        val_interval=val_interval,
        num_workers=num_workers,
    )

    logger.info(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir, repo_id=model_repo_id, push_to_hub=push_to_hub)
    logger.info("Model saved successfully")

    if push_to_hub:
        logger.info(f"Model pushed to Hugging Face Hub: {model_repo_id}")

    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
