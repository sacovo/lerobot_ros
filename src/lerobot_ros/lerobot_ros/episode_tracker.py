import datetime
import logging
import random
from typing import Optional

import click
import torch
from huggingface_hub import PyTorchModelHubMixin
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import ACTION, OBS_STATE
from torch import nn
from torch.utils.data import Subset
from torchvision.models import ResNet18_Weights, resnet18

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class EpisodeTracker(PyTorchModelHubMixin, nn.Module):
    """Track episode progress based on images, observations and actions."""

    def __init__(
        self,
        n_robot_state_inputs: int = 0,
        n_actions: int = 0,
        hidden_dim: int = 256,
        image_features: Optional[list[str]] = None,
    ) -> None:
        super().__init__()

        logger.info(
            f"Initializing EpisodeTracker with {n_robot_state_inputs} state inputs, "
            f"{n_actions} actions, hidden_dim={hidden_dim}"
        )
        logger.info(f"Image features: {image_features}")
        if image_features is None:
            image_features = []

        # Simpler backbone - just use resnet's avgpool output
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Remove the final FC layer, keep everything up to avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Freeze backbone to save memory
        for param in self.backbone.parameters():
            param.requires_grad = False

        logger.info("Loaded and froze ResNet18 backbone")

        self.image_features = image_features
        n_images = len(image_features)

        # Single MLP for all inputs
        # ResNet18 avgpool outputs 512 features per image
        input_dim = 512 * n_images + n_robot_state_inputs + n_actions

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model initialized with {total_params:,} trainable parameters")

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = []

        # Process images with no gradient for backbone
        with torch.no_grad():
            for key in self.image_features:
                img = batch[key]
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                # backbone already includes avgpool and flatten
                img_feat = self.backbone(img).squeeze(-1).squeeze(-1)  # (B, 512)
                features.append(img_feat)

        # Add state and action directly
        features.append(batch[OBS_STATE])
        features.append(batch[ACTION])

        # Concatenate and predict
        x = torch.cat(features, dim=-1)
        return self.mlp(x).squeeze(-1)


def create_train_val_split(
    dataset: LeRobotDataset, val_split: float = 0.1, seed: int = 42
):
    """Create train/val split by splitting episodes, not individual frames.

    This ensures frames from the same episode don't leak between train and val.
    """
    logger.info(f"Creating train/val split with {val_split * 100:.1f}% validation data")

    # Get episode information
    episodes = dataset.meta.episodes
    n_episodes = len(episodes)

    # Create episode indices and shuffle them
    episode_indices = list(range(n_episodes))
    torch.manual_seed(seed)
    episode_indices = torch.randperm(n_episodes).tolist()

    # Split episodes
    n_val_episodes = max(1, int(n_episodes * val_split))
    val_episode_indices = set(episode_indices[:n_val_episodes])
    train_episode_indices = set(episode_indices[n_val_episodes:])

    logger.info(
        f"Split: {len(train_episode_indices)} train episodes, {len(val_episode_indices)} val episodes"
    )

    # Get frame indices for each split
    train_indices = []
    val_indices = []

    for idx in range(len(dataset)):
        # Get episode index for this frame
        episode_idx = dataset.episode_data_index["episode_index"][idx].item()

        if episode_idx in train_episode_indices:
            train_indices.append(idx)
        else:
            val_indices.append(idx)

    logger.info(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset


def evaluate(model, dataloader, episode_lengths, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            episode_progress = (
                batch["frame_index"].float() / episode_lengths[batch["episode_index"]]
            )

            pred_progress = model(batch)
            loss = nn.functional.mse_loss(pred_progress, episode_progress)

            total_loss += loss.item()
            n_batches += 1

    model.train()
    return total_loss / n_batches if n_batches > 0 else 0


def train_episode_tracker(
    model: EpisodeTracker,
    dataset: LeRobotDataset,
    dataset_val: LeRobotDataset,
    steps: int = 50_000,
    batch_size: int = 32,
    val_interval: int = 500,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Train the episode tracker."""

    logger.info(f"Starting training for {steps} steps with batch_size={batch_size}")
    logger.info(f"Using device: {device}")

    # Create train/val split
    train_dataset, val_dataset = dataset, dataset_val

    model = model.to(device)
    episode_lengths = torch.tensor(dataset.meta.episodes["length"], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
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

            episode_progress = (
                batch["frame_index"].float() / episode_lengths[batch["episode_index"]]
            )

            pred_progress = model(batch)
            loss = nn.functional.mse_loss(pred_progress, episode_progress)

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
                val_loss = evaluate(model, val_dataloader, episode_lengths, device)
                logger.info(f"Step {step} - Validation Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = model.state_dict()
                    logger.info(f"New best validation loss: {best_val_loss:.4f}")

            step += 1

            # Explicitly delete batch to free memory immediately
            del batch, episode_progress, pred_progress, loss

        logger.info(f"Completed epoch {epoch}")

    # Final validation
    logger.info("Running final validation...")
    final_val_loss = evaluate(model, val_dataloader, episode_lengths, device)
    logger.info(f"Final Validation Loss: {final_val_loss:.4f}")
    logger.info(f"Best Validation Loss: {best_val_loss:.4f}")

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
    default=0.1,
    help="Fraction of episodes to use for validation",
)
@click.option("--val_interval", type=int, default=500, help="Validate every N steps")
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
    logger.info(f"Model will be saved to: {model_repo_id}")
    logger.info(f"Push to hub: {push_to_hub}")

    logger.info("Loading dataset...")

    metadata = LeRobotDatasetMetadata(repo_id)

    all_episodes = list(metadata.episodes["episode_index"])

    logger.info(f"Total episodes in dataset: {len(all_episodes)}")

    val_split = 0.2

    n_train = int(len(all_episodes) * (1 - val_split))

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

    model = EpisodeTracker(
        n_robot_state_inputs=n_states,
        n_actions=n_actions,
        image_features=image_features,
    )

    model = train_episode_tracker(
        model,
        ds,
        ds_val,
        steps=steps,
        batch_size=batch_size,
        val_split=val_split,
        val_interval=val_interval,
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
