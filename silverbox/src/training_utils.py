"""Reusable training and evaluation for architecture comparisons.

train.py keeps the introductory walkthrough. This module uses the same physical
state equation, with optional batches of training windows for experiments.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch

from data_loader import (
    u_train_centered, y_train_centered, u_test_centered, y_test_centered,
)
from src.silverbox_model import AccelerationModel, DT


@dataclass(frozen=True)
class TrainingConfig:
    """Settings recorded alongside each trained model."""

    hidden_sizes: tuple = (32, 32)
    optimizer: str = "Adam"
    learning_rate: float = 0.01
    steps: int = 3000
    rollout_length: int = 64
    batch_size: int = 1
    seed: int = 0
    # None reproduces train.py's original global random-number sequence.
    window_seed: int | None = 1


def data_tensors():
    """Return centered train/test signals; voltages have one final dimension."""
    return (
        torch.tensor(u_train_centered, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(y_train_centered, dtype=torch.float32),
        torch.tensor(u_test_centered, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(y_test_centered, dtype=torch.float32),
    )


def rms_scale(values):
    return torch.sqrt(torch.mean(values ** 2)).clamp_min(1e-8).item()


def fit_model(config, u_train, y_train, progress_every=500):
    """Learn weights from training windows, without receiving any test signals."""
    if not 2 <= config.rollout_length <= len(y_train):
        raise ValueError("rollout_length must be between 2 and the training length")
    if config.steps <= 0 or config.batch_size <= 0 or config.learning_rate <= 0:
        raise ValueError("steps, batch_size, and learning_rate must be positive")

    torch.manual_seed(config.seed)
    velocity = (y_train[1:] - y_train[:-1]) / DT
    acceleration = (velocity[1:] - velocity[:-1]) / DT
    model = AccelerationModel(
        hidden_sizes=config.hidden_sizes,
        input_scale=(rms_scale(y_train), rms_scale(velocity), rms_scale(u_train)),
        acceleration_scale=rms_scale(acceleration),
    )
    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=1e-4
        )
    else:
        raise ValueError("Supported optimizers: Adam, AdamW")

    # A separate generator makes window choices independent of network width.
    # Equal batch sizes and rollout lengths receive identical windows across models.
    window_rng = (
        torch.Generator().manual_seed(config.window_seed)
        if config.window_seed is not None else None
    )
    offsets = torch.arange(config.rollout_length).unsqueeze(1)
    high = len(y_train) - config.rollout_length + 1
    history = []
    seen = torch.zeros(len(y_train), dtype=torch.bool)
    model.train()
    started = perf_counter()

    for step in range(1, config.steps + 1):
        starts = torch.randint(high, (config.batch_size,), generator=window_rng)
        # Shape (time, batch): each column indexes one complete training window.
        indices = offsets + starts.unsqueeze(0)
        window_u = u_train[indices]  # Shape (time, batch, 1).
        window_y = y_train[indices]  # Shape (time, batch).
        seen[indices.flatten()] = True

        # Each window gets its own training-data initial state; no later correction.
        window_velocity = (window_y[1] - window_y[0]) / DT
        initial_states = torch.stack((window_y[0], window_velocity), dim=-1)
        optimizer.zero_grad()
        if config.window_seed is None and config.batch_size == 1:
            # Keep the original tutorial's vector shapes as well as its RNG sequence.
            predictions = model.rollout(initial_states[0], window_u[:, 0, :])[:, 0]
            loss = torch.mean((predictions - window_y[:, 0]) ** 2)
        else:
            predictions = model.rollout(initial_states, window_u)[..., 0]
            loss = torch.mean((predictions - window_y) ** 2)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at update {step}")
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError(f"Non-finite gradient at update {step}")
        optimizer.step()
        history.append(loss.item())
        if step == 1 or step % progress_every == 0 or step == config.steps:
            print(
                f"  update {step}/{config.steps}: sampled-window MSE={loss.item():.8f} "
                f"({perf_counter() - started:.1f} s)", flush=True
            )

    return model, {
        "training_seconds": perf_counter() - started,
        "covered_training_samples": int(seen.sum()),
        "loss_history": history,
    }


def free_run(model, u_train, y_train, u_test, y_test):
    """Run once from sample 0; measured test positions only score predictions."""
    started = perf_counter()
    initial_state = torch.stack((y_train[0], (y_train[1] - y_train[0]) / DT))
    u_full = torch.cat((u_train, u_test), dim=0)
    model.eval()
    with torch.no_grad():
        states = model.rollout(initial_state, u_full)
        if not torch.isfinite(states).all():
            raise RuntimeError("The continuous free run produced non-finite states")
        predictions = states[:, 0]
        n = len(y_train)
        metrics = {
            "train_rmse": torch.mean((predictions[:n] - y_train) ** 2).sqrt().item(),
            "test_rmse": torch.mean((predictions[n:] - y_test) ** 2).sqrt().item(),
            "zero_train_rmse": torch.mean(y_train ** 2).sqrt().item(),
            "zero_test_rmse": torch.mean(y_test ** 2).sqrt().item(),
            "evaluation_seconds": perf_counter() - started,
        }
    return metrics, states


def save_checkpoint(path, model, config, metrics):
    """Save weights, scale buffers, and settings needed to reconstruct the model."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": asdict(config),
        "state_dict": model.state_dict(),
        "metrics": metrics,
        "sampling_interval": DT,
        "input_order": ["y", "v", "u"],
    }, path)


def load_checkpoint(path):
    """Restore a saved model without fitting it again."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = AccelerationModel(hidden_sizes=tuple(checkpoint["config"]["hidden_sizes"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
