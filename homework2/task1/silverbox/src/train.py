"""Train the acceleration model on windows sampled across the training record.

Run from the project folder with: .venv/bin/python src/train.py
Each optimizer update uses a randomly selected 64-sample training window.
After fitting, one continuous free run evaluates the full training and test
records without resetting the state at the test boundary.
"""

from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_loader import (
    u_train_centered,
    u_test_centered,
    y_train_centered,
    y_test_centered,
)
from src.silverbox_model import AccelerationModel, DT


# Number of samples in this short prediction window, not training step count.
# 64 samples span 63 sampling intervals, approximately 0.103 seconds.
ROLLOUT_LENGTH = 64

# One training step means one weight update, not one RK4 integration step.
NUM_STEPS = 3000
# Adam's learning rate controls the scale of its weight updates.
LEARNING_RATE = 1e-2  # 0.01; a starting setting to tune later.


def rms_scale(values):
    """Choose a typical magnitude: square values, average, then take the root."""
    # A tiny minimum avoids dividing by zero if a signal is entirely zero.
    return torch.sqrt(torch.mean(values ** 2)).clamp_min(1e-8).item()


def plot_test_predictions(measured_y, predicted_y, test_rmse, baseline_rmse, show=True, output_path=None):
    """Plot the already-completed free run; measured y is only a comparison."""
    # Saving without a window needs a renderer that works without a desktop.
    if not show:
        plt.switch_backend("Agg")

    # Matplotlib uses NumPy arrays. detach() removes gradient tracking, and cpu()
    # ensures conversion works even if the tensors are later moved to a GPU.
    measured = measured_y.detach().cpu().numpy()
    predicted = predicted_y.detach().cpu().numpy()

    # Time is relative to the FIRST TEST sample, not the start of the full excerpt.
    time = np.arange(len(measured)) * DT
    error = predicted - measured
    zoom = time <= 0.15  # Show about ten oscillation cycles in detail.

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), layout="constrained")
    fig.suptitle("Silverbox: test predictions from one continuous free run", fontsize=15)

    # Full segment: check whether amplitude and timing stay accurate over time.
    axes[0].plot(time, measured, color="#2563eb", linewidth=1.0, label="Measured output")
    axes[0].plot(time, predicted, color="#ea580c", linewidth=1.0, linestyle="--", label="Predicted output")
    axes[0].set_title(
        f"Full test segment | RMSE: {1000 * test_rmse:.2f} mV | "
        f"Zero-prediction baseline: {1000 * baseline_rmse:.2f} mV"
    )
    axes[0].set_ylabel("Centered output (V)")
    axes[0].set_xlim(time[0], time[-1])
    axes[0].legend(loc="upper right")

    # Zoom: inspect individual peaks and whether the curves line up in time.
    axes[1].plot(time[zoom], measured[zoom], color="#2563eb", linewidth=1.6, label="Measured output")
    axes[1].plot(time[zoom], predicted[zoom], color="#ea580c", linewidth=1.6, linestyle="--", label="Predicted output")
    axes[1].set_title("Zoom: first 0.15 seconds of the test segment")
    axes[1].set_ylabel("Centered output (V)")
    axes[1].set_xlim(0, 0.15)

    # Residual: perfect predictions would give a flat line at zero.
    axes[2].plot(time, 1000 * error, color="#7c3aed", linewidth=1.0)
    axes[2].axhline(0, color="#475569", linewidth=0.8)
    axes[2].set_title("Prediction error = predicted output - measured output")
    axes[2].set_ylabel("Error (mV)")
    axes[2].set_xlim(time[0], time[-1])

    for axis in axes:
        axis.set_xlabel("Time since test segment began (s)")
        axis.grid(alpha=0.2)

    # Resolve the output path relative to the project, independent of launch folder.
    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    # A timestamp gives each run its own image instead of replacing the old one.
    # Use Yerevan time; microseconds also distinguish saves within the same second.
    timestamp = datetime.now(ZoneInfo("Asia/Yerevan")).strftime("%Y%m%d_%H%M%S_%f")
    plot_path = Path(output_path) if output_path is not None else output_dir / f"test_predictions_{timestamp}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180)
    print("Test comparison plot saved to:", plot_path)

    # Display the saved graph with the environment's default plotting backend.
    # On an interactive desktop backend, the script waits until you close the window.
    if show:
        plt.show()
    plt.close(fig)  # Release the figure after saving it.
    return plot_path


def main():
    # Use the same random initialization each time so this run is repeatable.
    torch.manual_seed(0)

    # Convert NumPy arrays into tensors matching the model's float32 weights.
    # Voltage needs a final dimension of size 1 for the rollout: (3072, 1).
    # unsqueeze(-1) adds that dimension; it does not change the voltage values.
    u_train_tensor = torch.tensor(u_train_centered, dtype=torch.float32).unsqueeze(-1)
    y_train_tensor = torch.tensor(y_train_centered, dtype=torch.float32)

    # Keep the first window as a fixed before/after reference for reporting.
    # Optimization will sample other windows too; this reference stays unchanged.
    reference_u = u_train_tensor[:ROLLOUT_LENGTH]  # Shape (64, 1).
    reference_y = y_train_tensor[:ROLLOUT_LENGTH]  # Shape (64,).

    # We know the starting position from the first measured training sample.
    initial_position = reference_y[0]

    # Estimate starting velocity: change in position divided by elapsed time.
    initial_velocity = (reference_y[1] - reference_y[0]) / DT

    # Preserve this sample-0 state for reference and final free-run evaluation.
    # It must not be overwritten with a randomly selected window's initial state.
    initial_state = torch.stack((initial_position, initial_velocity))

    # Estimate derivative magnitudes using ONLY training positions.
    # These finite differences choose scales; acceleration is not a training target.
    velocity_estimate = (y_train_tensor[1:] - y_train_tensor[:-1]) / DT
    acceleration_estimate = (velocity_estimate[1:] - velocity_estimate[:-1]) / DT

    # Keep the 3 -> 32 -> 32 -> 1 architecture, but scale each input separately.
    # forward() divides inputs by these scales and rescales its acceleration output.
    # RK4 still integrates physical position, velocity, and acceleration.
    model = AccelerationModel(
        input_scale=(
            rms_scale(y_train_tensor),
            rms_scale(velocity_estimate),
            rms_scale(u_train_tensor),
        ),
        acceleration_scale=rms_scale(acceleration_estimate),
    )
    # Adam receives the model's trainable weights and biases, not its scale buffers.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Select training mode. This alone does not update weights.
    model.train()

    print("Training input tensor shape:", tuple(u_train_tensor.shape))
    print("Training output tensor shape:", tuple(y_train_tensor.shape))
    print("Initial state [y0, v0]:", initial_state.tolist())
    print("Input scales [y, v, u]:", model.input_scale.tolist())
    print("Acceleration scale:", model.acceleration_scale.item())
    print(f"Signal duration in one training window: {(ROLLOUT_LENGTH - 1) * DT:.6f} s")
    print(f"Optimizer updates: {NUM_STEPS}; RK4 steps per update: {ROLLOUT_LENGTH - 1}")

    # Measure the starting error without building a gradient calculation graph.
    # no_grad is appropriate for reporting, but not for the training rollout below.
    with torch.no_grad():
        predicted_states = model.rollout(initial_state, reference_u)
        initial_loss = torch.mean((predicted_states[:, 0] - reference_y) ** 2)
    print("Predicted state tensor shape:", tuple(predicted_states.shape))
    print(f"Untrained first-window MSE: {initial_loss.item():.8f} V^2")

    # A 64-sample window must fit entirely inside the 3072 training samples.
    # Valid starts are 0 through 3008, including the last window [3008:3072].
    last_window_start = len(y_train_tensor) - ROLLOUT_LENGTH
    seen_samples = torch.zeros(len(y_train_tensor), dtype=torch.bool)

    # perf_counter measures elapsed computer time, not the signal's physical time.
    # This timer includes rollouts, backward calculations, updates, and loop prints.
    training_started = perf_counter()
    for step in range(1, NUM_STEPS + 1):
        # Draw a new start index. randint excludes its upper bound, hence +1.
        # These slices use training tensors only; they cannot enter the test set.
        window_start = torch.randint(0, last_window_start + 1, (1,)).item()
        window_stop = window_start + ROLLOUT_LENGTH
        u_window = u_train_tensor[window_start:window_stop]
        y_window = y_train_tensor[window_start:window_stop]
        seen_samples[window_start:window_stop] = True

        # Initialize this window using its measured TRAINING position and velocity.
        # Inside the rollout, every subsequent state still comes from prediction.
        window_position = y_window[0]
        window_velocity = (y_window[1] - y_window[0]) / DT
        window_initial_state = torch.stack((window_position, window_velocity))

        # 1. Clear previous gradients: backward() otherwise adds to them.
        optimizer.zero_grad()

        # 2. Generate a fresh rollout using the current weights.
        # Inside each rollout, later states come from predictions, not measured y.
        predicted_states = model.rollout(window_initial_state, u_window)  # Shape (64, 2).
        predicted_y = predicted_states[:, 0]  # Column 0 is y; column 1 is v.

        # 3. Compare predicted and measured positions with mean squared error.
        # Keep loss as a tensor so PyTorch can differentiate it.
        loss = torch.mean((predicted_y - y_window) ** 2)

        # 4. Differentiate loss through the rollout and store parameter gradients.
        # This calculates gradients; it does not change the weights yet.
        loss.backward()

        # 5. Adam uses those gradients to update the weights and biases.
        optimizer.step()

        # Each printed loss uses a different window, so it need not decrease steadily.
        # This loss was calculated before the current step's weight update.
        if step == 1 or step % 50 == 0:
            print(f"Step {step:3d}/{NUM_STEPS} - start {window_start:4d}, window MSE: {loss.item():.8f} V^2")

    training_seconds = perf_counter() - training_started
    print(f"Training loop elapsed time: {training_seconds:.2f} s")
    print(f"Average time per optimizer update: {1000 * training_seconds / NUM_STEPS:.2f} ms")
    print(f"Training samples included in at least one window: {seen_samples.sum().item()}/{len(y_train_tensor)}")

    # Recompute the SAME reference window used before training for a fair comparison.
    model.eval()
    with torch.no_grad():
        predicted_states = model.rollout(initial_state, reference_u)
        final_loss = torch.mean((predicted_states[:, 0] - reference_y) ** 2)
    print(f"Trained first-window MSE: {final_loss.item():.8f} V^2")
    # MSE is in squared volts; its square root (RMSE) is in volts.
    print(f"Trained first-window RMSE: {torch.sqrt(final_loss).item():.8f} V")

    # Evaluate generalization after the weights have been fixed.
    # Join the voltage records, then run ONCE from sample 0 through sample 4095.
    # There is no state reset or measured position correction at the test boundary.
    u_test_tensor = torch.tensor(u_test_centered, dtype=torch.float32).unsqueeze(-1)
    u_full = torch.cat((u_train_tensor, u_test_tensor), dim=0)
    evaluation_started = perf_counter()
    with torch.no_grad():
        full_states = model.rollout(initial_state, u_full)
        full_predicted_y = full_states[:, 0]

        # Split predictions after the continuous rollout, not before integration.
        train_size = len(y_train_tensor)
        predicted_train_y = full_predicted_y[:train_size]
        predicted_test_y = full_predicted_y[train_size:]

        # Test positions are used only now, to score the completed predictions.
        y_test_tensor = torch.tensor(y_test_centered, dtype=torch.float32)
        train_rmse = torch.sqrt(torch.mean((predicted_train_y - y_train_tensor) ** 2))
        test_rmse = torch.sqrt(torch.mean((predicted_test_y - y_test_tensor) ** 2))

        # Predicting zero is a useful baseline on each corresponding segment.
        zero_train_rmse = torch.sqrt(torch.mean(y_train_tensor ** 2))
        zero_test_rmse = torch.sqrt(torch.mean(y_test_tensor ** 2))

    evaluation_seconds = perf_counter() - evaluation_started
    print(f"Signal duration in the full excerpt: {(len(u_full) - 1) * DT:.6f} s")
    print(f"Full free-run evaluation elapsed time: {evaluation_seconds:.2f} s")
    print(f"Full free-run training RMSE: {train_rmse.item():.8f} V")
    print(f"Full free-run test RMSE: {test_rmse.item():.8f} V")
    print(f"Zero-prediction training RMSE: {zero_train_rmse.item():.8f} V")
    print(f"Zero-prediction test RMSE: {zero_test_rmse.item():.8f} V")
    if not torch.isfinite(full_states).all():
        print("The free run diverged to non-finite values; these predictions are unusable.")
    else:
        # Plot these exact predictions; do not restart the rollout at the test boundary.
        plot_test_predictions(
            y_test_tensor, predicted_test_y, test_rmse.item(), zero_test_rmse.item()
        )

    # A low training error with a much higher test error suggests poor generalization.
    # If both errors are high, the model also fails across the training record.
    # This evaluation supplies evidence; it cannot prove the absence of overfitting.


# Run main only when this file is launched as a script, not when it is imported.
if __name__ == "__main__":
    main()
