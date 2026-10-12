"""Execute model-selected configurations with consistent benchmark artifacts.

The executor's generated script supplies architectures and settings. These
operational helpers handle checkpoint/CSV/plot bookkeeping; the scientific
training and continuous RK4 simulation remain in training_utils.
"""

from dataclasses import asdict
import json
import shutil
import numpy as np
import torch
from training_utils import (
    TrainingConfig, data_tensors, fit_model, free_run, load_checkpoint, save_checkpoint,
)
from train import plot_test_predictions


def run_study(architectures, settings):
    """Fit every architecture/setting pair from scratch in the current directory."""
    if len(architectures) != 2 or len(settings) != 2:
        raise ValueError("Supply the scientist's two architectures and two settings")
    torch.set_num_threads(4)
    signals = data_tensors()
    u_train, y_train, u_test, y_test = signals
    results = []
    # Nested loops try both settings for BOTH architectures: four fresh fits.
    for widths in architectures:
        for setting in settings:
            # A new fit_model call creates a fresh network and optimizer.
            config = TrainingConfig(hidden_sizes=tuple(widths), **setting, seed=0, window_seed=1)
            print("Fitting:", asdict(config), flush=True)
            model, info = fit_model(config, u_train, y_train)
            metrics, _ = free_run(model, *signals)
            filename = f"run_{len(results)}.pt"
            save_checkpoint(filename, model, config, metrics)
            results.append({"checkpoint": filename, "config": asdict(config),
                **{key: metrics[key] for key in ("train_rmse", "test_rmse", "zero_test_rmse")},
                "training_seconds": info["training_seconds"],
                "covered_training_samples": info["covered_training_samples"]})
    with open("results.json", "w") as file:
        json.dump(results, file, indent=2)
    # Select by test RMSE, as requested; this test set also serves model selection.
    best = min(results, key=lambda row: row["test_rmse"])
    shutil.copyfile(best["checkpoint"], "best_model.pt")
    model, _ = load_checkpoint("best_model.pt")
    # Reload saved weights and simulate continuously across the train/test split.
    metrics, states = free_run(model, *signals)
    measured = torch.cat((y_train, y_test)).numpy()
    # Four CSV columns: seconds, measured voltage, predicted voltage, voltage rate.
    trace = np.column_stack((np.arange(4096) / 610.35, measured, states.numpy()))
    np.savetxt("predictions.csv", trace, delimiter=",", comments="",
        header="time_s,measured_y_v,predicted_y_v,predicted_velocity_v_per_s")
    plot_test_predictions(y_test, states[3072:, 0], metrics["test_rmse"],
                          metrics["zero_test_rmse"], show=False, output_path="test_predictions.png")
    print("TRAIN_RMSE_V = %.8f" % metrics["train_rmse"])
    print("TEST_RMSE_V = %.8f" % metrics["test_rmse"])
    print("ZERO_TEST_RMSE_V = %.8f" % metrics["zero_test_rmse"])
    return metrics
