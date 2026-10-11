"""Check printed scores against actual newly trained Silverbox checkpoints."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import sys


def printed_number(stdout, name):
    matches = re.findall(rf"^{re.escape(name)}\s*=\s*([-+0-9.eE]+)\s*$", stdout, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name} line in actual stdout")
    value = float(matches[0])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def artifact_path(directory, name):
    path = (directory / name).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError(f"Missing artifact or path outside working directory: {name}")
    return path


def verify_execution(execution, plan, support_directory):
    if execution["exit_code"] != 0 or execution["timed_out"]:
        raise ValueError("The script must exit successfully before acceptance")
    helper_path = str(Path(support_directory).resolve() / "src")
    if helper_path not in sys.path:
        sys.path.insert(0, helper_path)
    import numpy as np
    import torch
    from training_utils import TrainingConfig, data_tensors, free_run, load_checkpoint

    torch.set_num_threads(4)
    directory = Path(execution["working_directory"]).resolve()
    rows = json.loads(artifact_path(directory, "results.json").read_text())
    expected = [asdict(TrainingConfig(hidden_sizes=tuple(widths), **setting.model_dump()))
                for widths in plan.architectures for setting in plan.settings]
    expected = json.loads(json.dumps(expected))  # JSON converts tuples into lists.
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("Not every planned configuration completed")
    signals = data_tensors()
    recomputed = []
    for row in rows:
        config = row["config"]
        if config not in expected:
            raise ValueError("A run changed or duplicated the scientist's settings")
        expected.remove(config)
        model, checkpoint = load_checkpoint(artifact_path(directory, row["checkpoint"]))
        if json.loads(json.dumps(checkpoint["config"])) != config:
            raise ValueError("Checkpoint and comparison settings disagree")
        metrics, _ = free_run(model, *signals)
        for key in ("train_rmse", "test_rmse", "zero_test_rmse"):
            if not math.isclose(float(row[key]), metrics[key], rel_tol=0, abs_tol=1e-7):
                raise ValueError(f"Saved {key} does not match its checkpoint")
        recomputed.append((row, metrics, checkpoint))
    best_row, _, best_checkpoint = min(recomputed, key=lambda item: item[1]["test_rmse"])
    selected, selected_checkpoint = load_checkpoint(artifact_path(directory, "best_model.pt"))
    if any(not torch.equal(value, selected_checkpoint["state_dict"][key])
           for key, value in best_checkpoint["state_dict"].items()):
        raise ValueError("Selected weights are not the lowest-RMSE checkpoint")
    metrics, states = free_run(selected, *signals)
    for name, key in (("TRAIN_RMSE_V", "train_rmse"), ("TEST_RMSE_V", "test_rmse"),
                      ("ZERO_TEST_RMSE_V", "zero_test_rmse")):
        if not math.isclose(printed_number(execution["stdout"], name), metrics[key],
                            rel_tol=0, abs_tol=1e-7):
            raise ValueError(f"Printed {name} does not match the actual free run")
    trace = np.loadtxt(artifact_path(directory, "predictions.csv"), delimiter=",", skiprows=1)
    measured = torch.cat((signals[1], signals[3])).numpy()
    expected_trace = np.column_stack((np.arange(4096) / 610.35, measured, states.numpy()))
    if trace.shape != expected_trace.shape or not np.allclose(trace, expected_trace, atol=1e-7, rtol=0):
        raise ValueError("Saved predictions do not match the continuous free run")
    artifact_path(directory, "test_predictions.png")
    return {"verified": True, "metrics": metrics, "best_config": best_row["config"],
            "completed_fits": len(rows)}
