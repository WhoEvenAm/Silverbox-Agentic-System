"""The executor's simulated-bench tool: real measurements and saved images."""

from dataclasses import asdict
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from langchain_core.tools import tool

from bench import BenchConfig, search_focus
from schemas import FocusSettings


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def save_camera(path, image, settings):
    fig, axis = plt.subplots(figsize=(5, 4))
    shown = axis.imshow(image, origin="lower", cmap="inferno")
    half = settings.focus_size / 2
    axis.add_patch(Rectangle((settings.focus_x - half, settings.focus_y - half),
                            settings.focus_size, settings.focus_size, fill=False,
                            edgecolor="cyan", linewidth=1))
    axis.set(xlabel="Camera x pixel", ylabel="Camera y pixel")
    fig.colorbar(shown, ax=axis, label="Measured intensity (normalized units)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_comparison(path, initial, final, settings):
    """Show actual before/after values on one scale, with the target marked."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    vmax = max(initial.max(), final.max())
    for axis, image, title in zip(axes, (initial, final), ("Before search", "After search")):
        shown = axis.imshow(image, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
        half = settings.focus_size / 2
        axis.add_patch(Rectangle((settings.focus_x - half, settings.focus_y - half),
                                settings.focus_size, settings.focus_size, fill=False,
                                edgecolor="cyan", linewidth=1))
        axis.set(title=title, xlabel="Camera x pixel", ylabel="Camera y pixel")
    fig.colorbar(shown, ax=list(axes), label="Measured intensity (same scale)", shrink=.8)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_experiment(directory, settings, event=None, bench_config=BenchConfig()):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    numbers = [int(path.name.split("_")[-1]) for path in root.glob("experiment_*")
               if path.is_dir() and path.name.split("_")[-1].isdigit()]
    folder = root / f"experiment_{max(numbers, default=0) + 1:02d}"
    folder.mkdir(exist_ok=False)
    started = time.perf_counter()
    write_json(folder / "settings.json", settings.model_dump())
    if event:
        event("tool_started", {"tool": "focus_beam", "directory": str(folder),
                               "settings": settings.model_dump()})

    def progress(iteration, row, image):
        save_camera(folder / "live_camera.png", image, settings)
        if event:
            event("experiment_progress", {"directory": str(folder), "iteration": iteration,
                "iterations": settings.iterations, **row, "camera_image": str(folder / "live_camera.png")})

    result = search_focus(settings, config=bench_config, progress=progress)
    # Raw arrays preserve actual camera values independently of plot formatting.
    for key in ("initial_image", "final_image", "initial_pattern", "final_pattern", "focus_mask"):
        np.save(folder / f"{key}.npy", result[key])
    np.save(folder / "background_masks.npy", np.stack(result["background_masks"]))
    save_camera(folder / "initial_camera.png", result["initial_image"], settings)
    save_camera(folder / "final_camera.png", result["final_image"], settings)
    save_comparison(folder / "before_after.png", result["initial_image"], result["final_image"], settings)
    history = result["history"]
    fig, axis = plt.subplots(figsize=(7, 3))
    axis.plot([row["iteration"] for row in history], [row["retained_measurement"] for row in history])
    axis.set(xlabel="Macropixel-flip iteration", ylabel="Measured enhancement",
             title="Noisy feedback during search")
    fig.tight_layout()
    fig.savefig(folder / "enhancement_history.png", dpi=140)
    plt.close(fig)
    write_json(folder / "history.json", history)
    summary = {key: result[key] for key in ("settings", "initial", "final", "camera_shots",
                                           "accepted_flips", "final_active_fraction")}
    summary.update(tool="focus_beam", working_directory=str(folder), bench=asdict(bench_config),
        duration_seconds=time.perf_counter() - started,
        camera_images={"initial": str(folder / "initial_camera.png"), "final": str(folder / "final_camera.png"),
                       "comparison": str(folder / "before_after.png")})
    # Returned enhancement is the independent validation-shot mean, not the
    # luckiest noisy optimization score. Keep its uncertainty and all raw shots.
    shots = np.asarray(summary["final"]["shot_enhancements"])
    if not np.isfinite(shots).all() or not np.isclose(shots.mean(), summary["final"]["enhancement"]):
        raise ValueError("Final enhancement does not match its saved validation measurements")
    summary["verified"] = True
    write_json(folder / "result.json", summary)
    if event:
        event("tool_completed", summary)
    return summary


def make_focusing_tool(directory, event=None):
    @tool(args_schema=FocusSettings)
    def focus_beam(**settings) -> dict:
        """Optimize binary DMD macropixels on the noisy bench; return enhancement and camera image paths."""
        return run_experiment(directory, FocusSettings.model_validate(settings), event=event)
    return focus_beam
