"""Plotting utilities for simulation reports."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from warehouse_sim.simulation.models import SimulationResult


def prepare_queue_length_series(result: "SimulationResult") -> dict[str, list[float]]:
    """Prepare event-time queue-length data for plotting."""

    return {
        "time": [snapshot.time for snapshot in result.queue_snapshots],
        "ready_tasks": [float(snapshot.ready_tasks) for snapshot in result.queue_snapshots],
        "busy_robots": [float(snapshot.busy_robots) for snapshot in result.queue_snapshots],
    }


def prepare_robot_utilization_series(result: "SimulationResult") -> dict[str, list[float | str]]:
    """Prepare per-robot utilization data for plotting."""

    return {
        "robot_id": [metric.robot_id for metric in result.metrics.robot_metrics],
        "utilization": [metric.utilization for metric in result.metrics.robot_metrics],
    }


def write_default_plots(output_dir: Path, result: "SimulationResult") -> tuple[Path, Path]:
    """Write baseline PNG plots using a headless matplotlib backend."""

    plt = _load_matplotlib_pyplot()

    output_dir.mkdir(parents=True, exist_ok=True)
    queue_data = prepare_queue_length_series(result)
    utilization_data = prepare_robot_utilization_series(result)

    queue_path = output_dir / "queue_length.png"
    plt.figure(figsize=(8, 4))
    plt.step(queue_data["time"], queue_data["ready_tasks"], where="post")
    plt.xlabel("Time (s)")
    plt.ylabel("Ready tasks")
    plt.title("Queue Length Over Time")
    plt.tight_layout()
    plt.savefig(queue_path)
    plt.close()

    utilization_path = output_dir / "robot_utilization.png"
    plt.figure(figsize=(8, 4))
    plt.bar(utilization_data["robot_id"], utilization_data["utilization"])
    plt.xlabel("Robot")
    plt.ylabel("Utilization")
    plt.title("Robot Utilization")
    plt.tight_layout()
    plt.savefig(utilization_path)
    plt.close()

    return queue_path, utilization_path


def _load_matplotlib_pyplot():
    try:
        cache_root = Path(tempfile.gettempdir()) / "warehouse_sim_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(
            "MPLCONFIGDIR",
            str(cache_root / "mplconfig"),
        )
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to write plots. Install the standard package dependencies."
        ) from exc
    return plt
