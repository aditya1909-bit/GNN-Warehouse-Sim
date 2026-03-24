"""Machine-readable report writers for simulation runs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from warehouse_sim.simulation.models import SimulationResult


def write_simulation_report(
    output_dir: Path,
    result: "SimulationResult",
    experiment_name: str,
) -> dict[str, Path]:
    """Write summary and time-series artifacts for a simulation result."""

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    executions_path = output_dir / "executions.csv"
    queue_path = output_dir / "queue_snapshots.csv"
    robot_metrics_path = output_dir / "robot_metrics.csv"

    summary_payload = {
        "experiment_name": experiment_name,
        "policy_name": result.policy_name,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "metrics": asdict(result.metrics),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    _write_csv(executions_path, [asdict(execution) for execution in result.executions])
    _write_csv(queue_path, [asdict(snapshot) for snapshot in result.queue_snapshots])
    _write_csv(robot_metrics_path, [asdict(metric) for metric in result.metrics.robot_metrics])

    return {
        "summary": summary_path,
        "executions": executions_path,
        "queue_snapshots": queue_path,
        "robot_metrics": robot_metrics_path,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
