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
    robot_trajectories_path = output_dir / "robot_trajectories.csv"
    macro_decisions_path = output_dir / "macro_decisions.csv"
    collision_events_path = output_dir / "collision_events.csv"
    planner_plans_path = output_dir / "planner_plans.csv"

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
    _write_csv(robot_trajectories_path, [asdict(record) for record in result.robot_trajectories])
    _write_csv(macro_decisions_path, [asdict(record) for record in result.macro_decisions])
    _write_csv(collision_events_path, [asdict(record) for record in result.collision_events])
    _write_csv(planner_plans_path, [asdict(record) for record in result.planner_plans])

    written = {
        "summary": summary_path,
        "executions": executions_path,
        "queue_snapshots": queue_path,
        "robot_metrics": robot_metrics_path,
    }
    if result.robot_trajectories:
        written["robot_trajectories"] = robot_trajectories_path
    if result.macro_decisions:
        written["macro_decisions"] = macro_decisions_path
    if result.collision_events or result.planner_plans:
        written["collision_events"] = collision_events_path
        written["planner_plans"] = planner_plans_path
    return written


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
