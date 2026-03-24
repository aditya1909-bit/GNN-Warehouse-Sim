"""Tests for repeated-seed benchmark aggregation outputs."""

from __future__ import annotations

import json
from pathlib import Path

from warehouse_sim.metrics.benchmark_reports import write_benchmark_report


def test_write_benchmark_report_aggregates_seed_statistics(tmp_path: Path) -> None:
    rows = [
        _row("scenario_a", "fifo", 1, tasks_completed=8, waiting=10.0),
        _row("scenario_a", "fifo", 2, tasks_completed=10, waiting=14.0),
        _row("scenario_a", "trained_linear_model", 1, tasks_completed=11, waiting=8.0),
        _row("scenario_a", "trained_linear_model", 2, tasks_completed=12, waiting=6.0),
    ]

    written = write_benchmark_report(tmp_path, "seeded_benchmark", rows)

    assert written["summary_csv"].exists()
    assert written["aggregate_csv"].exists()
    payload = json.loads(written["summary_json"].read_text(encoding="utf-8"))

    aggregate_rows = payload["aggregates"]
    fifo_row = next(row for row in aggregate_rows if row["policy"] == "fifo")
    trained_row = next(row for row in aggregate_rows if row["policy"] == "trained_linear_model")
    assert fifo_row["tasks_completed_mean"] == 9.0
    assert trained_row["average_waiting_time_mean"] == 7.0
    assert payload["best_by_scenario"]["scenario_a"]["policy"] == "trained_linear_model"
    assert len(payload["per_seed_breakdown"]["scenario_a"]["fifo"]) == 2


def _row(
    scenario_name: str,
    policy: str,
    seed: int,
    *,
    tasks_completed: int,
    waiting: float,
) -> dict[str, object]:
    return {
        "scenario_name": scenario_name,
        "scenario_config": f"configs/{scenario_name}.toml",
        "seed": seed,
        "policy": policy,
        "execution_model": "idealized",
        "tasks_generated": 12,
        "tasks_completed": tasks_completed,
        "tasks_unassigned": 12 - tasks_completed,
        "average_waiting_time": waiting,
        "average_turnaround_time": waiting + 2.0,
        "average_travel_distance_per_task": 3.0,
        "realized_travel_time_total": 50.0,
        "realized_travel_distance_total": 25.0,
        "congestion_delay_total": 0.0,
        "average_congestion_delay_per_completed_task": 0.0,
        "blocked_traversal_events_total": 0,
        "average_queue_length": 1.0,
        "throughput_per_hour": 40.0,
        "makespan": 100.0,
        "summary_path": f"outputs/{scenario_name}/{policy}/summary.json",
    }
