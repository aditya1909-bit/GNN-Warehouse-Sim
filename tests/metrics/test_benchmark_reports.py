"""Tests for repeated-seed benchmark aggregation outputs."""

from __future__ import annotations

import json
from pathlib import Path

from warehouse_sim.metrics.benchmark_reports import write_benchmark_report
from warehouse_sim.reporting import METRIC_SCHEMA_VERSION


def test_write_benchmark_report_aggregates_seed_statistics(tmp_path: Path) -> None:
    rows = [
        _row("scenario_a", "fifo", "dispatch_baseline", 1, throughput=8.0, completion=10.0),
        _row("scenario_a", "fifo", "dispatch_baseline", 2, throughput=10.0, completion=14.0),
        _row("scenario_a", "trained_linear_model", "dispatch_learned", 1, throughput=11.0, completion=8.0),
        _row("scenario_a", "trained_linear_model", "dispatch_learned", 2, throughput=12.0, completion=6.0),
    ]

    written = write_benchmark_report(
        tmp_path,
        "seeded_benchmark",
        rows,
        config_sources={"benchmark": "[benchmark]\nname = 'seeded_benchmark'\n"},
        seed_bundle={"scenario_seeds": {"scenario_a": [1, 2]}},
    )

    assert written["summary_csv"].exists()
    assert written["aggregate_csv"].exists()
    assert written["claims_csv"].exists()
    assert written["manifest"].exists()
    payload = json.loads(written["summary_json"].read_text(encoding="utf-8"))

    aggregate_rows = payload["aggregates"]
    fifo_row = next(row for row in aggregate_rows if row["policy"] == "fifo")
    trained_row = next(row for row in aggregate_rows if row["policy"] == "trained_linear_model")
    assert fifo_row["throughput_mean"] == 9.0
    assert trained_row["mean_task_completion_time_mean"] == 7.0
    assert payload["metric_schema_version"] == METRIC_SCHEMA_VERSION
    assert payload["best_by_scenario"]["scenario_a"]["policy"] == "trained_linear_model"
    assert payload["claims"][0]["winner"] == "challenger"
    assert len(payload["per_seed_breakdown"]["scenario_a"]["fifo"]) == 2


def _row(
    scenario_name: str,
    policy: str,
    policy_role: str,
    seed: int,
    *,
    throughput: float,
    completion: float,
) -> dict[str, object]:
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "benchmark_name": "seeded_benchmark",
        "scenario_family": "test_family",
        "scenario_id": scenario_name,
        "scenario_name": scenario_name,
        "scenario_config": f"configs/{scenario_name}.toml",
        "seed": seed,
        "policy": policy,
        "policy_family": "learned_dispatch" if "trained" in policy else "heuristic_dispatch",
        "policy_role": policy_role,
        "coordination_mode": "dispatch",
        "execution_model": "idealized",
        "motion_model": "graph_embedded",
        "fleet_size": 3,
        "demand_mean_interval": 60.0,
        "demand_horizon_seconds": 600.0,
        "layout_rows": 5,
        "layout_columns": 5,
        "blocked_cell_count": 0,
        "directed_edge_count": 0,
        "topology_difficulty": "open",
        "summary_path": f"outputs/{scenario_name}/{policy}/summary.json",
        "throughput": throughput,
        "mean_task_completion_time": completion,
        "p95_task_completion_time": completion + 3.0,
        "makespan": 100.0,
        "mean_queue_length": 1.0,
        "p95_queue_length": 2.0,
        "robot_idle_fraction": 0.2,
        "travel_distance_per_completed_task": 3.0,
        "realized_waiting_time": 0.0,
        "congestion_event_count": 0,
        "collision_event_count": 0,
        "deadlock_livelock_incident_count": 0,
        "planning_latency": None,
        "replanning_count": 0,
        "planner_failure_count": 0,
        "timeout_count": 0,
        "path_conflict_count_before_resolution": None,
        "sipp_wait_insertion_count": None,
        "mapf_solve_success_rate": None,
        "reward_mean": None,
        "reward_std": None,
        "policy_entropy": None,
        "invalid_action_rate": None,
        "masked_action_rejection_rate": None,
        "ppo_kl": None,
        "ppo_clip_fraction": None,
        "value_loss": None,
        "generalization_gap_seen_vs_unseen": None,
    }
