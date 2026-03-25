"""Tests for stable metric-schema validation."""

from __future__ import annotations

import pytest

from warehouse_sim.reporting import METRIC_SCHEMA_VERSION, default_metric_payload, validate_benchmark_run_row


def test_validate_benchmark_run_row_accepts_complete_schema() -> None:
    row = {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "benchmark_name": "example",
        "scenario_family": "test",
        "scenario_id": "scenario_a",
        "scenario_name": "scenario_a",
        "scenario_config": "configs/scenario_a.toml",
        "seed": 7,
        "policy": "fifo",
        "policy_family": "heuristic_dispatch",
        "policy_role": "dispatch_baseline",
        "coordination_mode": "dispatch",
        "execution_model": "idealized",
        "motion_model": "graph_embedded",
        "fleet_size": 2,
        "demand_mean_interval": 60.0,
        "demand_horizon_seconds": 600.0,
        "layout_rows": 5,
        "layout_columns": 5,
        "blocked_cell_count": 0,
        "directed_edge_count": 0,
        "topology_difficulty": "open",
        "summary_path": "outputs/example/summary.json",
        **default_metric_payload(),
    }

    validate_benchmark_run_row(row)


def test_validate_benchmark_run_row_rejects_missing_field() -> None:
    row = {"benchmark_name": "example"}

    with pytest.raises(ValueError):
        validate_benchmark_run_row(row)
