"""Tests for benchmark manifest loading."""

from __future__ import annotations

from pathlib import Path

from warehouse_sim.config import load_benchmark_config


def test_load_policy_benchmark_manifest() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "policy_benchmark.toml"
    config = load_benchmark_config(config_path)

    assert config.name == "baseline_policy_benchmark"
    assert len(config.scenario_configs) == 3
    assert "nearest_robot_task" in config.policies
