"""Tests for experiment configuration loading."""

from __future__ import annotations

from pathlib import Path

from warehouse_sim.config import load_experiment_config


def test_load_baseline_experiment_config() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "baseline_experiment.toml"
    config = load_experiment_config(config_path)

    assert config.name == "baseline_fifo"
    assert config.layout.rows == 3
    assert config.demand.horizon_seconds == 600.0
    assert config.robots.count == 2
    assert config.simulation.policy == "fifo"
    assert config.reporting.output_dir.as_posix().endswith("outputs/baseline_fifo")
    assert config.reporting.write_observation_dataset is False


def test_load_linear_assignment_experiment_config() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "linear_assignment_experiment.toml"
    config = load_experiment_config(config_path)

    assert config.simulation.policy == "linear_assignment_model"
    assert config.policy_model is not None
    assert config.policy_model.weights["task_age"] == 0.5
