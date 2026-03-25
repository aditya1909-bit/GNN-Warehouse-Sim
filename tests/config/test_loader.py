"""Tests for experiment configuration loading."""

from __future__ import annotations

from pathlib import Path

from warehouse_sim.config import (
    load_experiment_config,
    load_integrated_rl_training_config,
    load_offline_training_config,
    load_rl_fine_tuning_config,
)


def test_load_baseline_experiment_config() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "baseline_experiment.toml"
    config = load_experiment_config(config_path)

    assert config.name == "baseline_fifo"
    assert config.layout.rows == 3
    assert config.demand.horizon_seconds == 600.0
    assert config.robots.count == 2
    assert config.simulation.policy == "fifo"
    assert config.simulation.execution_model == "idealized"
    assert config.reporting.output_dir.as_posix().endswith("outputs/baseline_fifo")
    assert config.reporting.write_observation_dataset is False


def test_load_linear_assignment_experiment_config() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "linear_assignment_experiment.toml"
    config = load_experiment_config(config_path)

    assert config.simulation.policy == "linear_assignment_model"
    assert config.policy_model is not None
    assert config.policy_model.weights["task_age"] == 0.5


def test_load_congestion_scenario_config() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "scenarios" / "narrow_bottleneck.toml"
    config = load_experiment_config(config_path)

    assert config.name == "narrow_bottleneck"
    assert config.simulation.execution_model == "reserved_edges"


def test_load_offline_graph_dispatch_config() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "offline_graph_dispatch_fit.toml"
    config = load_offline_training_config(config_path)

    assert config.model.type == "graph_dispatch"
    assert config.dataset.node_feature_names is not None
    assert config.dataset.edge_feature_names is not None
    assert config.model.message_passing_layers == 2


def test_load_graph_dispatch_rl_config() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "graph_dispatch_rl_fine_tune.toml"
    config = load_rl_fine_tuning_config(config_path)

    assert config.name == "graph_dispatch_rl_fine_tune"
    assert len(config.curriculum.scenario_configs) == 2
    assert config.ppo.rollout_horizon == 8


def test_load_integrated_experiment_config() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "scenarios" / "integrated_narrow_bottleneck.toml"
    config = load_experiment_config(config_path)

    assert config.simulation.coordination_mode == "integrated"
    assert config.simulation.policy == "prioritized_sipp_coordinator"
    assert config.coordination is not None
    assert config.coordination.robot_radius == 0.2


def test_load_integrated_optimal_mapf_experiment_config(tmp_path: Path) -> None:
    config_path = tmp_path / "integrated_optimal.toml"
    config_path.write_text(
        """
name = "integrated_optimal"

[layout]
rows = 3
columns = 3

[demand]
horizon_seconds = 60.0
mean_interval = 30.0
seed = 7

[robots]
count = 2

[tasks]

[simulation]
coordination_mode = "integrated"
policy = "optimal_mapf_coordinator"
execution_model = "idealized"

[coordination]
control_dt = 0.25
replan_period = 1.0
robot_radius = 0.2
collision_clearance = 0.05
k_shortest_paths = 2
max_route_options_per_pair = 2

[reporting]
output_dir = "outputs/integrated_optimal"
""".strip(),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.simulation.coordination_mode == "integrated"
    assert config.simulation.policy == "optimal_mapf_coordinator"
    assert config.coordination is not None


def test_load_integrated_rl_training_config() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "integrated_macro_ppo_training.toml"
    config = load_integrated_rl_training_config(config_path)

    assert config.name == "integrated_macro_ppo_training"
    assert len(config.curriculum.scenario_configs) == 2
    assert config.benchmark_gate.min_task_completion_rate == 0.98
