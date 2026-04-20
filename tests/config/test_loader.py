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


def test_load_integrated_free_space_experiment_config(tmp_path: Path) -> None:
    config_path = tmp_path / "integrated_free_space.toml"
    config_path.write_text(
        """
name = "integrated_free_space"

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
policy = "prioritized_sipp_coordinator"
execution_model = "idealized"

[coordination]
motion_model = "free_space"
control_dt = 0.25
replan_period = 1.0
robot_radius = 0.2
collision_clearance = 0.05
k_shortest_paths = 2
max_route_options_per_pair = 2

[reporting]
output_dir = "outputs/integrated_free_space"
""".strip(),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.coordination is not None
    assert config.coordination.motion_model == "free_space"


def test_load_integrated_obstacle_aware_free_space_experiment_config() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "scenarios" / "integrated_obstacle_slalom.toml"
    config = load_experiment_config(config_path)

    assert config.coordination is not None
    assert config.coordination.motion_model == "obstacle_aware_free_space"


def test_load_battery_and_polygon_layout_config(tmp_path: Path) -> None:
    config_path = tmp_path / "battery_polygon.toml"
    config_path.write_text(
        """
name = "battery_polygon"

[layout]
rows = 4
columns = 4
charging_cells = [[3, 0]]
obstacle_polygons = [[[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]]]

[demand]
horizon_seconds = 60.0
mean_interval = 30.0
seed = 7

[robots]
count = 1

[tasks]

[simulation]
policy = "fifo"

[battery]
enabled = true
capacity = 12.0
initial_charge_fraction = 0.25
travel_energy_per_distance = 1.5
service_energy = 0.5
charge_rate = 4.0
dispatch_charge_threshold = 0.4
minimum_reserve_fraction = 0.15

[reporting]
output_dir = "outputs/battery_polygon"
""".strip(),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.layout.charging_cells == ((3, 0),)
    assert config.layout.obstacle_polygons == (((0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)),)
    assert config.battery is not None
    assert config.battery.enabled is True
    assert config.battery.capacity == 12.0


def test_load_integrated_rl_training_config() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "integrated_macro_ppo_training.toml"
    config = load_integrated_rl_training_config(config_path)

    assert config.name == "integrated_macro_ppo_training"
    assert len(config.curriculum.scenario_configs) == 2
    assert config.benchmark_gate.min_task_completion_rate == 0.98
    assert config.warm_start.epochs == 0
