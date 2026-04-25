"""Tests for integrated dense-traffic macro PPO training."""

from __future__ import annotations

from pathlib import Path
import random

from warehouse_sim.config import load_experiment_config, load_integrated_rl_training_config
from warehouse_sim.integrated.engine import build_integrated_observation
from warehouse_sim.learning.integrated_rl import (
    ConflictGraphMacroPolicyNetwork,
    _teacher_output,
    load_conflict_graph_macro_model,
    load_end_to_end_macro_model,
    run_integrated_rl_training_from_config,
    IntegratedCoordinationRLEnv,
)
from warehouse_sim.simulation import run_experiment_from_path
from warehouse_sim.simulation.models import CoordinationMode, CoordinationRuntimeConfig, ExecutionModel, SimulationConfig
from warehouse_sim.agents import RobotSpec, RobotState
from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.graph import SyntheticGridLayoutConfig, build_synthetic_grid_layout
from warehouse_sim.integrated.planner import ContinuousOccupancyTable
from warehouse_sim.tasks import Task


def test_run_integrated_rl_training_smoke(tmp_path: Path) -> None:
    scenario_path = tmp_path / "scenario.toml"
    scenario_path.write_text(
        """
name = "integrated_smoke"

[layout]
rows = 2
columns = 3

[demand]
horizon_seconds = 180.0
mean_interval = 120.0
seed = 7
min_tasks = 1

[robots]
count = 2

[tasks]
default_service_time_estimate = 5.0

[simulation]
coordination_mode = "integrated"
policy = "prioritized_sipp_coordinator"
horizon_seconds = 180.0
continue_until_all_tasks_complete = true
execution_model = "idealized"

[coordination]
control_dt = 0.25
replan_period = 1.0
robot_radius = 0.2
collision_clearance = 0.05
k_shortest_paths = 2
max_route_options_per_pair = 2

[reporting]
output_dir = "outputs/integrated_smoke"
write_plots = false
""".strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "integrated_rl.toml"
    config_path.write_text(
        f"""
name = "integrated_rl_smoke"

[curriculum]
scenario_configs = ["{scenario_path.name}"]
train_seeds = [7]
validation_seeds = [11]
scenario_weights = {{ integrated_smoke = 2.0 }}

[model]
hidden_dim = 32
warehouse_message_passing_layers = 1
conflict_message_passing_layers = 1
dropout = 0.0
top_k_conflicting_robots = 2

[reward]
task_completion = 1.0
waiting_time = -0.01
congestion_delay = -0.02
safety_violation = -1.0
path_conflict = -0.05
planner_wait_time = -0.01
wait_insertion_time = -0.01

[ppo]
learning_rate = 0.0003
clip_epsilon = 0.2
gamma = 0.99
gae_lambda = 0.95
ppo_epochs = 1
total_episodes = 1
learner_minibatch_size = 2

[runtime]
device = "cpu"
rollout_workers = 1
episodes_per_sync = 1
inference_batch_size = 2

[warm_start]
epochs = 1
learning_rate = 0.001
teacher_policy = "prioritized_sipp_coordinator"
teacher_mixture = {{ prioritized_sipp_coordinator = 1.0, optimal_mapf_coordinator = 0.5 }}

[benchmark_gate]
max_safety_violations = 0
min_task_completion_rate = 0.0
min_throughput_ratio_vs_baseline = 0.0
min_policy_distinctness_vs_teacher = 0.0

[output]
output_dir = "integrated_rl_outputs"
""".strip(),
        encoding="utf-8",
    )

    written = run_integrated_rl_training_from_config(load_integrated_rl_training_config(config_path))
    loaded = load_conflict_graph_macro_model(written["artifact"])

    assert written["artifact"].exists()
    assert written["checkpoint"].exists()
    assert written["training_metrics"].exists()
    assert written["warm_start_metrics"].exists()
    assert written["evaluation_rollouts"].exists()
    assert loaded.artifact.model_type == "conflict_graph_macro_ppo"
    assert loaded.artifact.metadata["selected_checkpoint_stage"] in {"warm_start", "ppo_final", "initial"}
    assert loaded.artifact.metadata["decoder_type"] == "parallel_matching"
    assert loaded.artifact.metadata["training_runtime"] == "mps_learner_cpu_actors"
    assert "warm_start" in loaded.artifact.metadata["candidate_gate_evaluations"]
    assert "ppo_final" in loaded.artifact.metadata["candidate_gate_evaluations"]
    assert "observed_policy_distinctness_vs_teacher" in loaded.artifact.metadata["benchmark_gate"]
    assert loaded.artifact.metadata["runtime"]["rollout_workers"] == 1


def test_run_experiment_with_trained_conflict_graph_macro_policy(tmp_path: Path) -> None:
    test_run_integrated_rl_training_smoke(tmp_path)
    artifact_path = tmp_path / "integrated_rl_outputs" / "model_artifact.json"
    config_path = tmp_path / "trained_integrated.toml"
    config_path.write_text(
        f"""
name = "trained_integrated"

[layout]
rows = 2
columns = 3

[demand]
horizon_seconds = 180.0
mean_interval = 120.0
seed = 7
min_tasks = 1

[robots]
count = 2

[tasks]
default_service_time_estimate = 5.0

[simulation]
coordination_mode = "integrated"
policy = "trained_conflict_graph_macro_ppo"
horizon_seconds = 180.0
continue_until_all_tasks_complete = true
execution_model = "idealized"

[coordination]
control_dt = 0.25
replan_period = 1.0
robot_radius = 0.2
collision_clearance = 0.05
k_shortest_paths = 2
max_route_options_per_pair = 2

[policy_model]
artifact_path = "integrated_rl_outputs/{artifact_path.name}"

[reporting]
output_dir = "outputs/trained_integrated"
write_plots = false
""".strip(),
        encoding="utf-8",
    )

    result, written = run_experiment_from_path(
        config_path=config_path,
        output_dir_override=tmp_path / "trained_run",
        force_write_plots=False,
        force_write_observation_dataset=False,
    )

    assert result.policy_name == "trained_conflict_graph_macro_ppo"
    assert written["robot_trajectories"].exists()


def test_trained_end_to_end_alias_loads_new_conflict_graph_artifact(tmp_path: Path) -> None:
    test_run_integrated_rl_training_smoke(tmp_path)
    loaded = load_end_to_end_macro_model(tmp_path / "integrated_rl_outputs" / "model_artifact.json")

    assert loaded.artifact.model_type == "conflict_graph_macro_ppo"


def test_integrated_rl_training_rerun_reuses_completed_checkpoint(tmp_path: Path) -> None:
    test_run_integrated_rl_training_smoke(tmp_path)
    config = load_integrated_rl_training_config(tmp_path / "integrated_rl.toml")
    written = run_integrated_rl_training_from_config(config)

    assert written["artifact"].exists()
    assert written["checkpoint"].exists()


def test_parallel_decoder_avoids_duplicate_task_assignment() -> None:
    environment = WarehouseEnvironment(build_synthetic_grid_layout(SyntheticGridLayoutConfig(rows=2, columns=3)))
    tasks = (
        Task(
            task_id="shared_task",
            release_time=0.0,
            pickup_node="r0_c1",
            dropoff_node="r0_c2",
            service_time_estimate=0.0,
        ),
    )
    robots = (
        RobotState.from_spec(RobotSpec(robot_id="robot_1", initial_node="r0_c0")),
        RobotState.from_spec(RobotSpec(robot_id="robot_2", initial_node="r1_c0")),
    )
    config = SimulationConfig(
        coordination_mode=CoordinationMode.INTEGRATED,
        execution_model=ExecutionModel.IDEALIZED,
        coordination=CoordinationRuntimeConfig(
            control_dt=0.25,
            replan_period=1.0,
            robot_radius=0.2,
            collision_clearance=0.05,
            k_shortest_paths=2,
            max_route_options_per_pair=2,
        ),
    )
    observation = build_integrated_observation(
        environment=environment,
        robot_states=robots,
        tasks=tasks,
        released_task_ids={"shared_task"},
        claimed_task_ids=set(),
        completed_task_ids=set(),
        active_plans={},
        occupancy=ContinuousOccupancyTable(robot_radius=0.2, collision_clearance=0.05),
        current_time=0.0,
        config=config,
    )
    model = ConflictGraphMacroPolicyNetwork(
        node_dim=len(observation.node_features[0]),
        edge_dim=len(observation.edge_features[0]) if observation.edge_features else 3,
        robot_dim=len(observation.robot_features[0]),
        task_dim=len(observation.task_features[0]) if observation.task_features else 5,
        macro_dim=13,
        density_dim=len(observation.global_density_features) if observation.global_density_features else 1,
        robot_robot_edge_dim=len(observation.robot_robot_conflict_features[0]) if observation.robot_robot_conflict_features else 6,
        robot_macro_edge_dim=len(observation.robot_macro_incidence_features[0]) if observation.robot_macro_incidence_features else 6,
        macro_conflict_edge_dim=len(observation.macro_conflict_features[0]) if observation.macro_conflict_features else 6,
        hidden_dim=16,
        warehouse_message_passing_layers=1,
        conflict_message_passing_layers=1,
    )
    output = model.act(observation, greedy=True)
    selected_task_ids = []
    for robot_index, chosen_index in enumerate(output.chosen_indices):
        candidate = observation.macro_candidates[robot_index][chosen_index]
        if candidate.task_id is not None:
            selected_task_ids.append(candidate.task_id)
    assert len(selected_task_ids) == len(set(selected_task_ids))


def test_warm_start_skips_optimal_teacher_on_heavy_dense_scenario() -> None:
    config = load_integrated_rl_training_config(
        Path(__file__).resolve().parents[2] / "configs" / "canonical_artifacts" / "canonical_macro_ppo_training.toml"
    )
    scenario_path = next(
        path for path in config.curriculum.scenario_configs if path.name == "integrated_high_fleet_density_heavy.toml"
    )
    env = IntegratedCoordinationRLEnv(load_experiment_config(scenario_path), 7)
    observation = env.reset()
    teacher_name, _teacher_output_value = _teacher_output(
        env,
        observation,
        {"prioritized_sipp_coordinator": 1.0, "optimal_mapf_coordinator": 0.5},
        rng=random.Random(0),
    )
    assert teacher_name == "prioritized_sipp_coordinator"
