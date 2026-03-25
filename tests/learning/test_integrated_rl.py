"""Tests for integrated end-to-end macro PPO training."""

from __future__ import annotations

from pathlib import Path

from warehouse_sim.config import load_integrated_rl_training_config
from warehouse_sim.learning.integrated_rl import load_end_to_end_macro_model, run_integrated_rl_training_from_config
from warehouse_sim.simulation import run_experiment_from_path


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

[reward]
task_completion = 1.0
waiting_time = -0.01
congestion_delay = -0.02
safety_violation = -1.0

[ppo]
learning_rate = 0.0003
clip_epsilon = 0.2
gamma = 0.99
gae_lambda = 0.95
ppo_epochs = 1
total_episodes = 1

[warm_start]
epochs = 1
learning_rate = 0.001
teacher_policy = "prioritized_sipp_coordinator"

[benchmark_gate]
max_safety_violations = 0
min_task_completion_rate = 0.0
min_throughput_ratio_vs_baseline = 0.0

[output]
output_dir = "integrated_rl_outputs"
""".strip(),
        encoding="utf-8",
    )

    written = run_integrated_rl_training_from_config(load_integrated_rl_training_config(config_path))
    loaded = load_end_to_end_macro_model(written["artifact"])

    assert written["artifact"].exists()
    assert written["checkpoint"].exists()
    assert written["training_metrics"].exists()
    assert written["warm_start_metrics"].exists()
    assert written["evaluation_rollouts"].exists()
    assert loaded.artifact.model_type == "end_to_end_macro_ppo"
    assert loaded.artifact.metadata["selected_checkpoint_stage"] in {"warm_start", "ppo_final", "initial"}
    assert "warm_start" in loaded.artifact.metadata["candidate_gate_evaluations"]
    assert "ppo_final" in loaded.artifact.metadata["candidate_gate_evaluations"]


def test_run_experiment_with_trained_end_to_end_macro_policy(tmp_path: Path) -> None:
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
policy = "trained_end_to_end_macro_ppo"
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

    assert result.policy_name == "trained_end_to_end_macro_ppo"
    assert written["robot_trajectories"].exists()
