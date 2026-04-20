"""Tests for dispatch-event RL environment and PPO fine-tuning."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from warehouse_sim.agents import RobotSpec
from warehouse_sim.config import load_rl_fine_tuning_config
from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.graph import NodeType, SyntheticGridLayoutConfig, build_synthetic_grid_layout
from warehouse_sim.learning.artifacts import DispatchModelArtifact, write_dispatch_model_artifact
from warehouse_sim.learning.graph_data import (
    DEFAULT_GRAPH_CANDIDATE_FEATURES,
    DEFAULT_GRAPH_EDGE_FEATURES,
    DEFAULT_GRAPH_NODE_FEATURES,
)
from warehouse_sim.learning.graph_model import GraphDispatchScorer
from warehouse_sim.learning.rl import DispatchEventRLEnv, run_rl_fine_tuning_from_config
from warehouse_sim.simulation import SimulationConfig
from warehouse_sim.tasks import Task


def test_dispatch_event_rl_env_mask_and_reward(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    environment = WarehouseEnvironment(
        build_synthetic_grid_layout(
            SyntheticGridLayoutConfig(
                rows=2,
                columns=3,
                special_node_types={(0, 0): NodeType.STORAGE, (1, 2): NodeType.DROPOFF},
            )
        )
    )
    env = DispatchEventRLEnv(
        environment=environment,
        tasks=(
            Task(task_id="task_1", release_time=0.0, pickup_node="r0_c0", dropoff_node="r1_c2", service_time_estimate=1.0),
            Task(task_id="task_2", release_time=0.0, pickup_node="r0_c1", dropoff_node="r1_c2", service_time_estimate=1.0),
        ),
        robots=(
            RobotSpec(robot_id="robot_1", initial_node="r1_c0"),
            RobotSpec(robot_id="robot_2", initial_node="r1_c1"),
        ),
        simulation_config=SimulationConfig(),
        reward_weights=type("RewardWeights", (), {"task_completion": 1.0, "waiting_time": -0.01, "congestion_delay": -0.02, "blocked_events": -0.05})(),
        artifact=artifact,
    )

    observation, info = env.reset()
    assert observation is not None
    assert info["action_mask"].shape[0] == observation.candidate_count

    next_observation, reward, done, _, step_info = env.step(0)
    assert isinstance(reward, float)
    assert "reward_components" in step_info
    with pytest.raises(ValueError):
        env.step(999)
    assert next_observation is None or next_observation.candidate_count >= 1 or done


def test_run_rl_fine_tuning_smoke(tmp_path: Path) -> None:
    artifact_path = write_dispatch_model_artifact(_artifact(tmp_path), tmp_path / "pretrained_artifact.json")
    scenario_path = tmp_path / "scenario.toml"
    scenario_path.write_text(
        """
name = "rl_scenario"

[layout]
rows = 3
columns = 3

[demand]
horizon_seconds = 300.0
mean_interval = 120.0
seed = 7

[robots]
count = 2

[tasks]
default_service_time_estimate = 30.0

[simulation]
policy = "fifo"
horizon_seconds = 300.0
continue_until_all_tasks_complete = true

[reporting]
output_dir = "outputs/rl_scenario"
write_plots = false
write_observation_dataset = false
""".strip(),
        encoding="utf-8",
    )
    rl_config_path = tmp_path / "rl.toml"
    rl_config_path.write_text(
        f"""
name = "graph_rl_smoke"
pretrained_artifact_path = "{artifact_path.name}"

[curriculum]
scenario_configs = ["{scenario_path.name}"]
train_seeds = [7]
validation_seeds = [11]

[reward]
task_completion = 1.0
waiting_time = -0.01
congestion_delay = -0.02
blocked_events = -0.05

[ppo]
learning_rate = 0.0005
clip_epsilon = 0.2
gamma = 0.99
gae_lambda = 0.95
ppo_epochs = 1
rollout_horizon = 1
total_episodes = 1

[output]
output_dir = "rl_outputs"
""".strip(),
        encoding="utf-8",
    )

    written = run_rl_fine_tuning_from_config(load_rl_fine_tuning_config(rl_config_path))

    assert written["artifact"].exists()
    assert written["checkpoint"].exists()
    assert written["training_metrics"].exists()
    assert written["evaluation_rollouts"].exists()


def _artifact(tmp_path: Path) -> DispatchModelArtifact:
    model = GraphDispatchScorer(
        node_dim=len(DEFAULT_GRAPH_NODE_FEATURES),
        edge_dim=len(DEFAULT_GRAPH_EDGE_FEATURES),
        candidate_dim=len(DEFAULT_GRAPH_CANDIDATE_FEATURES),
        hidden_dim=16,
        message_passing_layers=2,
        dropout=0.0,
    )
    state_path = tmp_path / "graph_dispatch_model.pt"
    torch.save(model.state_dict(), state_path)
    return DispatchModelArtifact(
        artifact_version=2,
        model_type="pyg_graph_dispatch",
        objective="dispatch_group_softmax_cross_entropy",
        feature_names=DEFAULT_GRAPH_CANDIDATE_FEATURES,
        parameters={
            "node_feature_names": list(DEFAULT_GRAPH_NODE_FEATURES),
            "edge_feature_names": list(DEFAULT_GRAPH_EDGE_FEATURES),
            "candidate_feature_names": list(DEFAULT_GRAPH_CANDIDATE_FEATURES),
            "node_dim": len(DEFAULT_GRAPH_NODE_FEATURES),
            "edge_dim": len(DEFAULT_GRAPH_EDGE_FEATURES),
            "candidate_dim": len(DEFAULT_GRAPH_CANDIDATE_FEATURES),
            "hidden_dim": 16,
            "message_passing_layers": 2,
            "dropout": 0.0,
            "state_dict_path": state_path.name,
        },
        metadata={"training": {"parameter_count": sum(parameter.numel() for parameter in model.parameters())}},
    )
