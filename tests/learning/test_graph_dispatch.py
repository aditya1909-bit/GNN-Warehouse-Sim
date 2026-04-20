"""Tests for graph-conditioned dispatch data and model training."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from warehouse_sim.agents import RobotSpec
from warehouse_sim.learning.artifacts import DispatchModelArtifact, write_dispatch_model_artifact
from warehouse_sim.learning.graph_data import (
    DEFAULT_GRAPH_CANDIDATE_FEATURES,
    DEFAULT_GRAPH_EDGE_FEATURES,
    DEFAULT_GRAPH_NODE_FEATURES,
    load_graph_dispatch_dataset,
)
from warehouse_sim.learning.graph_evaluation import evaluate_graph_dispatch_artifact
from warehouse_sim.learning.graph_fit import GraphDispatchFitConfig, fit_graph_dispatch_model
from warehouse_sim.learning.graph_model import GraphDispatchScorer, load_graph_dispatch_model
from warehouse_sim.learning.splits import SplitConfig, split_dispatch_observation_dataset
from warehouse_sim.metrics import write_observation_dataset
from warehouse_sim.policies import FIFODispatchPolicy
from warehouse_sim.simulation import SimulationConfig, run_experiment_from_path, run_simulation
from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.graph import NodeType, SyntheticGridLayoutConfig, build_synthetic_grid_layout
from warehouse_sim.tasks import Task


def test_write_and_load_graph_dispatch_dataset(tmp_path: Path) -> None:
    written = _write_dataset(tmp_path)

    assert written["dispatch_node_observations"].exists()
    assert written["dispatch_arc_observations"].exists()

    dataset = load_graph_dispatch_dataset(
        written["dataset_manifest"],
        candidate_feature_names=("travel_to_pickup_time", "task_age"),
        node_feature_names=("x", "y", "robot_count", "is_ready_task_pickup"),
        edge_feature_names=("distance", "is_reserved_arc"),
    )
    manifest = json.loads(written["dataset_manifest"].read_text(encoding="utf-8"))

    assert dataset.row_count == manifest["dispatch_events"]
    assert dataset.examples[0].candidate_features.shape[1] == 2
    assert dataset.examples[0].node_features.shape[1] == 4
    assert dataset.examples[0].edge_features.shape[1] == 2


def test_fit_and_evaluate_graph_dispatch_model(tmp_path: Path) -> None:
    written = _write_dataset(tmp_path)
    dataset = load_graph_dispatch_dataset(
        written["dataset_manifest"],
        candidate_feature_names=("travel_to_pickup_time", "task_age"),
        node_feature_names=("x", "y", "robot_count", "is_ready_task_pickup"),
        edge_feature_names=("distance", "is_reserved_arc"),
    )
    splits = split_dispatch_observation_dataset(
        dataset,
        SplitConfig(train_fraction=0.6, validation_fraction=0.4, test_fraction=0.0, seed=3),
    )
    output_dir = tmp_path / "graph_fit"
    result = fit_graph_dispatch_model(
        train_dataset=splits.train.dataset,
        validation_dataset=splits.validation.dataset,
        config=GraphDispatchFitConfig(
            node_feature_names=("x", "y", "robot_count", "is_ready_task_pickup"),
            edge_feature_names=("distance", "is_reserved_arc"),
            candidate_feature_names=("travel_to_pickup_time", "task_age"),
            hidden_dim=16,
            max_epochs=3,
            patience=2,
            learning_rate=0.01,
        ),
        output_dir=output_dir,
    )

    artifact_path = output_dir / "model_artifact.json"
    loaded = load_graph_dispatch_model(artifact_path)
    evaluation = evaluate_graph_dispatch_artifact(splits.validation.dataset, artifact_path)

    assert artifact_path.exists()
    assert (output_dir / "graph_dispatch_model.pt").exists()
    assert loaded.artifact.model_type == "pyg_graph_dispatch"
    assert result.artifact.parameters["hidden_dim"] == 16
    assert evaluation.metrics["dispatch_groups"] == splits.validation.dataset.group_count


def test_run_experiment_with_trained_graph_dispatch_model(tmp_path: Path) -> None:
    artifact_path = _write_random_graph_artifact(tmp_path)
    config_path = tmp_path / "trained_graph_dispatch.toml"
    config_path.write_text(
        f"""
name = "trained_graph_dispatch"

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
policy = "trained_graph_dispatch_model"
horizon_seconds = 300.0
continue_until_all_tasks_complete = true

[policy_model]
artifact_path = "{artifact_path}"

[reporting]
output_dir = "outputs/trained_graph_dispatch"
write_plots = false
write_observation_dataset = true
""".strip(),
        encoding="utf-8",
    )

    result, written = run_experiment_from_path(
        config_path=config_path,
        output_dir_override=tmp_path / "outputs",
        force_write_observation_dataset=True,
    )

    assert result.policy_name == "trained_graph_dispatch_model"
    assert written["dispatch_node_observations"].exists()
    assert written["dispatch_arc_observations"].exists()


def _write_dataset(output_dir: Path) -> dict[str, Path]:
    environment = WarehouseEnvironment(
        build_synthetic_grid_layout(
            SyntheticGridLayoutConfig(
                rows=2,
                columns=3,
                special_node_types={(0, 0): NodeType.STORAGE, (1, 2): NodeType.DROPOFF},
            )
        )
    )
    result = run_simulation(
        environment=environment,
        tasks=(
            Task(task_id="task_1", release_time=0.0, pickup_node="r0_c0", dropoff_node="r1_c2", service_time_estimate=1.0),
            Task(task_id="task_2", release_time=0.0, pickup_node="r0_c1", dropoff_node="r1_c2", service_time_estimate=1.0),
        ),
        robots=(
            RobotSpec(robot_id="robot_1", initial_node="r1_c0"),
            RobotSpec(robot_id="robot_2", initial_node="r1_c1"),
        ),
        dispatch_policy=FIFODispatchPolicy(),
        config=SimulationConfig(),
    )
    return write_observation_dataset(output_dir, environment, result, experiment_name="graph_dataset_test")


def _write_random_graph_artifact(tmp_path: Path) -> Path:
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
    artifact = DispatchModelArtifact(
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
    return write_dispatch_model_artifact(artifact, tmp_path / "model_artifact.json")
