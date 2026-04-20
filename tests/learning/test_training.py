"""Tests for offline linear and MLP dispatch model fitting."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from warehouse_sim.learning import (
    GroupedLinearFitConfig,
    GroupedMLPFitConfig,
    load_dispatch_model_artifact,
    load_dispatch_observation_dataset,
    split_dispatch_observation_dataset,
    write_dispatch_model_artifact,
)
from warehouse_sim.learning.linear_fit import fit_grouped_linear_model
from warehouse_sim.learning.mlp_fit import fit_grouped_mlp_model
from warehouse_sim.learning.splits import SplitConfig


def test_fit_linear_model_and_serialize_artifact(tmp_path: Path) -> None:
    dataset_path = _write_training_dataset(tmp_path / "dispatch_observations.csv")
    dataset = load_dispatch_observation_dataset(dataset_path, feature_names=("travel_to_pickup_time", "task_age"))
    splits = split_dispatch_observation_dataset(
        dataset,
        SplitConfig(train_fraction=0.6, validation_fraction=0.2, test_fraction=0.2, seed=7),
    )

    result = fit_grouped_linear_model(
        splits.train,
        splits.validation,
        GroupedLinearFitConfig(learning_rate=0.1, max_epochs=250, patience=40),
    )
    artifact_path = write_dispatch_model_artifact(result.artifact, tmp_path / "linear_model.json")
    loaded = load_dispatch_model_artifact(artifact_path)

    assert artifact_path.exists()
    assert loaded.model_type == "grouped_linear"
    assert loaded.feature_names == ("travel_to_pickup_time", "task_age")
    assert loaded.parameters["weights"][0] < 0.0
    assert loaded.parameters["weights"][1] > 0.0


def test_fit_grouped_mlp_model_smoke(tmp_path: Path) -> None:
    dataset_path = _write_training_dataset(tmp_path / "dispatch_observations.csv")
    dataset = load_dispatch_observation_dataset(dataset_path, feature_names=("travel_to_pickup_time", "task_age"))
    splits = split_dispatch_observation_dataset(
        dataset,
        SplitConfig(train_fraction=0.6, validation_fraction=0.2, test_fraction=0.2, seed=3),
    )

    result = fit_grouped_mlp_model(
        splits.train,
        splits.validation,
        GroupedMLPFitConfig(hidden_dim=8, learning_rate=0.05, max_epochs=120, patience=20, seed=4),
    )

    assert result.artifact.model_type == "grouped_mlp"
    assert len(result.training_history) >= 1
    assert len(result.artifact.parameters["hidden_bias"]) == 8


def test_weighted_linear_training_records_benchmark_weighting(tmp_path: Path) -> None:
    dataset_path = _write_training_dataset(tmp_path / "dispatch_observations.csv")
    dataset = load_dispatch_observation_dataset(dataset_path, feature_names=("travel_to_pickup_time", "task_age"))
    splits = split_dispatch_observation_dataset(
        dataset,
        SplitConfig(train_fraction=0.6, validation_fraction=0.2, test_fraction=0.2, seed=7),
    )

    result = fit_grouped_linear_model(
        splits.train,
        splits.validation,
        GroupedLinearFitConfig(
            learning_rate=0.1,
            max_epochs=100,
            patience=10,
            benchmark_weighting=True,
        ),
    )

    assert result.artifact.metadata["training"]["benchmark_weighting"] is True


def test_load_dispatch_model_artifact_rejects_legacy_version(tmp_path: Path) -> None:
    artifact_path = tmp_path / "legacy.json"
    legacy_payload = {
        "artifact_version": 1,
        "model_type": "grouped_linear",
        "objective": "dispatch_group_softmax_cross_entropy",
        "feature_names": ["travel_to_pickup_time", "task_age"],
        "parameters": {"weights": [-1.0, 1.0], "bias": 0.0},
        "metadata": {},
    }
    artifact_path.write_text(json.dumps(legacy_payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="battery-aware dispatch requires retraining"):
        load_dispatch_model_artifact(artifact_path)


def _write_training_dataset(path: Path) -> Path:
    rows = []
    for dispatch_index in range(10):
        selected_is_first = dispatch_index % 2 == 0
        rows.extend(
            [
                _row(
                    dispatch_index=dispatch_index,
                    candidate_robot_id=f"robot_{dispatch_index}_a",
                    candidate_task_id=f"task_{dispatch_index}_a",
                    travel_to_pickup_time=1.0 if selected_is_first else 4.0,
                    task_age=4.0 if selected_is_first else 0.5,
                    is_selected=selected_is_first,
                ),
                _row(
                    dispatch_index=dispatch_index,
                    candidate_robot_id=f"robot_{dispatch_index}_b",
                    candidate_task_id=f"task_{dispatch_index}_b",
                    travel_to_pickup_time=4.0 if selected_is_first else 1.0,
                    task_age=0.5 if selected_is_first else 4.0,
                    is_selected=not selected_is_first,
                ),
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(
    *,
    dispatch_index: int,
    candidate_robot_id: str,
    candidate_task_id: str,
    travel_to_pickup_time: float,
    task_age: float,
    is_selected: bool,
) -> dict[str, object]:
    return {
        "dispatch_index": dispatch_index,
        "decision_time": float(dispatch_index),
        "selected_robot_id": "selected_robot",
        "selected_action_type": "task",
        "selected_task_id": "selected_task",
        "selected_charging_node_id": "",
        "candidate_robot_id": candidate_robot_id,
        "candidate_action_type": "task",
        "candidate_task_id": candidate_task_id,
        "candidate_charging_node_id": "",
        "is_selected": is_selected,
        "robot_current_node": "r0_c0",
        "robot_current_zone": "staging_zone",
        "robot_speed_multiplier": 1.0,
        "robot_completed_task_count": 0,
        "robot_total_busy_time": 0.0,
        "robot_total_idle_time": 0.0,
        "robot_total_travel_time": 0.0,
        "robot_total_travel_distance": 0.0,
        "task_release_time": 0.0,
        "task_age": task_age,
        "task_priority": 1,
        "task_service_time_estimate": 5.0,
        "task_pickup_node": "r0_c0",
        "task_dropoff_node": "r0_c1",
        "task_source_zone": "storage_zone",
        "task_destination_zone": "dropoff_zone",
        "travel_to_pickup_time": travel_to_pickup_time,
        "travel_to_pickup_distance": travel_to_pickup_time,
        "pickup_to_dropoff_time": 1.0,
        "pickup_to_dropoff_distance": 1.0,
        "pending_task_count": 2,
        "ready_task_count": 2,
        "future_task_count": 0,
        "idle_robot_count": 1,
        "busy_robot_count": 0,
        "mean_ready_task_age": 1.0,
        "average_robot_time_until_available": 0.0,
        "execution_model": "idealized",
        "active_reserved_edge_count": 0,
        "active_reserved_node_count": 0,
        "estimated_pickup_congestion_delay": 0.0,
        "estimated_dropoff_congestion_delay": 0.0,
        "estimated_pickup_blocked_segments": 0,
        "estimated_dropoff_blocked_segments": 0,
    }
