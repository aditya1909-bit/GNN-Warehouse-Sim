"""Tests for offline evaluation metrics and reporting."""

from __future__ import annotations

import csv
from pathlib import Path

from warehouse_sim.learning import load_dispatch_observation_dataset
from warehouse_sim.learning.artifacts import DispatchModelArtifact
from warehouse_sim.learning.evaluation import evaluate_dispatch_model


def test_evaluate_dispatch_model_computes_group_ranking_metrics(tmp_path: Path) -> None:
    path = tmp_path / "dispatch_observations.csv"
    rows = [
        _row(0, "robot_1", "task_1", True, 1.0, 4.0),
        _row(0, "robot_2", "task_2", False, 4.0, 0.5),
        _row(1, "robot_3", "task_3", False, 4.0, 0.5),
        _row(1, "robot_4", "task_4", True, 1.0, 5.0),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    dataset = load_dispatch_observation_dataset(path, feature_names=("travel_to_pickup_time", "task_age"))
    artifact = DispatchModelArtifact(
        artifact_version=1,
        model_type="grouped_linear",
        objective="dispatch_group_softmax_cross_entropy",
        feature_names=("travel_to_pickup_time", "task_age"),
        parameters={"weights": [-2.0, 1.0], "bias": 0.0},
    )

    result = evaluate_dispatch_model(dataset, artifact)

    assert result.metrics["dispatch_groups"] == 2
    assert result.metrics["group_top_1_accuracy"] == 1.0
    assert result.metrics["mean_reciprocal_rank"] == 1.0
    assert len(result.prediction_rows) == 4


def _row(
    dispatch_index: int,
    candidate_robot_id: str,
    candidate_task_id: str,
    is_selected: bool,
    travel_to_pickup_time: float,
    task_age: float,
) -> dict[str, object]:
    return {
        "dispatch_index": dispatch_index,
        "decision_time": float(dispatch_index),
        "selected_robot_id": "selected_robot",
        "selected_task_id": "selected_task",
        "candidate_robot_id": candidate_robot_id,
        "candidate_task_id": candidate_task_id,
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
