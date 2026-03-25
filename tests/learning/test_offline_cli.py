"""Tests for config-driven offline training orchestration."""

from __future__ import annotations

import csv
from pathlib import Path

from warehouse_sim.agents import RobotSpec
from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.graph import NodeType, SyntheticGridLayoutConfig, build_synthetic_grid_layout
from warehouse_sim.learning.cli import run_offline_training_from_path
from warehouse_sim.metrics import write_observation_dataset
from warehouse_sim.policies import FIFODispatchPolicy
from warehouse_sim.simulation import SimulationConfig, run_simulation
from warehouse_sim.tasks import Task


def test_run_offline_training_from_config_path(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dispatch_observations.csv"
    rows = [
        _row(0, "robot_1", "task_1", True, 1.0, 4.0),
        _row(0, "robot_2", "task_2", False, 4.0, 0.5),
        _row(1, "robot_3", "task_3", False, 4.0, 0.5),
        _row(1, "robot_4", "task_4", True, 1.0, 4.0),
        _row(2, "robot_5", "task_5", True, 1.0, 5.0),
        _row(2, "robot_6", "task_6", False, 4.0, 0.5),
    ]
    with dataset_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    config_path = tmp_path / "offline_training.toml"
    config_path.write_text(
        """
name = "offline_linear_fit_test"
seed = 7

[dataset]
source = "dispatch_observations.csv"
feature_names = ["travel_to_pickup_time", "task_age"]

[split]
split_unit = "dispatch_group"
train_fraction = 0.67
validation_fraction = 0.33
test_fraction = 0.0

[model]
type = "linear"
learning_rate = 0.1
max_epochs = 100
patience = 20

[reporting]
output_dir = "offline_outputs"
""".strip(),
        encoding="utf-8",
    )

    written = run_offline_training_from_path(config_path)

    assert written["artifact"].exists()
    assert written["training_summary"].exists()
    assert written["train_evaluation"].exists()
    assert written["validation_predictions"].exists()


def test_run_graph_offline_training_from_config_path(tmp_path: Path) -> None:
    written_dataset = _write_graph_dataset(tmp_path / "dataset")
    config_path = tmp_path / "offline_graph_training.toml"
    config_path.write_text(
        """
name = "offline_graph_dispatch_fit_test"
seed = 7

[dataset]
source = "dataset/dataset_manifest.json"
feature_names = ["travel_to_pickup_time", "task_age"]
node_feature_names = ["x", "y", "robot_count", "is_ready_task_pickup"]
edge_feature_names = ["distance", "is_reserved_arc"]

[split]
split_unit = "dispatch_group"
train_fraction = 0.6
validation_fraction = 0.4
test_fraction = 0.0

[model]
type = "graph_dispatch"
hidden_dim = 16
message_passing_layers = 2
dropout = 0.0
batch_size = 4
learning_rate = 0.01
max_epochs = 3
patience = 2

[reporting]
output_dir = "offline_graph_outputs"
""".strip(),
        encoding="utf-8",
    )

    written = run_offline_training_from_path(config_path)

    assert written_dataset["dispatch_node_observations"].exists()
    assert written_dataset["dispatch_arc_observations"].exists()
    assert written["artifact"].exists()
    assert written["state_dict"].exists()
    assert written["training_summary"].exists()
    assert written["validation_predictions"].exists()


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


def _write_graph_dataset(output_dir: Path) -> dict[str, Path]:
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
            Task(
                task_id="task_1",
                release_time=0.0,
                pickup_node="r0_c0",
                dropoff_node="r1_c2",
                service_time_estimate=1.0,
            ),
            Task(
                task_id="task_2",
                release_time=0.0,
                pickup_node="r0_c1",
                dropoff_node="r1_c2",
                service_time_estimate=1.0,
            ),
        ),
        robots=(
            RobotSpec(robot_id="robot_1", initial_node="r1_c0"),
            RobotSpec(robot_id="robot_2", initial_node="r1_c1"),
        ),
        dispatch_policy=FIFODispatchPolicy(),
        config=SimulationConfig(),
    )
    return write_observation_dataset(output_dir, environment, result, experiment_name="graph_cli_dataset_test")
