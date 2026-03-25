"""Tests for offline dispatch dataset loading and grouped splitting."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from warehouse_sim.learning import load_dispatch_observation_dataset, split_dispatch_observation_dataset
from warehouse_sim.learning.splits import SplitConfig


def test_load_dispatch_observation_dataset_from_manifest_tree(tmp_path: Path) -> None:
    _write_dataset_run(tmp_path / "run_a", run_id="run_a", scenario_name="scenario_a", seed=11, dispatch_start=0)
    _write_dataset_run(tmp_path / "run_b", run_id="run_b", scenario_name="scenario_b", seed=13, dispatch_start=10)

    dataset = load_dispatch_observation_dataset(tmp_path)

    assert dataset.row_count == 12
    assert dataset.group_count == 6
    assert dataset.feature_names[:2] == ("travel_to_pickup_time", "travel_to_pickup_distance")
    assert set(dataset.metadata["run_id"]) == {"run_a", "run_b"}
    assert set(dataset.metadata["scenario_name"]) == {"scenario_a", "scenario_b"}
    assert set(dataset.group_ids.tolist()) == {
        "run_a::dispatch_0",
        "run_a::dispatch_1",
        "run_a::dispatch_2",
        "run_b::dispatch_10",
        "run_b::dispatch_11",
        "run_b::dispatch_12",
    }


def test_grouped_splits_prevent_dispatch_and_run_leakage(tmp_path: Path) -> None:
    _write_dataset_run(tmp_path / "run_a", run_id="run_a", scenario_name="scenario_shared", seed=1, dispatch_start=0)
    _write_dataset_run(tmp_path / "run_b", run_id="run_b", scenario_name="scenario_shared", seed=2, dispatch_start=10)
    dataset = load_dispatch_observation_dataset(tmp_path)

    dispatch_splits = split_dispatch_observation_dataset(
        dataset,
        SplitConfig(train_fraction=0.5, validation_fraction=0.25, test_fraction=0.25, seed=5),
    )
    train_groups = set(dispatch_splits.train.dataset.group_ids.tolist())
    validation_groups = set(dispatch_splits.validation.dataset.group_ids.tolist())
    test_groups = set(dispatch_splits.test.dataset.group_ids.tolist())
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)

    run_splits = split_dispatch_observation_dataset(
        dataset,
        SplitConfig(
            train_fraction=0.5,
            validation_fraction=0.5,
            test_fraction=0.0,
            split_unit="run",
            seed=9,
        ),
    )
    assert len(set(run_splits.train.dataset.metadata["run_id"])) == 1
    assert len(set(run_splits.validation.dataset.metadata["run_id"])) == 1
    assert set(run_splits.train.dataset.metadata["run_id"]).isdisjoint(
        set(run_splits.validation.dataset.metadata["run_id"])
    )

    scenario_seed_splits = split_dispatch_observation_dataset(
        dataset,
        SplitConfig(
            train_fraction=0.5,
            validation_fraction=0.5,
            test_fraction=0.0,
            split_unit="scenario_seed",
            seed=2,
        ),
    )
    assert len(set(scenario_seed_splits.train.dataset.metadata["scenario_seed"])) == 1
    assert len(set(scenario_seed_splits.validation.dataset.metadata["scenario_seed"])) == 1
    assert set(scenario_seed_splits.train.dataset.metadata["scenario_seed"]).isdisjoint(
        set(scenario_seed_splits.validation.dataset.metadata["scenario_seed"])
    )


def test_manifest_tree_preferred_over_stale_root_dataset(tmp_path: Path) -> None:
    _write_dataset_run(tmp_path, run_id="stale_root", scenario_name="stale", seed=1, dispatch_start=0)
    _write_dataset_run(tmp_path / "nested_a", run_id="run_a", scenario_name="scenario_a", seed=11, dispatch_start=10)
    _write_dataset_run(tmp_path / "nested_b", run_id="run_b", scenario_name="scenario_b", seed=13, dispatch_start=20)

    dataset = load_dispatch_observation_dataset(tmp_path)

    assert dataset.row_count == 18
    assert set(dataset.metadata["run_id"]) == {"stale_root", "run_a", "run_b"}
    assert set(dataset.metadata["scenario_seed"]) == {"stale::seed_1", "scenario_a::seed_11", "scenario_b::seed_13"}


def _write_dataset_run(
    output_dir: Path,
    *,
    run_id: str,
    scenario_name: str,
    seed: int,
    dispatch_start: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dispatch_path = output_dir / "dispatch_observations.csv"
    manifest_path = output_dir / "dataset_manifest.json"
    rows = []
    for offset in range(3):
        dispatch_index = dispatch_start + offset
        rows.extend(
            [
                _dispatch_row(
                    dispatch_index=dispatch_index,
                    decision_time=float(dispatch_index),
                    candidate_robot_id=f"robot_{offset}_a",
                    candidate_task_id=f"task_{offset}_a",
                    is_selected=True,
                    travel_to_pickup_time=1.0 + offset,
                    task_age=3.0 + offset,
                ),
                _dispatch_row(
                    dispatch_index=dispatch_index,
                    decision_time=float(dispatch_index),
                    candidate_robot_id=f"robot_{offset}_b",
                    candidate_task_id=f"task_{offset}_b",
                    is_selected=False,
                    travel_to_pickup_time=4.0 + offset,
                    task_age=0.5,
                ),
            ]
        )

    with dispatch_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "scenario_name": scenario_name,
                "experiment_name": scenario_name,
                "policy_name": "fifo",
                "demand_seed": seed,
                "files": {
                    "dispatch_observations": dispatch_path.name,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _dispatch_row(
    *,
    dispatch_index: int,
    decision_time: float,
    candidate_robot_id: str,
    candidate_task_id: str,
    is_selected: bool,
    travel_to_pickup_time: float,
    task_age: float,
) -> dict[str, object]:
    return {
        "dispatch_index": dispatch_index,
        "decision_time": decision_time,
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
