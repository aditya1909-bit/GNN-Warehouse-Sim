"""Tests for machine-readable metric report writers."""

from __future__ import annotations

import json
from pathlib import Path

from warehouse_sim.agents import RobotSpec
from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.graph import NodeType, SyntheticGridLayoutConfig, build_synthetic_grid_layout
from warehouse_sim.metrics import (
    prepare_queue_length_series,
    prepare_robot_utilization_series,
    write_observation_dataset,
    write_simulation_report,
)
from warehouse_sim.policies import FIFODispatchPolicy
from warehouse_sim.simulation import SimulationConfig, run_simulation
from warehouse_sim.tasks import Task


def _result():
    environment = WarehouseEnvironment(
        build_synthetic_grid_layout(
            SyntheticGridLayoutConfig(
                rows=2,
                columns=2,
                special_node_types={(0, 0): NodeType.STORAGE, (1, 1): NodeType.DROPOFF},
            )
        )
    )
    return run_simulation(
        environment=environment,
        tasks=(
            Task(task_id="task_1", release_time=0.0, pickup_node="r0_c0", dropoff_node="r1_c1", service_time_estimate=1.0),
        ),
        robots=(RobotSpec(robot_id="robot_1", initial_node="r0_c1"),),
        dispatch_policy=FIFODispatchPolicy(),
        config=SimulationConfig(),
    )


def test_write_simulation_report(tmp_path: Path) -> None:
    result = _result()
    written = write_simulation_report(tmp_path, result, experiment_name="report_test")

    assert written["summary"].exists()
    assert written["executions"].exists()
    assert written["queue_snapshots"].exists()
    assert written["robot_metrics"].exists()

    payload = json.loads(written["summary"].read_text(encoding="utf-8"))
    assert payload["experiment_name"] == "report_test"
    assert payload["metrics"]["tasks_completed"] == 1


def test_plot_data_preparation() -> None:
    result = _result()
    queue_series = prepare_queue_length_series(result)
    robot_series = prepare_robot_utilization_series(result)

    assert queue_series["time"]
    assert "ready_tasks" in queue_series
    assert robot_series["robot_id"] == ["robot_1"]


def test_write_default_plots(tmp_path: Path) -> None:
    pytest = __import__("pytest")
    pytest.importorskip("matplotlib")

    from warehouse_sim.metrics import write_default_plots

    result = _result()
    queue_plot, robot_plot = write_default_plots(tmp_path, result)

    assert queue_plot.exists()
    assert robot_plot.exists()


def test_write_observation_dataset(tmp_path: Path) -> None:
    environment = WarehouseEnvironment(
        build_synthetic_grid_layout(
            SyntheticGridLayoutConfig(
                rows=2,
                columns=2,
                special_node_types={(0, 0): NodeType.STORAGE, (1, 1): NodeType.DROPOFF},
            )
        )
    )
    result = run_simulation(
        environment=environment,
        tasks=(
            Task(task_id="task_1", release_time=0.0, pickup_node="r0_c0", dropoff_node="r1_c1", service_time_estimate=1.0),
        ),
        robots=(RobotSpec(robot_id="robot_1", initial_node="r0_c1"),),
        dispatch_policy=FIFODispatchPolicy(),
        config=SimulationConfig(),
    )

    written = write_observation_dataset(
        output_dir=tmp_path,
        environment=environment,
        result=result,
        experiment_name="dataset_test",
    )

    assert written["dataset_manifest"].exists()
    assert written["graph_nodes"].exists()
    assert written["graph_arcs"].exists()
    assert written["dispatch_observations"].exists()

    payload = json.loads(written["dataset_manifest"].read_text(encoding="utf-8"))
    assert payload["experiment_name"] == "dataset_test"
    assert payload["candidate_rows"] == len(result.dispatch_traces)
