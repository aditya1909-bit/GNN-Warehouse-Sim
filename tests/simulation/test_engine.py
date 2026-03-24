"""Tests for the baseline discrete-event simulation engine."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from warehouse_sim.agents import RobotSpec
from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.graph import NodeType, SyntheticGridLayoutConfig, build_synthetic_grid_layout
from warehouse_sim.policies import FIFODispatchPolicy, NearestRobotTaskPolicy
from warehouse_sim.simulation import SimulationConfig, run_simulation
from warehouse_sim.tasks import Task


def _simulation_environment() -> WarehouseEnvironment:
    graph = build_synthetic_grid_layout(
        SyntheticGridLayoutConfig(
            rows=2,
            columns=2,
            special_node_types={(0, 0): NodeType.STORAGE, (1, 1): NodeType.DROPOFF, (1, 0): NodeType.STAGING},
            zone_labels={(0, 0): "storage_zone", (1, 1): "dropoff_zone", (1, 0): "staging_zone"},
        )
    )
    return WarehouseEnvironment(graph=graph)


def test_fifo_simulation_smoke_and_metrics() -> None:
    environment = _simulation_environment()
    tasks = (
        Task(task_id="task_1", release_time=0.0, pickup_node="r0_c0", dropoff_node="r1_c1", service_time_estimate=5.0),
        Task(task_id="task_2", release_time=3.0, pickup_node="r0_c0", dropoff_node="r1_c1", service_time_estimate=5.0),
    )
    robots = (RobotSpec(robot_id="robot_1", initial_node="r1_c0"),)

    result = run_simulation(
        environment=environment,
        tasks=tasks,
        robots=robots,
        dispatch_policy=FIFODispatchPolicy(),
        config=SimulationConfig(),
    )

    assert result.metrics.tasks_generated == 2
    assert result.metrics.tasks_completed == 2
    assert result.metrics.tasks_unassigned == 0
    assert result.executions[0].waiting_time == pytest.approx(0.0)
    assert result.executions[1].waiting_time > 0.0
    assert result.metrics.average_turnaround_time is not None
    assert result.metrics.average_queue_length >= 0.0
    assert result.metrics.robot_metrics[0].tasks_completed == 2


def test_nearest_robot_policy_assigns_closer_robot() -> None:
    environment = _simulation_environment()
    tasks = (
        Task(task_id="task_1", release_time=0.0, pickup_node="r0_c0", dropoff_node="r1_c1", service_time_estimate=1.0),
    )
    robots = (
        RobotSpec(robot_id="robot_far", initial_node="r1_c1"),
        RobotSpec(robot_id="robot_close", initial_node="r1_c0"),
    )

    result = run_simulation(
        environment=environment,
        tasks=tasks,
        robots=robots,
        dispatch_policy=NearestRobotTaskPolicy(),
        config=SimulationConfig(),
    )

    assert result.executions[0].robot_id == "robot_close"


def test_simulation_cli_smoke() -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_simulation_baseline.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--policy",
            "fifo",
            "--horizon-seconds",
            "600",
            "--mean-interval",
            "120",
            "--min-tasks",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Tasks completed:" in completed.stdout
