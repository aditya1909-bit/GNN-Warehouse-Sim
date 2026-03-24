"""Tests for baseline dispatch policies."""

from __future__ import annotations

from warehouse_sim.agents import RobotSpec, RobotState
from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.graph import SyntheticGridLayoutConfig, build_synthetic_grid_layout
from warehouse_sim.policies import (
    FIFODispatchPolicy,
    NearestRobotTaskPolicy,
    RandomDispatchPolicy,
)
from warehouse_sim.tasks import Task


def _environment() -> WarehouseEnvironment:
    return WarehouseEnvironment(build_synthetic_grid_layout(SyntheticGridLayoutConfig(rows=1, columns=4)))


def test_fifo_policy_selects_first_ready_task_and_first_robot() -> None:
    environment = _environment()
    robots = (
        RobotState.from_spec(RobotSpec(robot_id="robot_b", initial_node="r0_c1")),
        RobotState.from_spec(RobotSpec(robot_id="robot_a", initial_node="r0_c0")),
    )
    tasks = (
        Task(task_id="task_2", release_time=2.0, pickup_node="r0_c2", dropoff_node="r0_c3"),
        Task(task_id="task_1", release_time=1.0, pickup_node="r0_c1", dropoff_node="r0_c2"),
    )

    decision = FIFODispatchPolicy().select_assignment(robots, tasks, environment)

    assert decision.robot_id == "robot_a"
    assert decision.task_id == "task_1"


def test_nearest_robot_task_policy_prefers_closest_pair() -> None:
    environment = _environment()
    robots = (
        RobotState.from_spec(RobotSpec(robot_id="robot_1", initial_node="r0_c0")),
        RobotState.from_spec(RobotSpec(robot_id="robot_2", initial_node="r0_c3")),
    )
    tasks = (
        Task(task_id="task_left", release_time=0.0, pickup_node="r0_c0", dropoff_node="r0_c1"),
        Task(task_id="task_right", release_time=0.0, pickup_node="r0_c3", dropoff_node="r0_c2"),
    )

    decision = NearestRobotTaskPolicy().select_assignment(robots, tasks, environment)

    assert decision == decision.__class__(robot_id="robot_1", task_id="task_left")


def test_random_policy_is_seeded() -> None:
    environment = _environment()
    robots = (
        RobotState.from_spec(RobotSpec(robot_id="robot_1", initial_node="r0_c0")),
        RobotState.from_spec(RobotSpec(robot_id="robot_2", initial_node="r0_c1")),
    )
    tasks = (
        Task(task_id="task_1", release_time=0.0, pickup_node="r0_c2", dropoff_node="r0_c3"),
        Task(task_id="task_2", release_time=0.0, pickup_node="r0_c1", dropoff_node="r0_c0"),
    )

    first = RandomDispatchPolicy(seed=11).select_assignment(robots, tasks, environment)
    second = RandomDispatchPolicy(seed=11).select_assignment(robots, tasks, environment)

    assert first == second
