"""Tests for route-aware and congestion-aware execution modes."""

from __future__ import annotations

from warehouse_sim.agents import RobotSpec
from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.graph import SyntheticGridLayoutConfig, build_synthetic_grid_layout
from warehouse_sim.policies import FIFODispatchPolicy
from warehouse_sim.simulation import ExecutionModel, SimulationConfig, run_simulation
from warehouse_sim.tasks import Task


def _line_environment(length: int = 4) -> WarehouseEnvironment:
    return WarehouseEnvironment(build_synthetic_grid_layout(SyntheticGridLayoutConfig(rows=1, columns=length)))


def _blocking_tasks() -> tuple[Task, ...]:
    return (
        Task(
            task_id="task_1",
            release_time=0.0,
            pickup_node="r0_c1",
            dropoff_node="r0_c3",
            service_time_estimate=0.0,
        ),
        Task(
            task_id="task_2",
            release_time=0.0,
            pickup_node="r0_c1",
            dropoff_node="r0_c3",
            service_time_estimate=0.0,
        ),
    )


def _blocking_robots() -> tuple[RobotSpec, ...]:
    return (
        RobotSpec(robot_id="robot_1", initial_node="r0_c0"),
        RobotSpec(robot_id="robot_2", initial_node="r0_c0"),
    )


def test_idealized_mode_has_zero_congestion_delay_even_on_shared_routes() -> None:
    result = run_simulation(
        environment=_line_environment(),
        tasks=_blocking_tasks(),
        robots=_blocking_robots(),
        dispatch_policy=FIFODispatchPolicy(),
        config=SimulationConfig(execution_model=ExecutionModel.IDEALIZED),
    )

    assert result.metrics.congestion_delay_total == 0.0
    assert result.metrics.blocked_traversal_events_total == 0
    assert all(execution.congestion_delay_time == 0.0 for execution in result.executions)
    assert all(execution.travel_to_pickup_wait_time == 0.0 for execution in result.executions)


def test_reserved_edges_mode_adds_positive_congestion_delay_for_blocked_route() -> None:
    result = run_simulation(
        environment=_line_environment(),
        tasks=_blocking_tasks(),
        robots=_blocking_robots(),
        dispatch_policy=FIFODispatchPolicy(),
        config=SimulationConfig(execution_model=ExecutionModel.RESERVED_EDGES),
    )

    assert result.metrics.congestion_delay_total > 0.0
    assert result.metrics.blocked_traversal_events_total > 0
    assert result.executions[1].congestion_delay_time > 0.0
    assert result.executions[1].travel_to_pickup_path_nodes == ("r0_c0", "r0_c1")
    assert result.executions[1].travel_to_dropoff_path_arcs == ("r0_c1->r0_c2", "r0_c2->r0_c3")


def test_reserved_edges_contention_resolution_is_deterministic() -> None:
    first = run_simulation(
        environment=_line_environment(),
        tasks=_blocking_tasks(),
        robots=_blocking_robots(),
        dispatch_policy=FIFODispatchPolicy(),
        config=SimulationConfig(execution_model=ExecutionModel.RESERVED_EDGES),
    )
    second = run_simulation(
        environment=_line_environment(),
        tasks=_blocking_tasks(),
        robots=_blocking_robots(),
        dispatch_policy=FIFODispatchPolicy(),
        config=SimulationConfig(execution_model=ExecutionModel.RESERVED_EDGES),
    )

    first_signature = tuple(
        (
            execution.robot_id,
            execution.task_id,
            execution.travel_to_pickup_time,
            execution.travel_to_dropoff_time,
            execution.congestion_delay_time,
            execution.blocked_traversal_events,
        )
        for execution in first.executions
    )
    second_signature = tuple(
        (
            execution.robot_id,
            execution.task_id,
            execution.travel_to_pickup_time,
            execution.travel_to_dropoff_time,
            execution.congestion_delay_time,
            execution.blocked_traversal_events,
        )
        for execution in second.executions
    )

    assert first_signature == second_signature


def test_reserved_nodes_mode_blocks_station_arrivals() -> None:
    environment = _line_environment(length=3)
    tasks = (
        Task(task_id="task_1", release_time=0.0, pickup_node="r0_c1", dropoff_node="r0_c2", service_time_estimate=1.0),
        Task(task_id="task_2", release_time=0.0, pickup_node="r0_c1", dropoff_node="r0_c2", service_time_estimate=1.0),
    )
    robots = (
        RobotSpec(robot_id="robot_1", initial_node="r0_c0"),
        RobotSpec(robot_id="robot_2", initial_node="r0_c0"),
    )

    result = run_simulation(
        environment=environment,
        tasks=tasks,
        robots=robots,
        dispatch_policy=FIFODispatchPolicy(),
        config=SimulationConfig(execution_model=ExecutionModel.RESERVED_NODES),
    )

    assert result.metrics.congestion_delay_total > 0.0
    assert result.executions[1].blocked_traversal_events > 0
