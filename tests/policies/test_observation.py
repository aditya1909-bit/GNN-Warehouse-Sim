"""Tests for dispatch-context and policy-observation hooks."""

from __future__ import annotations

from warehouse_sim.agents import RobotSpec, RobotState
from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.graph import SyntheticGridLayoutConfig, build_synthetic_grid_layout
from warehouse_sim.policies import DispatchContextBuilder, DispatchDecision, DispatchPolicy
from warehouse_sim.simulation import SimulationConfig, run_simulation
from warehouse_sim.tasks import Task


def _environment() -> WarehouseEnvironment:
    return WarehouseEnvironment(build_synthetic_grid_layout(SyntheticGridLayoutConfig(rows=1, columns=4)))


def test_dispatch_context_builder_splits_ready_and_future_tasks() -> None:
    environment = _environment()
    builder = DispatchContextBuilder(environment)
    robots = (
        RobotState.from_spec(RobotSpec(robot_id="robot_1", initial_node="r0_c0")),
        RobotState.from_spec(RobotSpec(robot_id="robot_2", initial_node="r0_c1", available_from=5.0)),
    )
    pending_tasks = (
        Task(task_id="task_ready", release_time=2.0, pickup_node="r0_c1", dropoff_node="r0_c2"),
        Task(task_id="task_future", release_time=6.0, pickup_node="r0_c2", dropoff_node="r0_c3"),
    )

    context = builder.build(current_time=4.0, robot_states=robots, pending_tasks=pending_tasks)

    assert [task.task_id for task in context.ready_tasks] == ["task_ready"]
    assert [task.task_id for task in context.future_tasks] == ["task_future"]
    assert context.global_observation.ready_task_count == 1
    assert context.global_observation.future_task_count == 1
    assert context.global_observation.idle_robot_count == 1
    assert context.global_observation.busy_robot_count == 1
    assert context.robot_observations[0].is_idle is True
    assert context.robot_observations[1].time_until_available == 1.0
    assert context.task_observations[0].age == 2.0
    assert context.task_observations[1].time_until_release == 2.0
    assert len(context.graph_features.nodes) == 4


class ContextOnlyPolicy(DispatchPolicy):
    """Policy used to verify engine integration with dispatch contexts."""

    name = "context_only"

    def __init__(self) -> None:
        self.context_calls = 0

    def select_assignment_from_context(self, context):  # type: ignore[override]
        self.context_calls += 1
        assert context.global_observation.ready_task_count == 1
        ready_task = context.task_observations[0]
        idle_robot = context.robot_observations[0]
        return DispatchDecision(robot_id=idle_robot.robot_id, task_id=ready_task.task_id)

    def select_assignment(self, idle_robots, ready_tasks, environment):  # type: ignore[override]
        raise AssertionError("Engine should use select_assignment_from_context.")


def test_run_simulation_uses_dispatch_context_hook() -> None:
    environment = _environment()
    tasks = (
        Task(task_id="task_1", release_time=0.0, pickup_node="r0_c1", dropoff_node="r0_c3"),
    )
    robots = (
        RobotSpec(robot_id="robot_1", initial_node="r0_c0"),
    )

    policy = ContextOnlyPolicy()

    result = run_simulation(
        environment=environment,
        tasks=tasks,
        robots=robots,
        dispatch_policy=policy,
        config=SimulationConfig(),
    )

    assert len(result.executions) == 1
    assert result.executions[0].task_id == "task_1"
    assert policy.context_calls == 1
