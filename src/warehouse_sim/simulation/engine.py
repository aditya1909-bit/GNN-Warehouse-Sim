"""Discrete-event simulation engine for the first warehouse baseline."""

from __future__ import annotations

from warehouse_sim.agents import RobotSpec, RobotState
from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.learning.graph_data import (
    build_dispatch_arc_observation_records,
    build_dispatch_node_observation_records,
)
from warehouse_sim.metrics.collector import compute_simulation_metrics
from warehouse_sim.policies import (
    DispatchContext,
    DispatchContextBuilder,
    DispatchPolicy,
    build_candidate_assignment_observations,
)
from warehouse_sim.simulation.execution import ResourceReservationTable
from warehouse_sim.simulation.models import (
    DispatchTraceRecord,
    ExecutionModel,
    QueueSnapshot,
    SimulationConfig,
    SimulationResult,
    TaskExecution,
)
from warehouse_sim.tasks import Task, TaskQueue


def run_simulation(
    environment: WarehouseEnvironment,
    tasks: tuple[Task, ...],
    robots: tuple[RobotSpec, ...],
    dispatch_policy: DispatchPolicy,
    config: SimulationConfig | None = None,
) -> SimulationResult:
    """Run a minimal discrete-event warehouse simulation."""

    config = config or SimulationConfig()
    queue = TaskQueue(tasks)
    robot_states = tuple(RobotState.from_spec(robot) for robot in robots)
    context_builder = DispatchContextBuilder(environment)
    reservation_table = ResourceReservationTable(config.execution_model)
    for robot in robot_states:
        environment.graph.node(robot.current_node)
    for task in tasks:
        environment.graph.node(task.pickup_node)
        environment.graph.node(task.dropoff_node)

    current_time = 0.0
    executions: list[TaskExecution] = []
    dispatch_traces: list[DispatchTraceRecord] = []
    dispatch_node_observations = []
    dispatch_arc_observations = []
    snapshots: list[QueueSnapshot] = []
    _record_snapshot(current_time, queue, robot_states, executions, snapshots)

    while True:
        context = _build_dispatch_context(
            context_builder=context_builder,
            queue=queue,
            current_time=current_time,
            robot_states=robot_states,
            config=config,
            reservation_table=reservation_table,
        )
        ready_tasks = context.ready_tasks
        idle_robots = context.idle_robots

        while ready_tasks and idle_robots:
            decision = dispatch_policy.select_assignment_from_context(context)
            if decision is None:
                break

            dispatch_traces.extend(_build_dispatch_trace_records(context, decision, len(executions)))
            dispatch_node_observations.extend(
                build_dispatch_node_observation_records(
                    context=context,
                    dispatch_index=len(executions),
                    decision=decision,
                )
            )
            dispatch_arc_observations.extend(
                build_dispatch_arc_observation_records(
                    context=context,
                    dispatch_index=len(executions),
                )
            )
            robot = _robot_by_id(robot_states, decision.robot_id)
            task = _task_by_id(ready_tasks, decision.task_id)
            queue.remove_task(task.task_id)
            executions.append(
                _assign_task(
                    current_time=current_time,
                    environment=environment,
                    robot=robot,
                    task=task,
                    execution_model=config.execution_model,
                    reservation_table=reservation_table,
                )
            )
            context = _build_dispatch_context(
                context_builder=context_builder,
                queue=queue,
                current_time=current_time,
                robot_states=robot_states,
                config=config,
                reservation_table=reservation_table,
            )
            ready_tasks = context.ready_tasks
            idle_robots = context.idle_robots

        _record_snapshot(current_time, queue, robot_states, executions, snapshots)
        next_time = _next_event_time(current_time=current_time, queue=queue, robot_states=robot_states, config=config)
        if next_time is None:
            break
        current_time = next_time

    finished_at = max(
        [current_time, *(robot.available_time for robot in robot_states)],
        default=current_time,
    )
    _record_snapshot(finished_at, queue, robot_states, executions, snapshots)
    unassigned_tasks = tuple(queue.pending_tasks())
    result = SimulationResult(
        policy_name=dispatch_policy.name,
        started_at=0.0,
        finished_at=finished_at,
        tasks_generated=len(tasks),
        robot_states=robot_states,
        executions=tuple(executions),
        dispatch_traces=tuple(dispatch_traces),
        dispatch_node_observations=tuple(dispatch_node_observations),
        dispatch_arc_observations=tuple(dispatch_arc_observations),
        unassigned_tasks=unassigned_tasks,
        queue_snapshots=tuple(snapshots),
        metrics=None,  # type: ignore[arg-type]
    )
    metrics = compute_simulation_metrics(result)
    return SimulationResult(
        policy_name=result.policy_name,
        started_at=result.started_at,
        finished_at=result.finished_at,
        tasks_generated=result.tasks_generated,
        robot_states=result.robot_states,
        executions=result.executions,
        dispatch_traces=result.dispatch_traces,
        dispatch_node_observations=result.dispatch_node_observations,
        dispatch_arc_observations=result.dispatch_arc_observations,
        unassigned_tasks=result.unassigned_tasks,
        queue_snapshots=result.queue_snapshots,
        metrics=metrics,
    )


def _assign_task(
    current_time: float,
    environment: WarehouseEnvironment,
    robot: RobotState,
    task: Task,
    execution_model: ExecutionModel,
    reservation_table: ResourceReservationTable,
) -> TaskExecution:
    if current_time > robot.available_time:
        robot.total_idle_time += current_time - robot.available_time

    assignment = reservation_table.execute_assignment(
        environment=environment,
        execution_model=execution_model,
        current_time=current_time,
        start_node=robot.current_node,
        pickup_node=task.pickup_node,
        dropoff_node=task.dropoff_node,
        service_time=task.service_time_estimate,
        speed_multiplier=robot.spec.speed_multiplier,
    )
    travel_to_pickup = assignment.travel_to_pickup
    travel_to_dropoff = assignment.travel_to_dropoff
    pickup_arrival_time = assignment.pickup_arrival_time
    service_start_time = assignment.service_start_time
    completion_time = assignment.completion_time

    robot.total_busy_time += completion_time - current_time
    robot.total_travel_distance += travel_to_pickup.distance + travel_to_dropoff.distance
    robot.total_travel_time += travel_to_pickup.realized_travel_time + travel_to_dropoff.realized_travel_time
    robot.total_congestion_delay += assignment.congestion_delay_time
    robot.blocked_traversal_events += assignment.blocked_traversal_events
    robot.available_time = completion_time
    robot.current_node = task.dropoff_node
    robot.completed_task_ids.append(task.task_id)

    return TaskExecution(
        task_id=task.task_id,
        robot_id=robot.spec.robot_id,
        release_time=task.release_time,
        assigned_at=current_time,
        pickup_arrival_time=pickup_arrival_time,
        service_start_time=service_start_time,
        completion_time=completion_time,
        waiting_time=current_time - task.release_time,
        turnaround_time=completion_time - task.release_time,
        execution_model=execution_model,
        travel_to_pickup_time=travel_to_pickup.realized_travel_time,
        travel_to_pickup_distance=travel_to_pickup.distance,
        travel_to_pickup_ideal_time=travel_to_pickup.ideal_travel_time,
        travel_to_pickup_wait_time=travel_to_pickup.wait_time,
        travel_to_pickup_blocked_events=travel_to_pickup.blocked_events,
        travel_to_pickup_path_nodes=travel_to_pickup.path_nodes,
        travel_to_pickup_path_arcs=travel_to_pickup.path_arcs,
        travel_to_dropoff_time=travel_to_dropoff.realized_travel_time,
        travel_to_dropoff_distance=travel_to_dropoff.distance,
        travel_to_dropoff_ideal_time=travel_to_dropoff.ideal_travel_time,
        travel_to_dropoff_wait_time=travel_to_dropoff.wait_time,
        travel_to_dropoff_blocked_events=travel_to_dropoff.blocked_events,
        travel_to_dropoff_path_nodes=travel_to_dropoff.path_nodes,
        travel_to_dropoff_path_arcs=travel_to_dropoff.path_arcs,
        congestion_delay_time=assignment.congestion_delay_time,
        blocked_traversal_events=assignment.blocked_traversal_events,
    )


def _eligible_ready_tasks(
    queue: TaskQueue,
    current_time: float,
    config: SimulationConfig,
) -> tuple[Task, ...]:
    ready = queue.ready_tasks(current_time)
    if config.horizon_seconds is None:
        return ready
    return tuple(task for task in ready if task.release_time <= config.horizon_seconds)


def _build_dispatch_context(
    context_builder: DispatchContextBuilder,
    queue: TaskQueue,
    current_time: float,
    robot_states: tuple[RobotState, ...],
    config: SimulationConfig,
    reservation_table: ResourceReservationTable,
):
    pending_tasks = queue.pending_tasks()
    if config.horizon_seconds is not None:
        pending_tasks = tuple(task for task in pending_tasks if task.release_time <= config.horizon_seconds)
    return context_builder.build(
        current_time=current_time,
        robot_states=robot_states,
        pending_tasks=pending_tasks,
        congestion_observation=reservation_table.snapshot(current_time),
        execution_model=config.execution_model.value,
    )


def _next_event_time(
    current_time: float,
    queue: TaskQueue,
    robot_states: tuple[RobotState, ...],
    config: SimulationConfig,
) -> float | None:
    next_times: list[float] = []
    next_release = queue.next_release_time(current_time)
    if next_release is not None:
        if config.horizon_seconds is None or next_release <= config.horizon_seconds:
            next_times.append(next_release)
    next_robot_available = [robot.available_time for robot in robot_states if robot.available_time > current_time]
    if next_robot_available:
        next_times.append(min(next_robot_available))
    if not next_times:
        return None

    next_time = min(next_times)
    if (
        config.horizon_seconds is not None
        and next_time > config.horizon_seconds
        and not config.continue_until_all_tasks_complete
    ):
        return None
    return next_time


def _record_snapshot(
    time: float,
    queue: TaskQueue,
    robot_states: tuple[RobotState, ...],
    executions: list[TaskExecution],
    snapshots: list[QueueSnapshot],
) -> None:
    ready_tasks = len(queue.ready_tasks(time))
    future_tasks = len(queue.pending_tasks()) - ready_tasks
    busy_robots = sum(robot.available_time > time for robot in robot_states)
    snapshot = QueueSnapshot(
        time=time,
        ready_tasks=ready_tasks,
        future_tasks=future_tasks,
        busy_robots=busy_robots,
        completed_tasks=len(executions),
    )
    if snapshots and snapshots[-1].time == snapshot.time:
        snapshots[-1] = snapshot
    else:
        snapshots.append(snapshot)


def _build_dispatch_trace_records(
    context: DispatchContext,
    decision,
    dispatch_index: int,
) -> list[DispatchTraceRecord]:
    traces: list[DispatchTraceRecord] = []
    for candidate in build_candidate_assignment_observations(context):
        traces.append(
            DispatchTraceRecord(
                dispatch_index=dispatch_index,
                decision_time=context.current_time,
                selected_robot_id=decision.robot_id,
                selected_task_id=decision.task_id,
                candidate_robot_id=candidate.robot_id,
                candidate_task_id=candidate.task_id,
                is_selected=(candidate.robot_id == decision.robot_id and candidate.task_id == decision.task_id),
                robot_current_node=candidate.robot_current_node,
                robot_current_zone=candidate.robot_current_zone,
                robot_speed_multiplier=candidate.feature("robot_speed_multiplier"),
                robot_completed_task_count=int(candidate.feature("robot_completed_task_count")),
                robot_total_busy_time=candidate.feature("robot_total_busy_time"),
                robot_total_idle_time=candidate.feature("robot_total_idle_time"),
                robot_total_travel_time=candidate.feature("robot_total_travel_time"),
                robot_total_travel_distance=candidate.feature("robot_total_travel_distance"),
                task_release_time=context.current_time - candidate.feature("task_age"),
                task_age=candidate.feature("task_age"),
                task_priority=int(candidate.feature("task_priority")),
                task_service_time_estimate=candidate.feature("task_service_time_estimate"),
                task_pickup_node=candidate.task_pickup_node,
                task_dropoff_node=candidate.task_dropoff_node,
                task_source_zone=candidate.task_source_zone,
                task_destination_zone=candidate.task_destination_zone,
                travel_to_pickup_time=candidate.feature("travel_to_pickup_time"),
                travel_to_pickup_distance=candidate.feature("travel_to_pickup_distance"),
                pickup_to_dropoff_time=candidate.feature("pickup_to_dropoff_time"),
                pickup_to_dropoff_distance=candidate.feature("pickup_to_dropoff_distance"),
                pickup_node_inbound_degree=int(candidate.feature("pickup_node_inbound_degree")),
                pickup_node_outbound_degree=int(candidate.feature("pickup_node_outbound_degree")),
                dropoff_node_inbound_degree=int(candidate.feature("dropoff_node_inbound_degree")),
                dropoff_node_outbound_degree=int(candidate.feature("dropoff_node_outbound_degree")),
                travel_to_pickup_mean_transit_count=candidate.feature("travel_to_pickup_mean_transit_count"),
                travel_to_pickup_max_transit_count=candidate.feature("travel_to_pickup_max_transit_count"),
                travel_to_pickup_mean_arc_traversal_count=candidate.feature(
                    "travel_to_pickup_mean_arc_traversal_count"
                ),
                travel_to_pickup_max_arc_traversal_count=candidate.feature(
                    "travel_to_pickup_max_arc_traversal_count"
                ),
                pickup_to_dropoff_mean_transit_count=candidate.feature(
                    "pickup_to_dropoff_mean_transit_count"
                ),
                pickup_to_dropoff_max_transit_count=candidate.feature(
                    "pickup_to_dropoff_max_transit_count"
                ),
                pickup_to_dropoff_mean_arc_traversal_count=candidate.feature(
                    "pickup_to_dropoff_mean_arc_traversal_count"
                ),
                pickup_to_dropoff_max_arc_traversal_count=candidate.feature(
                    "pickup_to_dropoff_max_arc_traversal_count"
                ),
                pending_task_count=int(candidate.feature("pending_task_count")),
                ready_task_count=int(candidate.feature("ready_task_count")),
                future_task_count=int(candidate.feature("future_task_count")),
                idle_robot_count=int(candidate.feature("idle_robot_count")),
                busy_robot_count=int(candidate.feature("busy_robot_count")),
                mean_ready_task_age=candidate.feature("mean_ready_task_age"),
                average_robot_time_until_available=candidate.feature("average_robot_time_until_available"),
                execution_model=context.global_observation.execution_model,
                active_reserved_edge_count=context.global_observation.active_reserved_edge_count,
                active_reserved_node_count=context.global_observation.active_reserved_node_count,
                estimated_pickup_congestion_delay=candidate.feature("estimated_pickup_congestion_delay"),
                estimated_dropoff_congestion_delay=candidate.feature("estimated_dropoff_congestion_delay"),
                estimated_pickup_blocked_segments=int(candidate.feature("estimated_pickup_blocked_segments")),
                estimated_dropoff_blocked_segments=int(candidate.feature("estimated_dropoff_blocked_segments")),
            )
        )
    return traces


def _robot_by_id(robot_states: tuple[RobotState, ...], robot_id: str) -> RobotState:
    for robot in robot_states:
        if robot.spec.robot_id == robot_id:
            return robot
    raise KeyError(f"Unknown robot_id: {robot_id}")


def _task_by_id(tasks: tuple[Task, ...], task_id: str) -> Task:
    for task in tasks:
        if task.task_id == task_id:
            return task
    raise KeyError(f"Unknown task_id: {task_id}")
