"""Metrics collection helpers for simulation runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from warehouse_sim.metrics.models import RobotMetrics, SimulationMetrics

if TYPE_CHECKING:
    from warehouse_sim.simulation.models import QueueSnapshot, SimulationResult


def compute_simulation_metrics(result: "SimulationResult") -> SimulationMetrics:
    """Compute summary metrics from a completed simulation result."""

    waiting_times = [execution.waiting_time for execution in result.executions]
    turnaround_times = [execution.turnaround_time for execution in result.executions]
    realized_travel_times = [
        execution.travel_to_pickup_time + execution.travel_to_dropoff_time for execution in result.executions
    ]
    travel_distances = [execution.travel_to_pickup_distance + execution.travel_to_dropoff_distance for execution in result.executions]
    congestion_delays = [execution.congestion_delay_time for execution in result.executions]
    blocked_traversal_events_total = sum(execution.blocked_traversal_events for execution in result.executions)

    makespan = result.finished_at - result.started_at
    robot_metrics = tuple(
        _build_robot_metrics(result=result, robot=robot, makespan=makespan)
        for robot in result.robot_states
    )

    return SimulationMetrics(
        tasks_generated=result.tasks_generated,
        tasks_completed=len(result.executions),
        tasks_unassigned=len(result.unassigned_tasks),
        average_waiting_time=(
            sum(waiting_times) / len(waiting_times) if waiting_times else None
        ),
        average_turnaround_time=(
            sum(turnaround_times) / len(turnaround_times) if turnaround_times else None
        ),
        average_travel_distance_per_task=(
            sum(travel_distances) / len(travel_distances) if travel_distances else None
        ),
        realized_travel_time_total=sum(realized_travel_times),
        realized_travel_distance_total=sum(travel_distances),
        congestion_delay_total=sum(congestion_delays),
        average_congestion_delay_per_completed_task=(
            sum(congestion_delays) / len(congestion_delays) if congestion_delays else None
        ),
        blocked_traversal_events_total=blocked_traversal_events_total,
        average_queue_length=_average_ready_queue_length(result.queue_snapshots),
        throughput_per_hour=(len(result.executions) / makespan * 3600.0 if makespan > 0 else 0.0),
        makespan=makespan,
        robot_metrics=robot_metrics,
        safety_violations_total=len(result.collision_events),
        replans_total=len({getattr(plan, "plan_time", None) for plan in result.planner_plans}),
        planner_failures_total=sum(getattr(plan, "status", "") == "failed" for plan in result.planner_plans),
    )


def _build_robot_metrics(
    result: "SimulationResult",
    robot,
    makespan: float,
) -> RobotMetrics:
    total_idle_time = robot.total_idle_time + max(result.finished_at - robot.available_time, 0.0)
    utilization = robot.total_busy_time / makespan if makespan > 0 else 0.0
    return RobotMetrics(
        robot_id=robot.spec.robot_id,
        tasks_completed=len(robot.completed_task_ids),
        utilization=utilization,
        busy_time=robot.total_busy_time,
        idle_time=total_idle_time,
        travel_time=robot.total_travel_time,
        travel_distance=robot.total_travel_distance,
        congestion_delay_time=robot.total_congestion_delay,
        blocked_traversal_events=robot.blocked_traversal_events,
    )


def _average_ready_queue_length(snapshots: tuple["QueueSnapshot", ...]) -> float:
    if len(snapshots) < 2:
        return float(snapshots[0].ready_tasks) if snapshots else 0.0

    total_area = 0.0
    for left, right in zip(snapshots, snapshots[1:]):
        total_area += left.ready_tasks * (right.time - left.time)

    total_time = snapshots[-1].time - snapshots[0].time
    if total_time <= 0:
        return float(snapshots[-1].ready_tasks)
    return total_area / total_time
