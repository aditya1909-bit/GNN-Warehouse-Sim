"""Metrics collection helpers for simulation runs."""

from __future__ import annotations

from math import ceil
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
    tardiness_values = [execution.task_tardiness for execution in result.executions]
    on_time_completed = [execution.completed_on_time for execution in result.executions if execution.task_due_time is not None]
    blocked_traversal_events_total = sum(execution.blocked_traversal_events for execution in result.executions)
    total_energy_consumed = sum(robot.total_energy_consumed for robot in result.robot_states)
    total_energy_charged = sum(robot.total_energy_charged for robot in result.robot_states)
    total_charging_time = sum(robot.total_charging_time for robot in result.robot_states)
    charging_events_total = sum(robot.charging_events for robot in result.robot_states)
    battery_depletion_incidents_total = sum(robot.battery_depletion_events for robot in result.robot_states)
    planner_wait_time_total = sum(getattr(plan, "wait_insertion_time", 0.0) for plan in result.planner_plans)
    conflict_count_by_epoch = {
        getattr(plan, "plan_time", None): getattr(plan, "pre_resolution_conflict_count", 0)
        for plan in result.planner_plans
        if getattr(plan, "status", "") == "planned"
    }
    conflict_count_by_epoch.pop(None, None)
    path_conflicts_before_resolution_total = sum(conflict_count_by_epoch.values())
    sipp_wait_insertions_total = sum(getattr(plan, "wait_insertion_count", 0) for plan in result.planner_plans)

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
        total_energy_consumed=total_energy_consumed,
        total_energy_charged=total_energy_charged,
        total_charging_time=total_charging_time,
        charging_events_total=charging_events_total,
        battery_depletion_incidents_total=battery_depletion_incidents_total,
        average_queue_length=_average_ready_queue_length(result.queue_snapshots),
        throughput_per_hour=(len(result.executions) / makespan * 3600.0 if makespan > 0 else 0.0),
        makespan=makespan,
        robot_metrics=robot_metrics,
        safety_violations_total=len(result.collision_events),
        replans_total=len({getattr(plan, "plan_time", None) for plan in result.planner_plans}),
        planner_failures_total=sum(getattr(plan, "status", "") == "failed" for plan in result.planner_plans),
        on_time_completion_rate=(
            sum(1 for item in on_time_completed if item) / len(on_time_completed)
            if on_time_completed
            else None
        ),
        mean_tardiness=(sum(tardiness_values) / len(tardiness_values) if tardiness_values else None),
        p95_tardiness=_percentile(tardiness_values, 95.0),
        overdue_task_count=sum(1 for value in tardiness_values if value > 1e-9),
        planner_wait_time_total=planner_wait_time_total,
        path_conflicts_before_resolution_total=path_conflicts_before_resolution_total,
        sipp_wait_insertions_total=sipp_wait_insertions_total,
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
        battery_level=robot.battery_level,
        total_energy_consumed=robot.total_energy_consumed,
        total_energy_charged=robot.total_energy_charged,
        total_charging_time=robot.total_charging_time,
        charging_events=robot.charging_events,
        battery_depletion_events=robot.battery_depletion_events,
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


def _percentile(values: list[float | int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = max(0, ceil((percentile / 100.0) * len(ordered)) - 1)
    return ordered[min(position, len(ordered) - 1)]
