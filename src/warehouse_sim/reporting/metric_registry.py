"""Metric extraction helpers that populate the stable reporting schema."""

from __future__ import annotations

from math import ceil
from typing import Mapping, TYPE_CHECKING

from warehouse_sim.reporting.metrics_schema import METRIC_SCHEMA_VERSION, default_metric_payload

if TYPE_CHECKING:
    from warehouse_sim.simulation.models import QueueSnapshot, SimulationResult


def build_simulation_metric_record(result: "SimulationResult") -> dict[str, float | int | None]:
    """Extract canonical benchmark metrics from one simulation result."""

    payload = default_metric_payload()
    turnaround_times = [execution.turnaround_time for execution in result.executions]
    queue_lengths = [snapshot.ready_tasks for snapshot in result.queue_snapshots]
    robot_busy_time = sum(metric.busy_time for metric in result.metrics.robot_metrics)
    robot_idle_time = sum(metric.idle_time for metric in result.metrics.robot_metrics)
    planner_statuses = [getattr(plan, "status", "") for plan in result.planner_plans]
    replanning_epochs = {getattr(plan, "plan_time", None) for plan in result.planner_plans}
    replanning_epochs.discard(None)

    payload.update(
        {
            "throughput": result.metrics.throughput_per_hour,
            "mean_task_completion_time": result.metrics.average_turnaround_time,
            "p95_task_completion_time": _percentile(turnaround_times, 95.0),
            "makespan": result.metrics.makespan,
            "mean_queue_length": result.metrics.average_queue_length,
            "p95_queue_length": _percentile(queue_lengths, 95.0),
            "robot_idle_fraction": (
                robot_idle_time / (robot_busy_time + robot_idle_time)
                if robot_busy_time + robot_idle_time > 0
                else 0.0
            ),
            "travel_distance_per_completed_task": result.metrics.average_travel_distance_per_task,
            "realized_waiting_time": sum(
                execution.travel_to_pickup_wait_time + execution.travel_to_dropoff_wait_time
                for execution in result.executions
            ),
            "congestion_event_count": result.metrics.blocked_traversal_events_total,
            "collision_event_count": result.metrics.safety_violations_total,
            "deadlock_livelock_incident_count": 0,
            "on_time_completion_rate": result.metrics.on_time_completion_rate,
            "mean_tardiness": result.metrics.mean_tardiness,
            "p95_tardiness": result.metrics.p95_tardiness,
            "overdue_task_count": result.metrics.overdue_task_count,
            "planning_latency": None,
            "replanning_count": len(replanning_epochs),
            "planner_failure_count": result.metrics.planner_failures_total,
            "timeout_count": sum(status == "timeout" for status in planner_statuses),
            "path_conflict_count_before_resolution": result.metrics.path_conflicts_before_resolution_total,
            "sipp_wait_insertion_count": result.metrics.sipp_wait_insertions_total,
            "planner_wait_time_total": result.metrics.planner_wait_time_total,
            "mapf_solve_success_rate": (
                sum(status == "planned" for status in planner_statuses) / len(planner_statuses)
                if planner_statuses
                else None
            ),
        }
    )
    return payload


def build_learning_metric_record(metrics: Mapping[str, object] | None = None) -> dict[str, float | int | None]:
    """Populate the learning portion of the canonical metric schema."""

    payload = default_metric_payload()
    if metrics is None:
        return payload
    for metric_name in payload:
        value = metrics.get(metric_name)
        if value is None:
            continue
        if not isinstance(value, int | float):
            raise ValueError(f"Learning metric {metric_name!r} must be numeric or null.")
        payload[metric_name] = value
    return payload


def metric_schema_version() -> str:
    """Expose the current metric schema version."""

    return METRIC_SCHEMA_VERSION


def _percentile(values: list[float | int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = max(0, ceil((percentile / 100.0) * len(ordered)) - 1)
    return ordered[min(position, len(ordered) - 1)]
