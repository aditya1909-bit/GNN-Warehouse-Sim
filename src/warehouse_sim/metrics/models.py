"""Metrics models for simulation runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotMetrics:
    """Per-robot summary metrics for a simulation run."""

    robot_id: str
    tasks_completed: int
    utilization: float
    busy_time: float
    idle_time: float
    travel_time: float
    travel_distance: float
    congestion_delay_time: float
    blocked_traversal_events: int
    battery_level: float
    total_energy_consumed: float
    total_energy_charged: float
    total_charging_time: float
    charging_events: int
    battery_depletion_events: int


@dataclass(frozen=True)
class SimulationMetrics:
    """Aggregated metrics for a simulation run."""

    tasks_generated: int
    tasks_completed: int
    tasks_unassigned: int
    average_waiting_time: float | None
    average_turnaround_time: float | None
    average_travel_distance_per_task: float | None
    realized_travel_time_total: float
    realized_travel_distance_total: float
    congestion_delay_total: float
    average_congestion_delay_per_completed_task: float | None
    blocked_traversal_events_total: int
    total_energy_consumed: float
    total_energy_charged: float
    total_charging_time: float
    charging_events_total: int
    battery_depletion_incidents_total: int
    average_queue_length: float
    throughput_per_hour: float
    makespan: float
    robot_metrics: tuple[RobotMetrics, ...]
    safety_violations_total: int = 0
    replans_total: int = 0
    planner_failures_total: int = 0
