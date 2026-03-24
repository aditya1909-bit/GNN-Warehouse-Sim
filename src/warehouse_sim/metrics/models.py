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


@dataclass(frozen=True)
class SimulationMetrics:
    """Aggregated metrics for a simulation run."""

    tasks_generated: int
    tasks_completed: int
    tasks_unassigned: int
    average_waiting_time: float | None
    average_turnaround_time: float | None
    average_travel_distance_per_task: float | None
    average_queue_length: float
    throughput_per_hour: float
    makespan: float
    robot_metrics: tuple[RobotMetrics, ...]

