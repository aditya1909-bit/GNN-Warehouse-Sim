"""Models for the first discrete-event warehouse simulation baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from warehouse_sim.agents import RobotState
from warehouse_sim.tasks import Task

if TYPE_CHECKING:
    from warehouse_sim.metrics import SimulationMetrics


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a discrete-event baseline simulation."""

    horizon_seconds: float | None = None
    continue_until_all_tasks_complete: bool = True

    def __post_init__(self) -> None:
        if self.horizon_seconds is not None and self.horizon_seconds < 0:
            raise ValueError("horizon_seconds must be >= 0 when provided.")


@dataclass(frozen=True)
class TaskExecution:
    """Execution record for a completed task assignment."""

    task_id: str
    robot_id: str
    release_time: float
    assigned_at: float
    pickup_arrival_time: float
    service_start_time: float
    completion_time: float
    waiting_time: float
    turnaround_time: float
    travel_to_pickup_time: float
    travel_to_pickup_distance: float
    travel_to_dropoff_time: float
    travel_to_dropoff_distance: float


@dataclass(frozen=True)
class QueueSnapshot:
    """Event-time queue snapshot for metrics and visualization."""

    time: float
    ready_tasks: int
    future_tasks: int
    busy_robots: int
    completed_tasks: int


@dataclass(frozen=True)
class DispatchTraceRecord:
    """Flattened candidate-pair observation captured at a dispatch event."""

    dispatch_index: int
    decision_time: float
    selected_robot_id: str
    selected_task_id: str
    candidate_robot_id: str
    candidate_task_id: str
    is_selected: bool
    robot_current_node: str
    robot_current_zone: str | None
    robot_speed_multiplier: float
    robot_completed_task_count: int
    robot_total_busy_time: float
    robot_total_idle_time: float
    robot_total_travel_time: float
    robot_total_travel_distance: float
    task_release_time: float
    task_age: float
    task_priority: int
    task_service_time_estimate: float
    task_pickup_node: str
    task_dropoff_node: str
    task_source_zone: str | None
    task_destination_zone: str | None
    travel_to_pickup_time: float
    travel_to_pickup_distance: float
    pickup_to_dropoff_time: float
    pickup_to_dropoff_distance: float
    pending_task_count: int
    ready_task_count: int
    future_task_count: int
    idle_robot_count: int
    busy_robot_count: int
    mean_ready_task_age: float


@dataclass(frozen=True)
class SimulationResult:
    """Completed simulation run and derived metrics."""

    policy_name: str
    started_at: float
    finished_at: float
    tasks_generated: int
    robot_states: tuple[RobotState, ...]
    executions: tuple[TaskExecution, ...]
    dispatch_traces: tuple[DispatchTraceRecord, ...]
    unassigned_tasks: tuple[Task, ...]
    queue_snapshots: tuple[QueueSnapshot, ...]
    metrics: "SimulationMetrics"
