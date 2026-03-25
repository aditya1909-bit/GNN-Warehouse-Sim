"""Models for the first discrete-event warehouse simulation baseline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from warehouse_sim.agents import RobotState
from warehouse_sim.tasks import Task

if TYPE_CHECKING:
    from warehouse_sim.metrics import SimulationMetrics


class ExecutionModel(StrEnum):
    """Supported task-execution fidelity modes."""

    IDEALIZED = "idealized"
    RESERVED_EDGES = "reserved_edges"
    RESERVED_NODES = "reserved_nodes"
    CONTINUOUS = "continuous"


class CoordinationMode(StrEnum):
    """Top-level coordination stack for a simulation run."""

    DISPATCH = "dispatch"
    INTEGRATED = "integrated"


@dataclass(frozen=True)
class CoordinationRuntimeConfig:
    """Runtime settings for integrated coordination mode."""

    control_dt: float = 0.25
    replan_period: float = 1.0
    robot_radius: float = 0.2
    collision_clearance: float = 0.05
    k_shortest_paths: int = 3
    max_route_options_per_pair: int = 3


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a discrete-event baseline simulation."""

    horizon_seconds: float | None = None
    continue_until_all_tasks_complete: bool = True
    coordination_mode: CoordinationMode = CoordinationMode.DISPATCH
    execution_model: ExecutionModel = ExecutionModel.IDEALIZED
    coordination: CoordinationRuntimeConfig | None = None

    def __post_init__(self) -> None:
        if self.horizon_seconds is not None and self.horizon_seconds < 0:
            raise ValueError("horizon_seconds must be >= 0 when provided.")
        if not isinstance(self.coordination_mode, CoordinationMode):
            object.__setattr__(self, "coordination_mode", CoordinationMode(self.coordination_mode))
        if not isinstance(self.execution_model, ExecutionModel):
            object.__setattr__(self, "execution_model", ExecutionModel(self.execution_model))
        if self.coordination_mode == CoordinationMode.INTEGRATED and self.coordination is None:
            raise ValueError("coordination must be provided when coordination_mode is integrated.")


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
    execution_model: ExecutionModel
    travel_to_pickup_time: float
    travel_to_pickup_distance: float
    travel_to_pickup_ideal_time: float
    travel_to_pickup_wait_time: float
    travel_to_pickup_blocked_events: int
    travel_to_pickup_path_nodes: tuple[str, ...]
    travel_to_pickup_path_arcs: tuple[str, ...]
    travel_to_dropoff_time: float
    travel_to_dropoff_distance: float
    travel_to_dropoff_ideal_time: float
    travel_to_dropoff_wait_time: float
    travel_to_dropoff_blocked_events: int
    travel_to_dropoff_path_nodes: tuple[str, ...]
    travel_to_dropoff_path_arcs: tuple[str, ...]
    congestion_delay_time: float
    blocked_traversal_events: int


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
    pickup_node_inbound_degree: int
    pickup_node_outbound_degree: int
    dropoff_node_inbound_degree: int
    dropoff_node_outbound_degree: int
    travel_to_pickup_mean_transit_count: float
    travel_to_pickup_max_transit_count: float
    travel_to_pickup_mean_arc_traversal_count: float
    travel_to_pickup_max_arc_traversal_count: float
    pickup_to_dropoff_mean_transit_count: float
    pickup_to_dropoff_max_transit_count: float
    pickup_to_dropoff_mean_arc_traversal_count: float
    pickup_to_dropoff_max_arc_traversal_count: float
    pending_task_count: int
    ready_task_count: int
    future_task_count: int
    idle_robot_count: int
    busy_robot_count: int
    mean_ready_task_age: float
    average_robot_time_until_available: float
    execution_model: str
    active_reserved_edge_count: int
    active_reserved_node_count: int
    estimated_pickup_congestion_delay: float
    estimated_dropoff_congestion_delay: float
    estimated_pickup_blocked_segments: int
    estimated_dropoff_blocked_segments: int


@dataclass(frozen=True)
class DispatchNodeObservationRecord:
    """Dynamic node-level graph observation captured at a dispatch event."""

    dispatch_index: int
    decision_time: float
    node_id: str
    is_robot_occupied: bool
    robot_count: int
    is_ready_task_pickup: bool
    is_ready_task_dropoff: bool
    is_selected_task_pickup: bool
    is_selected_task_dropoff: bool
    is_reserved_node: bool
    reserved_time_remaining: float


@dataclass(frozen=True)
class DispatchArcObservationRecord:
    """Dynamic directed-arc graph observation captured at a dispatch event."""

    dispatch_index: int
    decision_time: float
    source_id: str
    target_id: str
    is_reserved_arc: bool
    reserved_time_remaining: float


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
    dispatch_node_observations: tuple[DispatchNodeObservationRecord, ...]
    dispatch_arc_observations: tuple[DispatchArcObservationRecord, ...]
    unassigned_tasks: tuple[Task, ...]
    queue_snapshots: tuple[QueueSnapshot, ...]
    metrics: "SimulationMetrics"
    robot_trajectories: tuple[object, ...] = ()
    macro_decisions: tuple[object, ...] = ()
    collision_events: tuple[object, ...] = ()
    planner_plans: tuple[object, ...] = ()
