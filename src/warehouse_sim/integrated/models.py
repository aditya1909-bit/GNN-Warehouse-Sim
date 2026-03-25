"""Models for integrated continuous-time coordination."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TimedWaypoint:
    """One node-time point on a robot trajectory."""

    node_id: str
    time: float


@dataclass(frozen=True)
class TimedTraversal:
    """Continuous traversal over one directed arc."""

    robot_id: str
    source_id: str
    target_id: str
    start_time: float
    end_time: float
    distance: float
    travel_time: float
    task_id: str | None = None
    phase: str = "transit"


@dataclass(frozen=True)
class IntegratedRobotTrajectoryRecord:
    """Flattened traversal record for report output."""

    robot_id: str
    task_id: str | None
    phase: str
    source_id: str
    target_id: str
    start_time: float
    end_time: float
    distance: float
    travel_time: float


@dataclass(frozen=True)
class MacroCandidate:
    """One macro action available to a robot at a replanning epoch."""

    macro_type: str
    task_id: str | None = None
    route_nodes: tuple[str, ...] = ()
    route_edges: tuple[tuple[str, str], ...] = ()
    estimated_completion_time: float = 0.0
    pickup_node: str | None = None
    dropoff_node: str | None = None


@dataclass(frozen=True)
class MacroDecisionRecord:
    """Chosen macro action for a robot at a replanning epoch."""

    decision_index: int
    decision_time: float
    robot_id: str
    macro_type: str
    task_id: str | None
    route_nodes: tuple[str, ...]
    route_edges: tuple[str, ...]
    estimated_completion_time: float
    selected_by_policy: str


@dataclass(frozen=True)
class CollisionEventRecord:
    """Explicit collision or safety violation record."""

    time: float
    robot_id: str
    other_robot_id: str | None
    event_type: str
    location_id: str
    severity: str = "violation"


@dataclass(frozen=True)
class PlannerPlanRecord:
    """One planner output for a robot at a replanning epoch."""

    plan_index: int
    plan_time: float
    robot_id: str
    task_id: str | None
    priority_rank: int
    path_nodes: tuple[str, ...]
    path_edges: tuple[str, ...]
    planned_start_time: float
    planned_end_time: float
    planner_name: str
    status: str


@dataclass
class IntegratedRobotRuntimeState:
    """Mutable runtime state for integrated coordination."""

    robot_id: str
    current_node: str
    speed_multiplier: float
    available_time: float = 0.0
    total_busy_time: float = 0.0
    total_idle_time: float = 0.0
    total_travel_time: float = 0.0
    total_travel_distance: float = 0.0
    total_congestion_delay: float = 0.0
    blocked_traversal_events: int = 0
    completed_task_ids: list[str] = field(default_factory=list)
    current_task_id: str | None = None
    plan_valid_until: float = 0.0


@dataclass(frozen=True)
class OccupancyObservation:
    """Continuous occupancy summary for learning and reporting."""

    active_edge_occupancies: tuple[TimedTraversal, ...]
    active_node_occupancies: tuple[TimedWaypoint, ...]


@dataclass(frozen=True)
class IntegratedObservation:
    """Centralized observation at a replanning epoch."""

    current_time: float
    graph_node_ids: tuple[str, ...]
    edge_index: tuple[tuple[int, int], ...]
    node_features: tuple[tuple[float, ...], ...]
    edge_features: tuple[tuple[float, ...], ...]
    robot_features: tuple[tuple[float, ...], ...]
    task_features: tuple[tuple[float, ...], ...]
    robot_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    macro_candidates: tuple[tuple[MacroCandidate, ...], ...]


@dataclass(frozen=True)
class IntegratedPolicyStep:
    """One sampled policy step for PPO fine-tuning."""

    observation: IntegratedObservation
    chosen_indices: tuple[int, ...]
    old_log_prob: float
    reward: float
    value: float
    done: bool
