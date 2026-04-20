"""Experiment configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class ConfigValidationError(ValueError):
    """Raised when an experiment configuration is invalid."""


GridCoordinate = tuple[int, int]
DirectedEdgeCoordinate = tuple[GridCoordinate, GridCoordinate]
Point2D = tuple[float, float]


@dataclass(frozen=True)
class LayoutConfig:
    """Declarative layout configuration for a baseline experiment."""

    rows: int
    columns: int
    edge_length: float = 1.0
    travel_speed: float = 1.0
    storage_cell: GridCoordinate = (0, 0)
    dropoff_cell: GridCoordinate = (2, 2)
    staging_cell: GridCoordinate = (2, 0)
    charging_cells: tuple[GridCoordinate, ...] = ()
    blocked_cells: tuple[GridCoordinate, ...] = ()
    obstacle_polygons: tuple[tuple[Point2D, ...], ...] = ()
    directed_edges: tuple[DirectedEdgeCoordinate, ...] = ()
    zone_cells: dict[str, tuple[GridCoordinate, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise ConfigValidationError("layout.rows must be > 0.")
        if self.columns <= 0:
            raise ConfigValidationError("layout.columns must be > 0.")
        for polygon in self.obstacle_polygons:
            if len(polygon) < 3:
                raise ConfigValidationError("layout.obstacle_polygons must contain at least 3 vertices per polygon.")
        for zone_name, cells in self.zone_cells.items():
            if not zone_name:
                raise ConfigValidationError("layout.zone_cells keys must be non-empty.")
            if not cells:
                raise ConfigValidationError("layout.zone_cells entries must contain at least one cell.")


@dataclass(frozen=True)
class BatteryConfig:
    """Declarative battery and charging configuration."""

    enabled: bool = False
    capacity: float = 100.0
    initial_charge_fraction: float = 1.0
    travel_energy_per_distance: float = 1.0
    service_energy: float = 0.0
    charge_rate: float = 5.0
    dispatch_charge_threshold: float = 0.2
    minimum_reserve_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ConfigValidationError("battery.capacity must be > 0.")
        if not 0.0 <= self.initial_charge_fraction <= 1.0:
            raise ConfigValidationError("battery.initial_charge_fraction must be between 0 and 1.")
        if self.travel_energy_per_distance < 0:
            raise ConfigValidationError("battery.travel_energy_per_distance must be >= 0.")
        if self.service_energy < 0:
            raise ConfigValidationError("battery.service_energy must be >= 0.")
        if self.charge_rate <= 0:
            raise ConfigValidationError("battery.charge_rate must be > 0.")
        if not 0.0 <= self.dispatch_charge_threshold <= 1.0:
            raise ConfigValidationError("battery.dispatch_charge_threshold must be between 0 and 1.")
        if not 0.0 <= self.minimum_reserve_fraction <= 1.0:
            raise ConfigValidationError("battery.minimum_reserve_fraction must be between 0 and 1.")


@dataclass(frozen=True)
class DemandConfig:
    """Declarative demand-generation configuration."""

    horizon_seconds: float
    mean_interval: float
    seed: int = 7
    min_tasks: int = 0
    rush_start: float | None = None
    rush_end: float | None = None
    rush_multiplier: float = 1.0
    lunch_start: float | None = None
    lunch_end: float | None = None

    def __post_init__(self) -> None:
        if self.horizon_seconds <= 0:
            raise ConfigValidationError("demand.horizon_seconds must be > 0.")
        if self.mean_interval <= 0:
            raise ConfigValidationError("demand.mean_interval must be > 0.")
        if self.min_tasks < 0:
            raise ConfigValidationError("demand.min_tasks must be >= 0.")


@dataclass(frozen=True)
class RobotsConfig:
    """Declarative robot-fleet configuration."""

    count: int
    speed_multiplier: float = 1.0
    initial_zone: str = "staging_zone"

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ConfigValidationError("robots.count must be > 0.")
        if self.speed_multiplier <= 0:
            raise ConfigValidationError("robots.speed_multiplier must be > 0.")
        if not self.initial_zone:
            raise ConfigValidationError("robots.initial_zone must be non-empty.")


@dataclass(frozen=True)
class TasksConfig:
    """Declarative defaults for converting demand records into tasks."""

    default_pickup_zone: str = "storage_zone"
    default_dropoff_zone: str = "dropoff_zone"
    default_task_type: str = "pick"
    default_priority: int = 1
    default_service_time_estimate: float = 60.0
    task_id_prefix: str = "task"


@dataclass(frozen=True)
class TaskMetadataConfig:
    """Declarative sampling rules for richer task metadata in simulation scenarios."""

    task_types: tuple[str, ...] = ("pick",)
    source_zones: tuple[str, ...] = ()
    destination_zones: tuple[str, ...] = ()
    priorities: tuple[int, ...] = (1,)
    service_duration_low: float = 30.0
    service_duration_high: float = 30.0
    due_time_slack_low: float | None = None
    due_time_slack_high: float | None = None

    def __post_init__(self) -> None:
        if not self.task_types:
            raise ConfigValidationError("task_metadata.task_types must contain at least one value.")
        if self.source_zones is not None and len(self.source_zones) == 0:
            raise ConfigValidationError("task_metadata.source_zones must contain at least one value when provided.")
        if self.destination_zones is not None and len(self.destination_zones) == 0:
            raise ConfigValidationError(
                "task_metadata.destination_zones must contain at least one value when provided."
            )
        if not self.priorities:
            raise ConfigValidationError("task_metadata.priorities must contain at least one value.")
        if self.service_duration_low <= 0:
            raise ConfigValidationError("task_metadata.service_duration_low must be > 0.")
        if self.service_duration_high <= 0:
            raise ConfigValidationError("task_metadata.service_duration_high must be > 0.")
        if self.service_duration_low > self.service_duration_high:
            raise ConfigValidationError(
                "task_metadata.service_duration_low must be <= service_duration_high."
            )
        if (self.due_time_slack_low is None) != (self.due_time_slack_high is None):
            raise ConfigValidationError(
                "task_metadata.due_time_slack_low and task_metadata.due_time_slack_high must be set together."
            )
        if self.due_time_slack_low is not None and self.due_time_slack_low <= 0:
            raise ConfigValidationError("task_metadata.due_time_slack_low must be > 0 when provided.")
        if self.due_time_slack_high is not None and self.due_time_slack_high <= 0:
            raise ConfigValidationError("task_metadata.due_time_slack_high must be > 0 when provided.")
        if (
            self.due_time_slack_low is not None
            and self.due_time_slack_high is not None
            and self.due_time_slack_low > self.due_time_slack_high
        ):
            raise ConfigValidationError(
                "task_metadata.due_time_slack_low must be <= due_time_slack_high."
            )


@dataclass(frozen=True)
class PolicyModelConfig:
    """Declarative configuration for observation-driven policy models."""

    bias: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)
    artifact_path: Path | None = None

    def __post_init__(self) -> None:
        for feature_name, weight in self.weights.items():
            if not feature_name:
                raise ConfigValidationError("policy_model weight names must be non-empty.")
            if not isinstance(weight, int | float):
                raise ConfigValidationError(
                    f"policy_model weight for {feature_name} must be numeric."
                )
        if self.artifact_path is not None and not str(self.artifact_path):
            raise ConfigValidationError("policy_model.artifact_path must be non-empty when provided.")


@dataclass(frozen=True)
class CoordinationConfig:
    """Declarative configuration for integrated coordination mode."""

    motion_model: str = "graph_embedded"
    control_dt: float = 0.25
    replan_period: float = 1.0
    robot_radius: float = 0.2
    collision_clearance: float = 0.05
    k_shortest_paths: int = 3
    max_route_options_per_pair: int = 3

    def __post_init__(self) -> None:
        if self.motion_model not in {"graph_embedded", "free_space", "obstacle_aware_free_space"}:
            raise ConfigValidationError(
                "coordination.motion_model must be one of: graph_embedded, free_space, obstacle_aware_free_space."
            )
        if self.control_dt <= 0:
            raise ConfigValidationError("coordination.control_dt must be > 0.")
        if self.replan_period <= 0:
            raise ConfigValidationError("coordination.replan_period must be > 0.")
        if self.robot_radius <= 0:
            raise ConfigValidationError("coordination.robot_radius must be > 0.")
        if self.collision_clearance < 0:
            raise ConfigValidationError("coordination.collision_clearance must be >= 0.")
        if self.k_shortest_paths <= 0:
            raise ConfigValidationError("coordination.k_shortest_paths must be > 0.")
        if self.max_route_options_per_pair <= 0:
            raise ConfigValidationError("coordination.max_route_options_per_pair must be > 0.")


@dataclass(frozen=True)
class SimulationRunConfig:
    """Declarative simulation-run settings."""

    policy: str = "fifo"
    horizon_seconds: float | None = None
    continue_until_all_tasks_complete: bool = True
    coordination_mode: str = "dispatch"
    execution_model: str = "idealized"

    def __post_init__(self) -> None:
        supported_models = {"idealized", "reserved_edges", "reserved_nodes"}
        supported_coordination_modes = {"dispatch", "integrated"}
        if self.execution_model not in supported_models:
            raise ConfigValidationError(
                "simulation.execution_model must be one of: idealized, reserved_edges, reserved_nodes"
            )
        if self.coordination_mode not in supported_coordination_modes:
            raise ConfigValidationError(
                "simulation.coordination_mode must be one of: dispatch, integrated"
            )


@dataclass(frozen=True)
class ReportingConfig:
    """Declarative output/reporting settings."""

    output_dir: Path
    write_plots: bool = False
    write_observation_dataset: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str
    layout: LayoutConfig
    demand: DemandConfig
    robots: RobotsConfig
    tasks: TasksConfig
    simulation: SimulationRunConfig
    reporting: ReportingConfig
    policy_model: PolicyModelConfig | None = None
    coordination: CoordinationConfig | None = None
    battery: BatteryConfig | None = None
    task_metadata: TaskMetadataConfig | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigValidationError("name must be non-empty.")
        dispatch_policies = {
            "fifo",
            "random",
            "nearest_robot_task",
            "nearest_task_for_idle_robot",
            "congestion_aware_nearest_robot_task",
            "linear_assignment_model",
            "trained_linear_model",
            "trained_mlp_model",
            "trained_graph_dispatch_model",
        }
        integrated_policies = {
            "prioritized_sipp_coordinator",
            "optimal_mapf_coordinator",
            "trained_end_to_end_macro_ppo",
            "random_macro",
        }
        if self.simulation.coordination_mode == "dispatch" and self.simulation.policy not in dispatch_policies:
            raise ConfigValidationError(
                f"simulation.policy {self.simulation.policy!r} is not supported in dispatch mode."
            )
        if self.simulation.coordination_mode == "integrated" and self.simulation.policy not in integrated_policies:
            raise ConfigValidationError(
                f"simulation.policy {self.simulation.policy!r} is not supported in integrated mode."
            )
        if self.simulation.policy == "linear_assignment_model" and self.policy_model is None:
            raise ConfigValidationError(
                "policy_model must be provided when simulation.policy is linear_assignment_model."
            )
        if self.simulation.policy == "linear_assignment_model" and not self.policy_model.weights:
            raise ConfigValidationError(
                "policy_model.weights must be provided when simulation.policy is linear_assignment_model."
            )
        if self.simulation.policy in {
            "trained_linear_model",
            "trained_mlp_model",
            "trained_graph_dispatch_model",
            "trained_end_to_end_macro_ppo",
        }:
            if self.policy_model is None or self.policy_model.artifact_path is None:
                raise ConfigValidationError(
                    "policy_model.artifact_path must be provided for trained model policies."
                )
        if self.simulation.coordination_mode == "integrated":
            if self.coordination is None:
                raise ConfigValidationError(
                    "coordination config must be provided when simulation.coordination_mode is integrated."
                )
            if self.simulation.execution_model != "idealized":
                raise ConfigValidationError(
                    "simulation.execution_model must remain idealized in integrated mode; continuous execution is implied."
                )
