"""Experiment configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class ConfigValidationError(ValueError):
    """Raised when an experiment configuration is invalid."""


GridCoordinate = tuple[int, int]
DirectedEdgeCoordinate = tuple[GridCoordinate, GridCoordinate]


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
    blocked_cells: tuple[GridCoordinate, ...] = ()
    directed_edges: tuple[DirectedEdgeCoordinate, ...] = ()

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise ConfigValidationError("layout.rows must be > 0.")
        if self.columns <= 0:
            raise ConfigValidationError("layout.columns must be > 0.")


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
class SimulationRunConfig:
    """Declarative simulation-run settings."""

    policy: str = "fifo"
    horizon_seconds: float | None = None
    continue_until_all_tasks_complete: bool = True
    execution_model: str = "idealized"

    def __post_init__(self) -> None:
        supported_models = {"idealized", "reserved_edges", "reserved_nodes"}
        if self.execution_model not in supported_models:
            raise ConfigValidationError(
                "simulation.execution_model must be one of: idealized, reserved_edges, reserved_nodes"
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

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigValidationError("name must be non-empty.")
        if self.simulation.policy == "linear_assignment_model" and self.policy_model is None:
            raise ConfigValidationError(
                "policy_model must be provided when simulation.policy is linear_assignment_model."
            )
        if self.simulation.policy == "linear_assignment_model" and not self.policy_model.weights:
            raise ConfigValidationError(
                "policy_model.weights must be provided when simulation.policy is linear_assignment_model."
            )
        if self.simulation.policy in {"trained_linear_model", "trained_mlp_model"}:
            if self.policy_model is None or self.policy_model.artifact_path is None:
                raise ConfigValidationError(
                    "policy_model.artifact_path must be provided for trained model policies."
                )
