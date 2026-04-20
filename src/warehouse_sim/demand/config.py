"""Configuration models for synthetic warehouse demand generation."""

from __future__ import annotations

from dataclasses import dataclass


class DemandValidationError(ValueError):
    """Raised when demand-generation parameters are invalid."""


class DemandGenerationError(RuntimeError):
    """Raised when valid parameters still cannot produce a usable dataset."""


@dataclass(frozen=True)
class TaskMetadataConfig:
    """Optional sampling rules for richer task metadata columns.

    Metadata remains opt-in so the default CSV contract stays backward
    compatible with the original four-column generator output.
    """

    task_types: tuple[str, ...] = ("pick", "replenishment", "cycle_count")
    source_zones: tuple[str, ...] = ("storage_a", "storage_b", "inbound")
    destination_zones: tuple[str, ...] = ("pick_station_1", "pick_station_2", "staging")
    priorities: tuple[int, ...] = (1, 2, 3)
    service_duration_low: float = 30.0
    service_duration_high: float = 180.0
    due_time_slack_low: float | None = None
    due_time_slack_high: float | None = None

    def __post_init__(self) -> None:
        if not self.task_types:
            raise DemandValidationError("task_types must contain at least one value.")
        if not self.source_zones:
            raise DemandValidationError("source_zones must contain at least one value.")
        if not self.destination_zones:
            raise DemandValidationError("destination_zones must contain at least one value.")
        if not self.priorities:
            raise DemandValidationError("priorities must contain at least one value.")
        if self.service_duration_low <= 0:
            raise DemandValidationError("service_duration_low must be > 0.")
        if self.service_duration_high <= 0:
            raise DemandValidationError("service_duration_high must be > 0.")
        if self.service_duration_low > self.service_duration_high:
            raise DemandValidationError(
                "service_duration_low must be <= service_duration_high."
            )
        if (self.due_time_slack_low is None) != (self.due_time_slack_high is None):
            raise DemandValidationError(
                "due_time_slack_low and due_time_slack_high must be set together."
            )
        if self.due_time_slack_low is not None and self.due_time_slack_low <= 0:
            raise DemandValidationError("due_time_slack_low must be > 0 when provided.")
        if self.due_time_slack_high is not None and self.due_time_slack_high <= 0:
            raise DemandValidationError("due_time_slack_high must be > 0 when provided.")
        if (
            self.due_time_slack_low is not None
            and self.due_time_slack_high is not None
            and self.due_time_slack_low > self.due_time_slack_high
        ):
            raise DemandValidationError("due_time_slack_low must be <= due_time_slack_high.")


@dataclass(frozen=True)
class DemandGenerationConfig:
    """Parameters for the stage-1 warehouse task-arrival generator.

    The arrival process is a non-homogeneous Poisson process simulated via
    thinning: a base exponential arrival rate, an optional morning rush rate
    multiplier, and an optional lunch shutdown window with zero arrivals.
    """

    horizon_seconds: float = 28_800.0
    mean_interval: float = 10.0
    rush_start: float = 1_800.0
    rush_end: float = 7_200.0
    rush_multiplier: float = 2.0
    lunch_start: float = 14_400.0
    lunch_end: float = 16_200.0
    seed: int = 7
    min_tasks: int = 200

    def __post_init__(self) -> None:
        if self.horizon_seconds <= 0:
            raise DemandValidationError("horizon_seconds must be > 0.")
        if self.mean_interval <= 0:
            raise DemandValidationError("mean_interval must be > 0.")
        if self.rush_multiplier <= 0:
            raise DemandValidationError("rush_multiplier must be > 0.")
        if self.min_tasks < 0:
            raise DemandValidationError("min_tasks must be >= 0.")

        _validate_window("rush", self.rush_start, self.rush_end, self.horizon_seconds)
        _validate_window("lunch", self.lunch_start, self.lunch_end, self.horizon_seconds)

        if _windows_overlap(
            self.rush_start,
            self.rush_end,
            self.lunch_start,
            self.lunch_end,
        ):
            raise DemandValidationError(
                "rush and lunch windows overlap. Stage 1 rejects overlapping regime windows."
            )


def _validate_window(name: str, start: float, end: float, horizon_seconds: float) -> None:
    if start < 0:
        raise DemandValidationError(f"{name}_start must be >= 0.")
    if end < 0:
        raise DemandValidationError(f"{name}_end must be >= 0.")
    if start > end:
        raise DemandValidationError(f"{name}_start must be <= {name}_end.")
    if start > horizon_seconds:
        raise DemandValidationError(f"{name}_start must be <= horizon_seconds.")
    if end > horizon_seconds:
        raise DemandValidationError(f"{name}_end must be <= horizon_seconds.")


def _windows_overlap(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> bool:
    return max(first_start, second_start) < min(first_end, second_end)
