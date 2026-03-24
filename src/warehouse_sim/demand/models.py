"""Typed record models and CSV schema for demand-generation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RegimeLabel = Literal["base", "morning_rush", "lunch"]

LEGACY_CSV_COLUMNS: tuple[str, ...] = (
    "Task_ID",
    "Timestamp",
    "Interarrival_Time",
    "Regime",
)
METADATA_CSV_COLUMNS: tuple[str, ...] = (
    "Task_Type",
    "Source_Zone",
    "Destination_Zone",
    "Priority",
    "Service_Duration",
)

CSV_SCHEMA_DESCRIPTIONS: dict[str, str] = {
    "Task_ID": "1-based sequential identifier for generated task arrivals.",
    "Timestamp": "Arrival time in seconds from the start of the shift horizon.",
    "Interarrival_Time": "Elapsed seconds since the previous generated task arrival.",
    "Regime": "Demand regime label at the task timestamp: base or morning_rush.",
    "Task_Type": "Optional sampled task category for downstream task modeling.",
    "Source_Zone": "Optional sampled origin zone for the task.",
    "Destination_Zone": "Optional sampled destination zone for the task.",
    "Priority": "Optional sampled integer priority for dispatching experiments.",
    "Service_Duration": "Optional sampled service duration estimate in seconds.",
}


@dataclass(frozen=True)
class TaskDemandRecord:
    """Single generated warehouse task-arrival record."""

    task_id: int
    timestamp: float
    interarrival_time: float
    regime: RegimeLabel
    task_type: str | None = None
    source_zone: str | None = None
    destination_zone: str | None = None
    priority: int | None = None
    service_duration: float | None = None

    def to_csv_dict(self, include_metadata: bool = False) -> dict[str, int | float | str]:
        """Convert the record into the explicit CSV schema used by the generator."""

        row: dict[str, int | float | str] = {
            "Task_ID": self.task_id,
            "Timestamp": round(self.timestamp, 3),
            "Interarrival_Time": round(self.interarrival_time, 3),
            "Regime": self.regime,
        }
        if include_metadata:
            row.update(
                {
                    "Task_Type": self.task_type or "",
                    "Source_Zone": self.source_zone or "",
                    "Destination_Zone": self.destination_zone or "",
                    "Priority": "" if self.priority is None else self.priority,
                    "Service_Duration": (
                        "" if self.service_duration is None else round(self.service_duration, 3)
                    ),
                }
            )
        return row


@dataclass(frozen=True)
class DemandSummary:
    """Summary statistics for a single demand-generation run."""

    tasks_generated: int
    shift_horizon_seconds: float
    observed_mean_interarrival: float | None
    interarrival_p95: float | None


@dataclass(frozen=True)
class DemandGenerationResult:
    """Generated demand records plus summary statistics."""

    records: tuple[TaskDemandRecord, ...]
    summary: DemandSummary


def csv_columns(include_metadata: bool = False) -> tuple[str, ...]:
    """Return the CSV header for legacy-only or enriched demand outputs."""

    if include_metadata:
        return LEGACY_CSV_COLUMNS + METADATA_CSV_COLUMNS
    return LEGACY_CSV_COLUMNS

