"""Task-domain models for warehouse experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


class TaskValidationError(ValueError):
    """Raised when a task definition is invalid."""


@dataclass(frozen=True)
class Task:
    """Immutable warehouse task specification."""

    task_id: str
    release_time: float
    pickup_node: str
    dropoff_node: str
    task_type: str = "pick"
    priority: int = 1
    service_time_estimate: float = 60.0
    due_time: float | None = None
    source_zone: str | None = None
    destination_zone: str | None = None
    metadata: dict[str, str | int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise TaskValidationError("task_id must be non-empty.")
        if self.release_time < 0:
            raise TaskValidationError("release_time must be >= 0.")
        if not self.pickup_node:
            raise TaskValidationError("pickup_node must be non-empty.")
        if not self.dropoff_node:
            raise TaskValidationError("dropoff_node must be non-empty.")
        if self.priority <= 0:
            raise TaskValidationError("priority must be > 0.")
        if self.service_time_estimate < 0:
            raise TaskValidationError("service_time_estimate must be >= 0.")
        if self.due_time is not None and self.due_time < self.release_time:
            raise TaskValidationError("due_time must be >= release_time.")

