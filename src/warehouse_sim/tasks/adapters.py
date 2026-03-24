"""Adapters that turn generated demand rows into explicit task objects."""

from __future__ import annotations

from dataclasses import dataclass

from warehouse_sim.demand.models import TaskDemandRecord
from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.tasks.models import Task, TaskValidationError


class TaskAdapterError(ValueError):
    """Raised when demand records cannot be mapped into task objects."""


@dataclass(frozen=True)
class DemandTaskAdapterConfig:
    """Rules for converting demand records into warehouse tasks."""

    default_pickup_zone: str
    default_dropoff_zone: str
    default_task_type: str = "pick"
    default_priority: int = 1
    default_service_time_estimate: float = 60.0
    task_id_prefix: str = "task"

    def __post_init__(self) -> None:
        if not self.default_pickup_zone:
            raise TaskAdapterError("default_pickup_zone must be non-empty.")
        if not self.default_dropoff_zone:
            raise TaskAdapterError("default_dropoff_zone must be non-empty.")
        if not self.default_task_type:
            raise TaskAdapterError("default_task_type must be non-empty.")
        if self.default_priority <= 0:
            raise TaskAdapterError("default_priority must be > 0.")
        if self.default_service_time_estimate < 0:
            raise TaskAdapterError("default_service_time_estimate must be >= 0.")
        if not self.task_id_prefix:
            raise TaskAdapterError("task_id_prefix must be non-empty.")


def tasks_from_demand_records(
    records: tuple[TaskDemandRecord, ...],
    environment: WarehouseEnvironment,
    config: DemandTaskAdapterConfig,
) -> tuple[Task, ...]:
    """Convert stage-1 demand records into explicit task objects."""

    tasks: list[Task] = []
    for record in records:
        source_zone = record.source_zone or config.default_pickup_zone
        destination_zone = record.destination_zone or config.default_dropoff_zone

        try:
            pickup_node = environment.default_node_for_zone(source_zone).node_id
            dropoff_node = environment.default_node_for_zone(destination_zone).node_id
        except Exception as exc:
            raise TaskAdapterError(
                f"Failed to resolve zones for demand record {record.task_id}: "
                f"{source_zone} -> {destination_zone}"
            ) from exc

        try:
            tasks.append(
                Task(
                    task_id=f"{config.task_id_prefix}_{record.task_id}",
                    release_time=record.timestamp,
                    pickup_node=pickup_node,
                    dropoff_node=dropoff_node,
                    task_type=record.task_type or config.default_task_type,
                    priority=record.priority or config.default_priority,
                    service_time_estimate=(
                        record.service_duration
                        if record.service_duration is not None
                        else config.default_service_time_estimate
                    ),
                    source_zone=source_zone,
                    destination_zone=destination_zone,
                    metadata={"regime": record.regime},
                )
            )
        except TaskValidationError as exc:
            raise TaskAdapterError(
                f"Invalid task derived from demand record {record.task_id}."
            ) from exc
    return tuple(tasks)
