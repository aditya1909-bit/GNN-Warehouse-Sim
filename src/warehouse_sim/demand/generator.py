"""Pure demand-generation utilities for stochastic warehouse task arrivals."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

from warehouse_sim.demand.config import (
    DemandGenerationConfig,
    DemandGenerationError,
    TaskMetadataConfig,
)
from warehouse_sim.demand.models import (
    DemandGenerationResult,
    DemandSummary,
    TaskDemandRecord,
    csv_columns,
)

logger = logging.getLogger(__name__)


def rate_at_time(timestamp: float, config: DemandGenerationConfig) -> float:
    """Return the instantaneous arrival rate at a timestamp."""

    base_rate = 1.0 / config.mean_interval
    if config.lunch_start <= timestamp < config.lunch_end:
        return 0.0

    rate = base_rate
    if config.rush_start <= timestamp < config.rush_end:
        rate *= config.rush_multiplier
    return rate


def regime_at_time(timestamp: float, config: DemandGenerationConfig) -> str:
    """Return the regime label active at a timestamp."""

    if config.lunch_start <= timestamp < config.lunch_end:
        return "lunch"
    if config.rush_start <= timestamp < config.rush_end:
        return "morning_rush"
    return "base"


def generate_event_times(
    config: DemandGenerationConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate task-arrival timestamps using thinning for an NHPP."""

    rng = rng or np.random.default_rng(config.seed)
    base_rate = 1.0 / config.mean_interval
    lambda_max = base_rate * max(1.0, config.rush_multiplier)

    current_time = 0.0
    event_times: list[float] = []

    while current_time < config.horizon_seconds:
        current_time += rng.exponential(1.0 / lambda_max)
        if current_time >= config.horizon_seconds:
            break

        if rng.random() <= rate_at_time(current_time, config) / lambda_max:
            event_times.append(float(current_time))

    return np.asarray(event_times, dtype=float)


def build_task_demand_records(
    event_times: np.ndarray,
    config: DemandGenerationConfig,
    metadata_config: TaskMetadataConfig | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[TaskDemandRecord, ...]:
    """Build typed task-arrival records from generated event times."""

    if event_times.size == 0:
        return ()

    interarrivals = np.diff(np.insert(event_times, 0, 0.0))
    rng = rng or np.random.default_rng(config.seed)

    records: list[TaskDemandRecord] = []
    for task_id, (timestamp, interarrival) in enumerate(
        zip(event_times, interarrivals),
        start=1,
    ):
        metadata = _sample_metadata(rng, metadata_config)
        records.append(
            TaskDemandRecord(
                task_id=task_id,
                timestamp=float(timestamp),
                interarrival_time=float(interarrival),
                regime=regime_at_time(float(timestamp), config),
                task_type=metadata["task_type"],
                source_zone=metadata["source_zone"],
                destination_zone=metadata["destination_zone"],
                priority=metadata["priority"],
                service_duration=metadata["service_duration"],
                due_time=(
                    None
                    if metadata["due_time"] is None
                    else float(timestamp) + float(metadata["due_time"])
                ),
            )
        )
    return tuple(records)


def generate_task_demand(
    config: DemandGenerationConfig,
    metadata_config: TaskMetadataConfig | None = None,
) -> DemandGenerationResult:
    """Generate demand records and summary statistics from a validated config."""

    rng = np.random.default_rng(config.seed)
    event_times = generate_event_times(config=config, rng=rng)

    if event_times.size < config.min_tasks:
        raise DemandGenerationError(
            f"Generated {event_times.size} tasks, below min_tasks={config.min_tasks}. "
            "Increase horizon_seconds or decrease mean_interval."
        )

    records = build_task_demand_records(
        event_times=event_times,
        config=config,
        metadata_config=metadata_config,
        rng=rng,
    )
    summary = _build_summary(records=records, config=config)

    logger.info(
        "Generated %s tasks over %.0f seconds with seed=%s.",
        summary.tasks_generated,
        summary.shift_horizon_seconds,
        config.seed,
    )
    return DemandGenerationResult(records=records, summary=summary)


def write_task_demand_csv(
    output_path: Path,
    records: tuple[TaskDemandRecord, ...],
    include_metadata: bool = False,
) -> None:
    """Write generated task-arrival records to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns(include_metadata))
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_csv_dict(include_metadata=include_metadata))


def _build_summary(
    records: tuple[TaskDemandRecord, ...],
    config: DemandGenerationConfig,
) -> DemandSummary:
    if not records:
        return DemandSummary(
            tasks_generated=0,
            shift_horizon_seconds=config.horizon_seconds,
            observed_mean_interarrival=None,
            interarrival_p95=None,
        )

    interarrivals = np.asarray([record.interarrival_time for record in records], dtype=float)
    return DemandSummary(
        tasks_generated=len(records),
        shift_horizon_seconds=config.horizon_seconds,
        observed_mean_interarrival=float(np.mean(interarrivals)),
        interarrival_p95=float(np.quantile(interarrivals, 0.95)),
    )


def _sample_metadata(
    rng: np.random.Generator,
    metadata_config: TaskMetadataConfig | None,
) -> dict[str, int | float | str | None]:
    if metadata_config is None:
        return {
            "task_type": None,
            "source_zone": None,
            "destination_zone": None,
            "priority": None,
            "service_duration": None,
            "due_time": None,
        }

    due_time = None
    if metadata_config.due_time_slacks:
        due_time = float(rng.choice(metadata_config.due_time_slacks))
    elif metadata_config.due_time_slack_low is not None and metadata_config.due_time_slack_high is not None:
        due_time = float(
            rng.uniform(
                metadata_config.due_time_slack_low,
                metadata_config.due_time_slack_high,
            )
        )
    return {
        "task_type": str(rng.choice(metadata_config.task_types)),
        "source_zone": str(rng.choice(metadata_config.source_zones)),
        "destination_zone": str(rng.choice(metadata_config.destination_zones)),
        "priority": int(rng.choice(metadata_config.priorities)),
        "service_duration": float(
            rng.uniform(
                metadata_config.service_duration_low,
                metadata_config.service_duration_high,
            )
        ),
        "due_time": due_time,
    }
