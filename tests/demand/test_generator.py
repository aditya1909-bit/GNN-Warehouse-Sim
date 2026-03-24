"""Tests for stage-1 demand generation behavior."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from warehouse_sim.demand import (
    DemandGenerationConfig,
    DemandValidationError,
    TaskMetadataConfig,
    generate_task_demand,
    rate_at_time,
    regime_at_time,
    write_task_demand_csv,
)
from warehouse_sim.demand.models import LEGACY_CSV_COLUMNS, METADATA_CSV_COLUMNS


def test_default_generation_is_reproducible() -> None:
    config = DemandGenerationConfig()

    first = generate_task_demand(config)
    second = generate_task_demand(config)

    assert len(first.records) == 3225
    assert len(second.records) == 3225
    assert first.records == second.records
    assert first.summary.tasks_generated == 3225
    assert first.records[0].timestamp == pytest.approx(6.3803895658758645, rel=1e-12)
    assert first.summary.observed_mean_interarrival == pytest.approx(
        8.925079352519756,
        rel=1e-12,
    )


def test_validation_rejects_overlapping_windows() -> None:
    with pytest.raises(DemandValidationError, match="overlap"):
        DemandGenerationConfig(
            rush_start=1_000.0,
            rush_end=2_000.0,
            lunch_start=1_500.0,
            lunch_end=2_100.0,
        )


@pytest.mark.parametrize(
    ("timestamp", "expected_regime"),
    [
        (0.0, "base"),
        (1_800.0, "morning_rush"),
        (7_200.0, "base"),
        (14_400.0, "lunch"),
        (16_200.0, "base"),
    ],
)
def test_regime_boundaries(timestamp: float, expected_regime: str) -> None:
    config = DemandGenerationConfig()
    assert regime_at_time(timestamp, config) == expected_regime


def test_rush_window_increases_rate() -> None:
    config = DemandGenerationConfig(mean_interval=10.0, rush_multiplier=3.0)

    assert rate_at_time(100.0, config) == pytest.approx(0.1)
    assert rate_at_time(2_000.0, config) == pytest.approx(0.3)


def test_lunch_window_has_zero_arrivals() -> None:
    config = DemandGenerationConfig()
    result = generate_task_demand(config)

    assert not any(14_400.0 <= record.timestamp < 16_200.0 for record in result.records)


def test_legacy_csv_schema_and_order(tmp_path: Path) -> None:
    output_path = tmp_path / "task_demand.csv"
    result = generate_task_demand(DemandGenerationConfig())

    write_task_demand_csv(output_path=output_path, records=result.records, include_metadata=False)

    with output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    assert rows[0] == list(LEGACY_CSV_COLUMNS)
    assert rows[1] == ["1", "6.38", "6.38", "base"]


def test_optional_metadata_columns_are_appended(tmp_path: Path) -> None:
    output_path = tmp_path / "task_demand_with_metadata.csv"
    config = DemandGenerationConfig(min_tasks=0)
    metadata_config = TaskMetadataConfig(
        task_types=("pick",),
        source_zones=("storage_a",),
        destination_zones=("pick_station_1",),
        priorities=(2,),
        service_duration_low=45.0,
        service_duration_high=45.0,
    )

    result = generate_task_demand(config=config, metadata_config=metadata_config)
    write_task_demand_csv(output_path=output_path, records=result.records, include_metadata=True)

    with output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == list(LEGACY_CSV_COLUMNS + METADATA_CSV_COLUMNS)
    assert rows[0]["Task_Type"] == "pick"
    assert rows[0]["Source_Zone"] == "storage_a"
    assert rows[0]["Destination_Zone"] == "pick_station_1"
    assert rows[0]["Priority"] == "2"
    assert rows[0]["Service_Duration"] == "45.0"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon_seconds": 0.0},
        {"mean_interval": 0.0},
        {"rush_multiplier": 0.0},
        {"min_tasks": -1},
        {"rush_start": 100.0, "rush_end": 50.0},
        {"lunch_start": 30_000.0, "lunch_end": 30_000.0},
    ],
)
def test_invalid_parameters_raise(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(DemandValidationError):
        DemandGenerationConfig(**kwargs)
