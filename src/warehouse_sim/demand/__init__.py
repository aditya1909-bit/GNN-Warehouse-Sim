"""Demand-generation APIs for stochastic warehouse task arrivals."""

from warehouse_sim.demand.config import (
    DemandGenerationConfig,
    DemandGenerationError,
    DemandValidationError,
    TaskMetadataConfig,
)
from warehouse_sim.demand.generator import (
    build_task_demand_records,
    generate_event_times,
    generate_task_demand,
    rate_at_time,
    regime_at_time,
    write_task_demand_csv,
)
from warehouse_sim.demand.models import (
    LEGACY_CSV_COLUMNS,
    METADATA_CSV_COLUMNS,
    DemandGenerationResult,
    DemandSummary,
    TaskDemandRecord,
)

__all__ = [
    "DemandGenerationConfig",
    "DemandGenerationError",
    "DemandGenerationResult",
    "DemandSummary",
    "DemandValidationError",
    "LEGACY_CSV_COLUMNS",
    "METADATA_CSV_COLUMNS",
    "TaskDemandRecord",
    "TaskMetadataConfig",
    "build_task_demand_records",
    "generate_event_times",
    "generate_task_demand",
    "rate_at_time",
    "regime_at_time",
    "write_task_demand_csv",
]

