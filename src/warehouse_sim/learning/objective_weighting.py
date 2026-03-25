"""Deployment-aware weighting helpers for dispatch learning."""

from __future__ import annotations

from math import log1p
from typing import Mapping


def benchmark_weight_from_row(row: Mapping[str, object]) -> float:
    """Return a moderate sample weight for congestion-critical dispatch states."""

    ready_task_count = _float_value(row.get("ready_task_count"))
    mean_ready_task_age = _float_value(row.get("mean_ready_task_age"))
    estimated_delay = _float_value(row.get("estimated_pickup_congestion_delay")) + _float_value(
        row.get("estimated_dropoff_congestion_delay")
    )
    blocked_segments = _float_value(row.get("estimated_pickup_blocked_segments")) + _float_value(
        row.get("estimated_dropoff_blocked_segments")
    )
    reserved_resources = _float_value(row.get("active_reserved_edge_count")) + _float_value(
        row.get("active_reserved_node_count")
    )

    queue_pressure = 0.35 * log1p(max(ready_task_count, 0.0))
    age_pressure = 0.25 * log1p(max(mean_ready_task_age, 0.0))
    congestion_pressure = 0.8 * log1p(max(estimated_delay, 0.0))
    blocked_pressure = 0.4 * max(blocked_segments, 0.0)
    reservation_pressure = 0.1 * log1p(max(reserved_resources, 0.0))
    return 1.0 + queue_pressure + age_pressure + congestion_pressure + blocked_pressure + reservation_pressure


def _float_value(value: object | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)
