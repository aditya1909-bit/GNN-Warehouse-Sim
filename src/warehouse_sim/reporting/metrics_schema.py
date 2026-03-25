"""Stable metric schema for benchmark, planner, and learning outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


METRIC_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class MetricDefinition:
    """Definition for one canonical metric."""

    name: str
    category: str
    direction: str
    description: str
    unit: str


METRIC_DEFINITIONS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        name="throughput",
        category="primary",
        direction="maximize",
        description="Completed tasks per hour over the realized makespan.",
        unit="tasks_per_hour",
    ),
    MetricDefinition(
        name="mean_task_completion_time",
        category="primary",
        direction="minimize",
        description="Mean task turnaround time from release to completion.",
        unit="seconds",
    ),
    MetricDefinition(
        name="p95_task_completion_time",
        category="primary",
        direction="minimize",
        description="95th percentile task turnaround time from release to completion.",
        unit="seconds",
    ),
    MetricDefinition(
        name="makespan",
        category="primary",
        direction="minimize",
        description="Wall-clock span from simulation start until all realized work completes.",
        unit="seconds",
    ),
    MetricDefinition(
        name="mean_queue_length",
        category="primary",
        direction="minimize",
        description="Time-weighted mean number of ready tasks waiting for assignment.",
        unit="tasks",
    ),
    MetricDefinition(
        name="p95_queue_length",
        category="primary",
        direction="minimize",
        description="95th percentile queue length over recorded event snapshots.",
        unit="tasks",
    ),
    MetricDefinition(
        name="robot_idle_fraction",
        category="primary",
        direction="minimize",
        description="Fleet-level fraction of robot time spent idle over the realized makespan.",
        unit="fraction",
    ),
    MetricDefinition(
        name="travel_distance_per_completed_task",
        category="primary",
        direction="minimize",
        description="Mean realized robot travel distance per completed task.",
        unit="distance_units_per_task",
    ),
    MetricDefinition(
        name="realized_waiting_time",
        category="primary",
        direction="minimize",
        description="Total realized waiting inserted by congestion or reservations during execution.",
        unit="seconds",
    ),
    MetricDefinition(
        name="congestion_event_count",
        category="primary",
        direction="minimize",
        description="Total realized blocked-traversal or congestion-induced wait events.",
        unit="count",
    ),
    MetricDefinition(
        name="collision_event_count",
        category="primary",
        direction="minimize",
        description="Total explicit collision or safety-violation events.",
        unit="count",
    ),
    MetricDefinition(
        name="deadlock_livelock_incident_count",
        category="primary",
        direction="minimize",
        description="Count of explicit deadlock or livelock incidents detected during a run.",
        unit="count",
    ),
    MetricDefinition(
        name="planning_latency",
        category="planner",
        direction="minimize",
        description="Mean wall-clock planner latency per planning call.",
        unit="seconds",
    ),
    MetricDefinition(
        name="replanning_count",
        category="planner",
        direction="minimize",
        description="Number of distinct replanning epochs encountered in the run.",
        unit="count",
    ),
    MetricDefinition(
        name="planner_failure_count",
        category="planner",
        direction="minimize",
        description="Number of planner calls that failed to produce a feasible plan.",
        unit="count",
    ),
    MetricDefinition(
        name="timeout_count",
        category="planner",
        direction="minimize",
        description="Number of planner calls that timed out.",
        unit="count",
    ),
    MetricDefinition(
        name="path_conflict_count_before_resolution",
        category="planner",
        direction="minimize",
        description="Count of path conflicts detected before planner conflict resolution.",
        unit="count",
    ),
    MetricDefinition(
        name="sipp_wait_insertion_count",
        category="planner",
        direction="minimize",
        description="Count of inserted SIPP wait actions attributable to collision avoidance.",
        unit="count",
    ),
    MetricDefinition(
        name="mapf_solve_success_rate",
        category="planner",
        direction="maximize",
        description="Fraction of planner calls that successfully produced a feasible joint plan.",
        unit="fraction",
    ),
    MetricDefinition(
        name="reward_mean",
        category="learning",
        direction="maximize",
        description="Mean episodic reward for a learning run.",
        unit="reward",
    ),
    MetricDefinition(
        name="reward_std",
        category="learning",
        direction="minimize",
        description="Standard deviation of episodic reward across seeds or episodes.",
        unit="reward",
    ),
    MetricDefinition(
        name="policy_entropy",
        category="learning",
        direction="maximize",
        description="Mean policy entropy during training or evaluation.",
        unit="nats",
    ),
    MetricDefinition(
        name="invalid_action_rate",
        category="learning",
        direction="minimize",
        description="Fraction of emitted actions that are invalid before masking or rejection.",
        unit="fraction",
    ),
    MetricDefinition(
        name="masked_action_rejection_rate",
        category="learning",
        direction="minimize",
        description="Fraction of proposed actions rejected by action masking.",
        unit="fraction",
    ),
    MetricDefinition(
        name="ppo_kl",
        category="learning",
        direction="minimize",
        description="Mean PPO KL divergence between old and updated policies.",
        unit="nats",
    ),
    MetricDefinition(
        name="ppo_clip_fraction",
        category="learning",
        direction="minimize",
        description="Fraction of PPO updates clipped by the trust region.",
        unit="fraction",
    ),
    MetricDefinition(
        name="value_loss",
        category="learning",
        direction="minimize",
        description="Mean critic loss for PPO or actor-critic training.",
        unit="loss",
    ),
    MetricDefinition(
        name="generalization_gap_seen_vs_unseen",
        category="learning",
        direction="minimize",
        description="Absolute performance gap between seen and unseen evaluation settings.",
        unit="metric_delta",
    ),
)

METRIC_DEFINITIONS_BY_NAME = {definition.name: definition for definition in METRIC_DEFINITIONS}
METRIC_NAMES = tuple(definition.name for definition in METRIC_DEFINITIONS)

BENCHMARK_RUN_METADATA_FIELDS: tuple[str, ...] = (
    "metric_schema_version",
    "benchmark_name",
    "scenario_family",
    "scenario_id",
    "scenario_name",
    "scenario_config",
    "seed",
    "policy",
    "policy_family",
    "policy_role",
    "coordination_mode",
    "execution_model",
    "motion_model",
    "fleet_size",
    "demand_mean_interval",
    "demand_horizon_seconds",
    "layout_rows",
    "layout_columns",
    "blocked_cell_count",
    "directed_edge_count",
    "topology_difficulty",
    "summary_path",
)

BENCHMARK_AGGREGATE_METADATA_FIELDS: tuple[str, ...] = (
    "metric_schema_version",
    "benchmark_name",
    "scenario_family",
    "scenario_id",
    "scenario_name",
    "policy",
    "policy_family",
    "policy_role",
    "coordination_mode",
    "execution_model",
    "motion_model",
    "fleet_size",
    "demand_mean_interval",
    "demand_horizon_seconds",
    "layout_rows",
    "layout_columns",
    "blocked_cell_count",
    "directed_edge_count",
    "topology_difficulty",
    "run_count",
    "seeds",
)

BENCHMARK_CLAIM_FIELDS: tuple[str, ...] = (
    "metric_schema_version",
    "benchmark_name",
    "scenario_family",
    "scenario_id",
    "scenario_name",
    "baseline_policy",
    "challenger_policy",
    "baseline_policy_family",
    "challenger_policy_family",
    "comparison_type",
    "primary_metric",
    "winner",
    "baseline_mean",
    "challenger_mean",
    "uplift_absolute",
    "uplift_percent",
    "improvement_ci95_low",
    "improvement_ci95_high",
    "artifact_path",
    "claim_text",
)


def default_metric_payload() -> dict[str, float | int | None]:
    """Return a metric payload populated with schema keys and null defaults."""

    return {name: None for name in METRIC_NAMES}


def ordered_run_fields(extra_fields: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Return benchmark run field order with optional non-schema extras appended."""

    return (*BENCHMARK_RUN_METADATA_FIELDS, *METRIC_NAMES, *extra_fields)


def ordered_aggregate_fields(extra_fields: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Return benchmark aggregate field order with metric statistics expanded."""

    fields = list(BENCHMARK_AGGREGATE_METADATA_FIELDS)
    for metric_name in METRIC_NAMES:
        fields.extend(
            (
                f"{metric_name}_mean",
                f"{metric_name}_std",
                f"{metric_name}_ci95_low",
                f"{metric_name}_ci95_high",
            )
        )
    fields.extend(extra_fields)
    return tuple(fields)


def validate_benchmark_run_row(row: Mapping[str, object]) -> None:
    """Validate that a benchmark run row contains the canonical schema."""

    _validate_required_fields(row, BENCHMARK_RUN_METADATA_FIELDS, METRIC_NAMES)


def validate_benchmark_aggregate_row(row: Mapping[str, object]) -> None:
    """Validate that a benchmark aggregate row contains the canonical schema."""

    required = list(BENCHMARK_AGGREGATE_METADATA_FIELDS)
    for metric_name in METRIC_NAMES:
        required.extend(
            (
                f"{metric_name}_mean",
                f"{metric_name}_std",
                f"{metric_name}_ci95_low",
                f"{metric_name}_ci95_high",
            )
        )
    _validate_required_fields(row, tuple(required), ())


def validate_benchmark_claim_row(row: Mapping[str, object]) -> None:
    """Validate that a benchmark claim row contains the canonical schema."""

    _validate_required_fields(row, BENCHMARK_CLAIM_FIELDS, ())


def _validate_required_fields(
    row: Mapping[str, object],
    required_fields: tuple[str, ...],
    metric_fields: tuple[str, ...],
) -> None:
    missing = [field for field in required_fields if field not in row]
    if missing:
        raise ValueError(f"Row is missing required fields: {missing}")
    for metric_name in metric_fields:
        value = row[metric_name]
        if value is not None and not isinstance(value, int | float):
            raise ValueError(f"Metric {metric_name!r} must be numeric or null, got {type(value).__name__}.")
