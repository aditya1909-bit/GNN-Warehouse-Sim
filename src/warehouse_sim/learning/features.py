"""Feature naming helpers for offline dispatch datasets."""

from __future__ import annotations

from warehouse_sim.candidate_features import SUPPORTED_CANDIDATE_FEATURES

LABEL_COLUMN = "is_selected"
DEFAULT_CANDIDATE_FEATURES = tuple(SUPPORTED_CANDIDATE_FEATURES)

REQUIRED_DISPATCH_COLUMNS = (
    "dispatch_index",
    "decision_time",
    "selected_robot_id",
    "selected_action_type",
    "selected_task_id",
    "selected_charging_node_id",
    "candidate_robot_id",
    "candidate_action_type",
    "candidate_task_id",
    "candidate_charging_node_id",
    LABEL_COLUMN,
)

METADATA_COLUMNS = (
    "dataset_id",
    "run_id",
    "scenario_name",
    "scenario_seed",
    "experiment_name",
    "source_policy_name",
    "demand_seed",
    "source_manifest_path",
    "benchmark_weight",
    "dispatch_group_id",
    "dispatch_index",
    "decision_time",
    "selected_robot_id",
    "selected_action_type",
    "selected_task_id",
    "selected_charging_node_id",
    "candidate_robot_id",
    "candidate_action_type",
    "candidate_task_id",
    "candidate_charging_node_id",
    "robot_current_node",
    "robot_current_zone",
    "robot_battery_level",
    "robot_battery_fraction",
    "robot_total_charging_time",
    "robot_total_energy_consumed",
    "robot_total_energy_charged",
    "task_release_time",
    "task_pickup_node",
    "task_dropoff_node",
    "task_source_zone",
    "task_destination_zone",
    "execution_model",
    "ready_task_count",
    "mean_ready_task_age",
    "active_reserved_edge_count",
    "active_reserved_node_count",
    "estimated_pickup_congestion_delay",
    "estimated_dropoff_congestion_delay",
    "estimated_pickup_blocked_segments",
    "estimated_dropoff_blocked_segments",
    "battery_fraction",
    "estimated_action_energy",
    "post_action_battery_fraction",
    "charger_reachable_after_action",
    "is_charge_action",
)

SUPPORTED_SPLIT_UNITS = ("dispatch_group", "run", "scenario", "scenario_seed")


def candidate_feature_names_from_columns(columns: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Return supported candidate features present in a dispatch dataset."""

    available = set(columns)
    return tuple(feature_name for feature_name in DEFAULT_CANDIDATE_FEATURES if feature_name in available)


def validate_candidate_feature_names(feature_names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Validate an ordered feature set against the live candidate contract."""

    names = tuple(feature_names)
    if not names:
        raise ValueError("At least one candidate feature name must be provided.")

    unsupported = [name for name in names if name not in DEFAULT_CANDIDATE_FEATURES]
    if unsupported:
        raise ValueError(f"Unsupported candidate features: {unsupported}")

    duplicates = [name for index, name in enumerate(names) if name in names[:index]]
    if duplicates:
        raise ValueError(f"Duplicate candidate features are not allowed: {duplicates}")
    return names
