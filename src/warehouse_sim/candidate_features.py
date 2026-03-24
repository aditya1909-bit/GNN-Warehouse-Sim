"""Shared candidate-feature contract for live and offline dispatch scoring."""

from __future__ import annotations

SUPPORTED_CANDIDATE_FEATURES = (
    "travel_to_pickup_time",
    "travel_to_pickup_distance",
    "pickup_to_dropoff_time",
    "pickup_to_dropoff_distance",
    "task_age",
    "task_priority",
    "task_service_time_estimate",
    "robot_speed_multiplier",
    "robot_completed_task_count",
    "robot_total_busy_time",
    "robot_total_idle_time",
    "robot_total_travel_time",
    "robot_total_travel_distance",
    "pending_task_count",
    "ready_task_count",
    "future_task_count",
    "idle_robot_count",
    "busy_robot_count",
    "mean_ready_task_age",
    "average_robot_time_until_available",
    "active_reserved_edge_count",
    "active_reserved_node_count",
    "estimated_pickup_congestion_delay",
    "estimated_dropoff_congestion_delay",
    "estimated_pickup_blocked_segments",
    "estimated_dropoff_blocked_segments",
)
