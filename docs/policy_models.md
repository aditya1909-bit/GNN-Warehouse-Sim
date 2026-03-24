# Policy Models

## Purpose

The repository still contains simple, honest dispatch policies. There is no neural model here. The most flexible policy remains a configurable linear scorer over exported/live candidate features.

## Implemented Now

- baseline heuristics: `fifo`, `random`, `nearest_robot_task`, `nearest_task_for_idle_robot`
- congestion-aware heuristic: `congestion_aware_nearest_robot_task`
- `linear_assignment_model` dispatch policy
- named scalar feature weights loaded from TOML
- validation of unsupported feature names

## Supported Linear Features

- `travel_to_pickup_time`
- `travel_to_pickup_distance`
- `pickup_to_dropoff_time`
- `pickup_to_dropoff_distance`
- `task_age`
- `task_priority`
- `task_service_time_estimate`
- `robot_speed_multiplier`
- `robot_completed_task_count`
- `robot_total_busy_time`
- `robot_total_idle_time`
- `robot_total_travel_time`
- `robot_total_travel_distance`
- `pending_task_count`
- `ready_task_count`
- `future_task_count`
- `idle_robot_count`
- `busy_robot_count`
- `mean_ready_task_age`
- `average_robot_time_until_available`
- `active_reserved_edge_count`
- `active_reserved_node_count`
- `estimated_pickup_congestion_delay`
- `estimated_dropoff_congestion_delay`
- `estimated_pickup_blocked_segments`
- `estimated_dropoff_blocked_segments`

## Why This Still Matters

- The policies run inside the real simulator.
- The linear scorer and dataset export use the same candidate feature contract.
- The congestion-aware heuristic gives a stronger non-learning baseline without pretending to be learned.

## Not Implemented Yet

- weight fitting or training
- neural models
- offline imitation or reinforcement-learning pipelines
