# Policy Models

## Purpose

This stage adds the first honest observation-driven policy integration. It is
not a GNN. It is a configurable linear scoring policy that consumes the same
candidate features exposed by the live dispatch context and exported observation
datasets.

## Implemented Now

- `linear_assignment_model` dispatch policy
- Named scalar feature weights loaded from TOML config
- Candidate robot-task observation builder shared between:
  - live dispatch scoring
  - dispatch trace export
- Validation of unsupported feature names

## Supported Feature Weights

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

## Why This Counts As A Real Integration

- The policy runs inside the real simulator.
- It scores the same candidate assignments that appear in exported datasets.
- Its behavior is fully specified by explicit model weights rather than hidden
  heuristics in ad hoc code.
- It establishes a stable bridge between simulator state and future learned
  policies without claiming any neural component exists yet.

## Not Implemented Yet

- Weight fitting or training
- Neural models
- Offline imitation or reinforcement-learning pipelines
- Model checkpoint formats beyond TOML-configured scalar weights
