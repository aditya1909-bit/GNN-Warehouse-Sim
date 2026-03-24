# Policy Models

## Purpose

The repository still contains simple, honest dispatch policies. Stage 10 adds offline-fitted candidate scorers, but the repository still does not train a GNN and still does not implement RL.

## Implemented Now

- baseline heuristics: `fifo`, `random`, `nearest_robot_task`, `nearest_task_for_idle_robot`
- congestion-aware heuristic: `congestion_aware_nearest_robot_task`
- `linear_assignment_model` dispatch policy
- `trained_linear_model` artifact-backed dispatch policy
- `trained_mlp_model` artifact-backed dispatch policy
- named scalar feature weights loaded from TOML
- offline grouped-softmax fitting for the linear scorer
- one-hidden-layer MLP baseline over the same candidate features
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
- The fitted linear scorer learns those weights from exported dispatch decisions instead of hard-coding them.
- The MLP baseline is still just a candidate-feature model, which keeps the comparison honest.
- The congestion-aware heuristic gives a stronger non-learning baseline without pretending to be learned.

## Artifact-Backed Policies

Experiment configs can now point to a saved artifact:

```toml
[simulation]
policy = "trained_linear_model"

[policy_model]
artifact_path = "artifacts/models/linear_dispatch_model.json"
```

The trained artifact still scores candidate robot-task pairs independently within a dispatch event and then ranks them. That is not the same thing as a graph policy, MAPF solver, or end-to-end learned coordination controller.

## Not Implemented

- true graph-neural dispatch models
- RL
- richer imitation-learning objectives beyond the current grouped candidate-selection fitting
