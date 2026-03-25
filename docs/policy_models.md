# Policy Models

## Purpose

The repository now contains two policy families:

- dispatch policies over candidate robot-task assignments
- integrated coordination policies over task-plus-route macros

## Implemented Now

- baseline heuristics: `fifo`, `random`, `nearest_robot_task`, `nearest_task_for_idle_robot`
- congestion-aware heuristic: `congestion_aware_nearest_robot_task`
- `linear_assignment_model` dispatch policy
- `trained_linear_model` artifact-backed dispatch policy
- `trained_mlp_model` artifact-backed dispatch policy
- `trained_graph_dispatch_model` artifact-backed graph dispatch policy
- `prioritized_sipp_coordinator` integrated non-learning coordinator
- `optimal_mapf_coordinator` integrated exact current-epoch routing coordinator
- `random_macro` integrated smoke baseline
- `trained_end_to_end_macro_ppo` integrated artifact-backed macro controller
- named scalar feature weights loaded from TOML
- offline grouped-softmax fitting for the linear scorer
- one-hidden-layer MLP baseline over the same candidate features
- PyG graph encoder with directed message passing and global graph pooling
- masked PPO fine-tuning over dispatch events
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
- The graph-conditioned scorer actually uses graph structure through message passing instead of only engineered candidate features.
- The integrated stack adds a separate centralized policy boundary for joint task-plus-route macro decisions.
- The exact integrated MAPF baseline is still bounded to the current replan epoch, which keeps the optimality claim honest.

## Artifact-Backed Policies

Experiment configs can now point to a saved artifact:

```toml
[simulation]
policy = "trained_graph_dispatch_model"

[policy_model]
artifact_path = "artifacts/models/graph_dispatch_model.json"
```

The dispatch artifacts still output candidate scores within one dispatch event and then rank them.

The integrated artifact chooses task-plus-route macros across robots at replanning boundaries. It is the repository's experimental end-to-end coordination controller, but stronger claims remain benchmark-gated.

## Not Implemented

- richer graph readouts such as endpoint-conditioned pooling
- broader RL algorithms beyond the current masked PPO fine-tuning loop
- stronger learned-coordination claims without passing the benchmark gate
