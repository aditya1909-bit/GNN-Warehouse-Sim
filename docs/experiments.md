# Experiments

## Scope

Experiments remain TOML-driven and backward compatible. Stage 12 adds an integrated coordination mode beside the dispatch-centric stack.

## Config Sections

- `layout`
- `demand`
- `robots`
- `tasks`
- `simulation`
- `coordination` when `simulation.coordination_mode = "integrated"`
- `policy_model` when a policy needs weights or an artifact
- `reporting`

## New Simulation Setting

`[simulation]` now supports:

- `coordination_mode`
- `policy`
- `horizon_seconds`
- `continue_until_all_tasks_complete`
- `execution_model`

Example:

```toml
[simulation]
coordination_mode = "dispatch"
policy = "congestion_aware_nearest_robot_task"
execution_model = "reserved_edges"
```

If `coordination_mode` is omitted, the default remains `dispatch`. In dispatch mode, `execution_model` keeps its previous meaning.

Integrated mode example:

```toml
[simulation]
coordination_mode = "integrated"
policy = "prioritized_sipp_coordinator"
execution_model = "idealized"

[coordination]
control_dt = 0.25
replan_period = 1.0
robot_radius = 0.2
collision_clearance = 0.05
k_shortest_paths = 3
max_route_options_per_pair = 3
```

## Trained Artifact Policies

The experiment layer now supports both hand-configured and trained candidate scorers:

- `linear_assignment_model`: hand-configured named weights
- `trained_linear_model`: load a fitted linear artifact
- `trained_mlp_model`: load a fitted MLP artifact
- `trained_graph_dispatch_model`: load a PyG graph-dispatch artifact
- `optimal_mapf_coordinator`: run the integrated exact current-epoch routing baseline
- `trained_end_to_end_macro_ppo`: load an integrated macro PPO artifact

Example:

```toml
[simulation]
policy = "trained_graph_dispatch_model"
execution_model = "reserved_edges"

[policy_model]
artifact_path = "artifacts/models/graph_dispatch_model.json"
```

## Outputs

A dispatch-mode run still writes:

- `summary.json`
- `executions.csv`
- `queue_snapshots.csv`
- `robot_metrics.csv`

Integrated runs add:

- `robot_trajectories.csv`
- `macro_decisions.csv`
- `collision_events.csv`
- `planner_plans.csv`

## Practical Use

- Use `idealized` when you want continuity with the original baseline.
- Use `reserved_edges` for corridor and aisle-sharing stress tests.
- Use `reserved_nodes` for station-like occupancy and chokepoint experiments.
- Use `coordination_mode = "integrated"` when you want centralized timed planning and continuous collision-aware execution.
