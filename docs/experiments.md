# Experiments

## Scope

Experiments remain TOML-driven and backward compatible. Stage 10 extends the existing schema rather than replacing it.

## Config Sections

- `layout`
- `demand`
- `robots`
- `tasks`
- `simulation`
- `policy_model` when a policy needs weights or an artifact
- `reporting`

## New Simulation Setting

`[simulation]` now supports:

- `policy`
- `horizon_seconds`
- `continue_until_all_tasks_complete`
- `execution_model`

Example:

```toml
[simulation]
policy = "congestion_aware_nearest_robot_task"
execution_model = "reserved_edges"
```

If `execution_model` is omitted, the default remains `idealized`.

## Trained Artifact Policies

The experiment layer now supports both hand-configured and trained candidate scorers:

- `linear_assignment_model`: hand-configured named weights
- `trained_linear_model`: load a fitted linear artifact
- `trained_mlp_model`: load a fitted MLP artifact

Example:

```toml
[simulation]
policy = "trained_linear_model"
execution_model = "reserved_edges"

[policy_model]
artifact_path = "artifacts/models/linear_dispatch_model.json"
```

## Outputs

A config-driven run still writes:

- `summary.json`
- `executions.csv`
- `queue_snapshots.csv`
- `robot_metrics.csv`

Stage 9 extends those files with congestion-aware fields rather than introducing a second report format.

## Practical Use

- Use `idealized` when you want continuity with the original baseline.
- Use `reserved_edges` for corridor and aisle-sharing stress tests.
- Use `reserved_nodes` for station-like occupancy and chokepoint experiments.
