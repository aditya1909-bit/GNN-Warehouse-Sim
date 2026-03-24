# Experiments

## Scope

Experiments remain TOML-driven and backward compatible. Stage 9 extends the existing schema rather than replacing it.

## Config Sections

- `layout`
- `demand`
- `robots`
- `tasks`
- `simulation`
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
