# Experiments

## Stage 4 Scope

Stage 4 makes the baseline simulation reproducible through explicit experiment configuration and machine-readable outputs.

- TOML experiment configs under `configs/`
- a config loader and runner in package code
- report writers for summary JSON and CSV artifacts
- first-class plotting support using matplotlib

## Config Format

The baseline experiment schema is TOML-based and organized into sections:

- `layout`
- `demand`
- `robots`
- `tasks`
- `simulation`
- `reporting`

The default preset is `configs/baseline_experiment.toml`.

## Outputs

A config-driven run writes:

- `summary.json`
- `executions.csv`
- `queue_snapshots.csv`
- `robot_metrics.csv`

If plotting is enabled, it also writes:

- `queue_length.png`
- `robot_utilization.png`

## Why TOML

TOML keeps the config layer dependency-light because Python 3.11 already ships with `tomllib`. That fits the project’s current goal of being reproducible and structured without pulling in a larger config framework too early.
