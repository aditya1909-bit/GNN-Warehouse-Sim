# GNN-Warehouse-Sim

Research-grade warehouse simulation framework scaffold with validated demand, graph/task, simulation, experiment, benchmark, and policy-observation foundations.

## Current Status

This repository is not yet a full warehouse simulator.

Implemented and tested now:

- Synthetic warehouse task-demand generation as a non-homogeneous Poisson process
- Base exponential interarrival model
- Morning rush rate multiplier
- Lunch zero-arrival shutdown window
- Optional task metadata sampling for downstream task-model work
- Importable package code under `src/warehouse_sim/demand/`
- Synthetic grid warehouse topology with weighted shortest-path utilities
- Warehouse environment and named zone abstractions
- Explicit task objects, release-time-aware task queues, and demand-to-task adapters
- Robot specifications and discrete-event simulation runtime state
- Baseline dispatch policies: FIFO, random, nearest-robot-task, nearest-task-for-idle-robot
- Run-level metrics for waiting time, turnaround time, throughput, queue length, and utilization
- TOML experiment configs and config-driven experiment execution
- Machine-readable reporting outputs: summary JSON plus execution, queue, and robot CSVs
- Scenario presets and benchmark comparison manifests
- GNN-preparation interfaces for graph featurization and dispatch-time observations
- Observation-dataset export for future learned-policy experiments
- Observation-driven linear scoring policy loaded from experiment config
- Pytest coverage for demand, graph, environment, task, policy, simulation, and benchmark layers

Planned next, but not implemented yet:

- Richer learned dispatch policies beyond the current linear scorer

## Install

For package-based development and tests:

```bash
python3 -m pip install -e .[dev]
```

This installs the standard runtime, including `matplotlib` for report plots.

The legacy script path also works directly from the repository root without installation, but config-driven plotting support assumes the package dependencies are installed.

## Generate Synthetic Task Demand

Legacy-compatible script:

```bash
python3 scripts/generate_task_demand.py
```

Package CLI:

```bash
PYTHONPATH=src python3 -m warehouse_sim.demand.cli
```

Default output: `data/task_demand.csv`

Key behavior:

- Base arrivals follow an exponential interarrival process.
- Morning rush increases the arrival rate.
- Lunch break enforces a zero-arrival period.
- Randomness is reproducible through an explicit seed.

Useful options:

```bash
python3 scripts/generate_task_demand.py \
  --horizon-seconds 28800 \
  --mean-interval 10 \
  --rush-start 1800 --rush-end 7200 --rush-multiplier 2.0 \
  --lunch-start 14400 --lunch-end 16200 \
  --seed 7
```

To append richer task metadata columns:

```bash
python3 scripts/generate_task_demand.py \
  --include-task-metadata \
  --task-types pick replenishment cycle_count \
  --source-zones storage_a storage_b inbound \
  --destination-zones pick_station_1 pick_station_2 staging \
  --priorities 1 2 3
```

## Output Schema

Default CSV columns:

- `Task_ID`: 1-based sequential task identifier
- `Timestamp`: arrival timestamp in seconds from the start of the horizon
- `Interarrival_Time`: elapsed seconds since the previous task arrival
- `Regime`: demand regime label at the timestamp (`base` or `morning_rush` in generated output)

Optional appended columns when `--include-task-metadata` is enabled:

- `Task_Type`
- `Source_Zone`
- `Destination_Zone`
- `Priority`
- `Service_Duration`

## Analyze The Demand Model

Open `notebooks/input_modeling_analysis.ipynb` and run all cells.

The notebook remains the downstream analysis workspace for:

- Data card and context for interarrival modeling
- EDA on homogeneous base-regime intervals
- Exponential, Gamma, and Weibull fitting
- Goodness-of-fit metrics and diagnostics
- Automated `distfit` cross-checks when available

Core generator behavior now lives in package code rather than in notebooks.

## Documentation

- [Repository audit](docs/repo_audit.md)
- [Architecture overview](docs/architecture.md)
- [Domain model](docs/domain_model.md)
- [Simulation baseline](docs/simulation_baseline.md)
- [Experiments](docs/experiments.md)
- [Benchmarks](docs/benchmarks.md)
- [GNN preparation layer](docs/gnn_preparation.md)
- [Observation datasets](docs/observation_datasets.md)
- [Policy models](docs/policy_models.md)

## Run The First Simulation Baseline

Legacy-compatible script:

```bash
python3 scripts/run_simulation_baseline.py --policy fifo
```

Package CLI:

```bash
PYTHONPATH=src python3 -m warehouse_sim.simulation.cli --policy nearest_robot_task
```

## Run A Config-Driven Experiment

Legacy-compatible script:

```bash
python3 scripts/run_experiment.py --config configs/baseline_experiment.toml
```

Package CLI:

```bash
PYTHONPATH=src python3 -m warehouse_sim.simulation.experiment_cli --config configs/baseline_experiment.toml
```

To also export graph and dispatch observation datasets:

```bash
PYTHONPATH=src python3 -m warehouse_sim.simulation.experiment_cli \
  --config configs/baseline_experiment.toml \
  --write-observation-dataset
```

To run the first observation-driven policy:

```bash
PYTHONPATH=src python3 -m warehouse_sim.simulation.experiment_cli \
  --config configs/linear_assignment_experiment.toml
```

## Run A Benchmark

Legacy-compatible script:

```bash
python3 scripts/run_benchmark.py --config configs/policy_benchmark.toml
```

Package CLI:

```bash
PYTHONPATH=src python3 -m warehouse_sim.simulation.benchmark_cli --config configs/policy_benchmark.toml
```
