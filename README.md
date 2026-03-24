# GNN-Warehouse-Sim

Research-grade warehouse dispatch-simulation scaffold with validated demand generation, warehouse graph/task primitives, discrete-event execution, config-driven experiments, benchmark scenarios, observation export, and simple observation-driven baselines.

## Current Status

This repository is still not a full warehouse simulator, and it still does not implement learned GNN policies.

Implemented and tested now:

- Synthetic warehouse task-demand generation as a non-homogeneous Poisson process
- Synthetic grid warehouse topology with weighted shortest-path utilities
- Warehouse environment and named zone abstractions
- Explicit task objects, release-time-aware task queues, and demand-to-task adapters
- Robot specifications and discrete-event simulation runtime state
- Baseline dispatch policies: FIFO, random, nearest-robot-task, nearest-task-for-idle-robot
- Observation-driven linear scoring policy loaded from experiment config
- Congestion-aware heuristic baseline: `congestion_aware_nearest_robot_task`
- Config-driven experiments and benchmark manifests
- Machine-readable reporting outputs: `summary.json`, `executions.csv`, `queue_snapshots.csv`, `robot_metrics.csv`
- Graph featurization, dispatch-time observations, and observation-dataset export
- Interaction-aware execution modes with explicit shortest-path materialization plus simplified node/edge reservation models
- Congestion-sensitive metrics, benchmark scenarios, and pytest coverage across the stack

Implemented now, but still deliberately simplified:

- `execution_model = "idealized"` preserves the original independent-travel baseline
- `execution_model = "reserved_edges"` adds directed-edge reservations and realized waiting
- `execution_model = "reserved_nodes"` adds single-node occupancy and realized waiting

Still not implemented:

- Full MAPF or optimal multi-agent planning
- Continuous-motion collision simulation
- Battery behavior or charging policies
- Learned GNN dispatch policies
- Reinforcement-learning or imitation-learning pipelines

The current scope is: multi-robot dispatch over a warehouse graph with optional congestion-aware realized execution. It is meant as a research scaffold for coordination experiments, not as an overclaimed end-state simulator.

## Install

```bash
python3 -m pip install -e .[dev]
```

## Run The Baseline Simulation

Legacy-compatible script:

```bash
python3 scripts/run_simulation_baseline.py --policy fifo
```

Package CLI:

```bash
PYTHONPATH=src python3 -m warehouse_sim.simulation.cli \
  --policy congestion_aware_nearest_robot_task \
  --execution-model reserved_edges
```

## Run A Config-Driven Experiment

```bash
python3 scripts/run_experiment.py --config configs/baseline_experiment.toml
```

```bash
PYTHONPATH=src python3 -m warehouse_sim.simulation.experiment_cli \
  --config configs/scenarios/narrow_bottleneck.toml
```

Execution mode is controlled in the `[simulation]` section:

```toml
[simulation]
policy = "congestion_aware_nearest_robot_task"
execution_model = "reserved_edges"
```

## Run Benchmarks

Baseline comparison:

```bash
python3 scripts/run_benchmark.py --config configs/policy_benchmark.toml
```

Contention-focused benchmark:

```bash
PYTHONPATH=src python3 -m warehouse_sim.simulation.benchmark_cli \
  --config configs/congestion_policy_benchmark.toml
```

The benchmark layer also supports optional repeated demand seeds through `benchmark.seeds`.

## Documentation

- [Repository audit](docs/repo_audit.md)
- [Architecture overview](docs/architecture.md)
- [Domain model](docs/domain_model.md)
- [Simulation baseline](docs/simulation_baseline.md)
- [Interaction-aware execution](docs/interaction_aware_execution.md)
- [Experiments](docs/experiments.md)
- [Benchmarks](docs/benchmarks.md)
- [GNN preparation layer](docs/gnn_preparation.md)
- [Observation datasets](docs/observation_datasets.md)
- [Policy models](docs/policy_models.md)

## Notebooks

- `notebooks/e2e_cli_workflow.ipynb`: runs the documented CLI workflows end to end
- `notebooks/e2e_python_api_workflow.ipynb`: runs the same stack through the Python APIs and compares idealized vs congestion-aware execution

## Generate Synthetic Task Demand

```bash
python3 scripts/generate_task_demand.py
```

```bash
PYTHONPATH=src python3 -m warehouse_sim.demand.cli
```

Default output: `data/task_demand.csv`
