# GNN-Warehouse-Sim

Research-grade warehouse warehouse-coordination scaffold with validated demand generation, warehouse graph/task primitives, config-driven experiments, dispatch learning, and an integrated MAPF-style coordination stack.

## Current Status

This repository is still not a full warehouse simulator, but it now includes both a dispatch-centric stack and an integrated coordination stack with graph-embedded continuous execution, an optional free-space off-graph motion mode, prioritized SIPP-style planning, an exact current-epoch MAPF routing baseline, and an experimental end-to-end macro PPO controller.

Implemented and tested now:

- Synthetic warehouse task-demand generation as a non-homogeneous Poisson process
- Synthetic grid warehouse topology with weighted shortest-path utilities
- Warehouse environment and named zone abstractions
- Explicit task objects, release-time-aware task queues, and demand-to-task adapters
- Robot specifications and discrete-event simulation runtime state
- Baseline dispatch policies: FIFO, random, nearest-robot-task, nearest-task-for-idle-robot
- Observation-driven linear scoring policy loaded from experiment config
- Offline fitting pipeline for candidate-scoring dispatch models
- Trained linear scorer artifact loading inside live simulation runs
- Modest nonlinear learned baseline: a small MLP over candidate features
- PyG-based graph-conditioned dispatch scorer with message passing over the warehouse graph
- Masked PPO fine-tuning at dispatch-event boundaries
- Integrated coordination mode with continuous-time graph execution
- Optional free-space off-graph motion realization over node coordinates
- Prioritized SIPP-style centralized planner
- Exact joint-search MAPF baseline over the current integrated macro candidate set
- Explicit robot trajectory, macro-decision, planner-plan, and collision-event outputs
- End-to-end macro PPO training and artifact loading for integrated mode
- Offline evaluation outputs for dispatch-ranking models
- Congestion-aware heuristic baseline: `congestion_aware_nearest_robot_task`
- Config-driven experiments and benchmark manifests
- Machine-readable reporting outputs: `summary.json`, `executions.csv`, `queue_snapshots.csv`, `robot_metrics.csv`
- Graph featurization, dispatch-time observations, and observation-dataset export
- Dispatch-indexed graph-state export through `dispatch_node_observations.csv` and `dispatch_arc_observations.csv`
- Interaction-aware execution modes with explicit shortest-path materialization plus simplified node/edge reservation models
- Congestion-sensitive metrics, benchmark scenarios, and pytest coverage across the stack

Implemented now, but still deliberately simplified:

- `execution_model = "idealized"` preserves the original independent-travel baseline
- `execution_model = "reserved_edges"` adds directed-edge reservations and realized waiting
- `execution_model = "reserved_nodes"` adds single-node occupancy and realized waiting

Still not implemented:

- global warehouse-level optimal MAPF guarantees across dynamic task allocation and future releases
- battery behavior or charging policies
- obstacle-aware free-space geometry beyond the current open-plane continuous motion mode

The current scope is: a dispatch-centric simulator plus an integrated coordination stack over the same warehouse graph. The dispatch stack remains the honest baseline for candidate scoring and congestion-aware execution. The integrated stack adds continuous-time graph coordination, a bounded free-space motion mode, and MAPF-style planning. The learned integrated controller is still benchmark-gated before stronger end-to-end coordination claims.

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

Integrated coordination benchmark:

```bash
PYTHONPATH=src python3 -m warehouse_sim.simulation.benchmark_cli \
  --config configs/integrated_coordination_benchmark.toml
```

Integrated exact-routing benchmark:

```bash
PYTHONPATH=src python3 -m warehouse_sim.simulation.benchmark_cli \
  --config configs/integrated_optimal_mapf_benchmark.toml
```

Free-space integrated scenario:

```bash
PYTHONPATH=src python3 -m warehouse_sim.simulation.experiment_cli \
  --config configs/scenarios/integrated_free_space_high_fleet_density.toml
```

The benchmark layer also supports optional repeated demand seeds through `benchmark.seeds`.

Repeated-seed benchmark outputs now include:

- per-run `benchmark_summary.csv`
- aggregate `benchmark_policy_aggregates.csv`
- JSON summaries with per-seed breakdowns, mean/std, and 95% confidence-interval bounds

## Fit Offline Dispatch Models

Fit the grouped linear scorer:

```bash
python3 scripts/run_offline_policy_fitting.py train --config configs/offline_linear_fit.toml
```

Fit the modest nonlinear baseline:

```bash
python3 scripts/run_offline_policy_fitting.py train --config configs/offline_mlp_fit.toml
```

Fit the graph-conditioned dispatch scorer:

```bash
python3 scripts/run_offline_policy_fitting.py train --config configs/offline_graph_dispatch_fit.toml
```

Fine-tune the graph scorer with masked PPO:

```bash
python3 scripts/run_offline_policy_fitting.py train-rl --config configs/graph_dispatch_rl_fine_tune.toml
```

Train the integrated end-to-end macro controller:

```bash
python3 scripts/run_offline_policy_fitting.py train-integrated-rl --config configs/integrated_macro_ppo_training.toml
```

Evaluate an existing artifact:

```bash
PYTHONPATH=src python3 -m warehouse_sim.learning.cli evaluate \
  --artifact outputs/offline/offline_linear_dispatch_fit/model_artifact.json \
  --dataset outputs/linear_assignment_policy/dataset_manifest.json \
  --output-dir outputs/offline/evaluation
```

The learning pipeline still stays aligned with simulator exports. Linear and MLP models consume candidate rows from `dispatch_observations.csv`. The graph-conditioned model additionally consumes dispatch-level node and arc tables exported by the simulator. This is still dispatch candidate scoring, not full learned warehouse coordination.

## Run Trained Artifacts In Simulation

Experiment configs can now point a trained scorer artifact back into the live simulator:

```toml
[simulation]
policy = "trained_linear_model"

[policy_model]
artifact_path = "artifacts/models/linear_dispatch_model.json"
```

Supported trained policy names are:

- `trained_linear_model`
- `trained_mlp_model`
- `trained_graph_dispatch_model`
- `trained_end_to_end_macro_ppo`

Integrated non-learning policy names now include:

- `prioritized_sipp_coordinator`
- `optimal_mapf_coordinator`
- `random_macro`

## Documentation

- [Repository audit](docs/repo_audit.md)
- [Architecture overview](docs/architecture.md)
- [Domain model](docs/domain_model.md)
- [Simulation baseline](docs/simulation_baseline.md)
- [Interaction-aware execution](docs/interaction_aware_execution.md)
- [Experiments](docs/experiments.md)
- [Benchmarks](docs/benchmarks.md)
- [Integrated coordination](docs/integrated_coordination.md)
- [GNN preparation layer](docs/gnn_preparation.md)
- [Observation datasets](docs/observation_datasets.md)
- [Offline policy fitting](docs/offline_policy_fitting.md)
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
