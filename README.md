# GNN-Warehouse-Sim

Benchmark-first warehouse coordination research scaffold for comparing dispatch heuristics, learned dispatch scorers, integrated planners, and end-to-end macro controllers under shared synthetic scenario families.

## Problem

Warehouse coordination claims are cheap when dispatch, execution, planning, and learning are all benchmarked on different settings. This repository is built around one graph-backed warehouse model so dispatch policies, congestion-aware execution, integrated planners, and learned controllers can be compared under the same scenario definitions and machine-readable artifact contract.

## What This Repo Contributes

- Config-driven dispatch and integrated coordination experiments over the same warehouse graph abstraction
- Graph-conditioned dispatch learning, masked PPO fine-tuning, and an experimental end-to-end macro PPO controller
- Prioritized SIPP-style coordination, an exact current-epoch MAPF baseline, and continuous-time integrated execution
- Canonical metric naming, confidence-interval aggregation, claim tables, config snapshots, seed bundles, and artifact manifests

## Headline Empirical Results

The current canonical artifact bundle supports the following benchmarked comparisons:

| scenario | best baseline | best learned/planner method | uplift | 95% CI | artifact path |
| --- | --- | --- | --- | --- | --- |
| `open_low_load` | `fifo` | `congestion_aware_nearest_robot_task` | `3.93%` lower p95 task completion time | `-8.00%` to `+11.20%` | [`outputs/benchmarks/canonical_full_matrix/dispatch/benchmark_claims.csv`](outputs/benchmarks/canonical_full_matrix/dispatch/benchmark_claims.csv) |
| `narrow_bottleneck` | `random` | `congestion_aware_nearest_robot_task` | `1.32%` lower p95 task completion time | `-0.46%` to `+1.42%` | [`outputs/benchmarks/canonical_full_matrix/dispatch/benchmark_claims.csv`](outputs/benchmarks/canonical_full_matrix/dispatch/benchmark_claims.csv) |
| `integrated_narrow_bottleneck` | `random_macro` | `optimal_mapf_coordinator` | `11.11%` lower p95 task completion time | `+3.26s` to `+5.95s` | [`outputs/benchmarks/canonical_full_matrix/integrated/benchmark_claims.csv`](outputs/benchmarks/canonical_full_matrix/integrated/benchmark_claims.csv) |

Supported claims from those artifacts:

- In `open_low_load`, `congestion_aware_nearest_robot_task` reduced p95 task completion time by `3.93%` versus `fifo`.
- In `narrow_bottleneck`, `congestion_aware_nearest_robot_task` reduced p95 task completion time by `1.32%` versus `random`.
- In `integrated_narrow_bottleneck`, `optimal_mapf_coordinator` reduced p95 task completion time by `11.11%` versus `random_macro`.

Constraint on interpretation:

- The canonical dispatch artifacts are now trained from a multi-scenario corpus with scenario-seed splits and benchmark-weighted objectives, but the learned dispatch family still does not beat the strongest heuristic in the current canonical suite.
- The integrated macro controller now supports planner-guided warm start and best-checkpoint retention, but it remains benchmark-gated because reserved-resource scenarios still produce unacceptable collision counts. See [`outputs/canonical_artifacts/macro_ppo/claim_gate.json`](outputs/canonical_artifacts/macro_ppo/claim_gate.json).

## Reproducibility Quickstart

Install:

```bash
python3 -m pip install -e .[dev]
```

Build the canonical trained artifact bundle:

```bash
PYTHONPATH=src python3 scripts/build_canonical_artifacts.py \
  --output-dir outputs/canonical_artifacts
```

Run the canonical benchmark harness against that bundle:

```bash
PYTHONPATH=src python3 scripts/run_canonical_benchmarks.py \
  --config configs/benchmarks/canonical_full_matrix.toml
```

Each benchmark root now writes:

- `benchmark_summary.csv`
- `benchmark_policy_aggregates.csv`
- `benchmark_claims.csv`
- `benchmark_claims.json`
- `benchmark_summary.json`
- `figures/`
- `manifest.json`
- `config_snapshot.toml`
- `seed_bundle.json`

## Architecture Overview

- Dispatch stack: heuristics, linear/MLP scorers, graph-conditioned dispatch scoring, masked PPO fine-tuning
- Execution stack: idealized dispatch execution plus reservation-aware `reserved_edges` and `reserved_nodes` modes
- Integrated stack: continuous-time macro coordination, prioritized SIPP-style planning, exact current-epoch MAPF routing, optional free-space motion
- Reporting stack: stable metric schema, aggregate benchmark writer, claim tables, figure outputs, config snapshots, and manifest capture

## Benchmark Suite

The repository now includes canonical benchmark configs under [`configs/benchmarks/`](configs/benchmarks/):

- `canonical_dispatch_benchmark.toml`
- `canonical_integrated_benchmark.toml`
- `canonical_full_matrix.toml`

The canonical scenario family includes:

- `open_low_load`
- `open_high_load`
- `narrow_bottleneck`
- `dense_crossing`
- `high_fleet_density`
- `integrated_reserved_edges`
- `integrated_reserved_nodes`
- `integrated_free_space`
- `unseen_layout_generalization`
- `unseen_demand_generalization`

Legacy benchmark configs remain useful for narrower checks:

- [`configs/policy_benchmark.toml`](configs/policy_benchmark.toml)
- [`configs/congestion_policy_benchmark.toml`](configs/congestion_policy_benchmark.toml)
- [`configs/integrated_coordination_benchmark.toml`](configs/integrated_coordination_benchmark.toml)
- [`configs/integrated_optimal_mapf_benchmark.toml`](configs/integrated_optimal_mapf_benchmark.toml)

## Learning Stack

Dispatch learning paths currently implemented:

- offline grouped linear scoring
- offline grouped MLP scoring
- PyG-based graph-conditioned dispatch scoring
- masked PPO fine-tuning over dispatch-event action masks

The canonical dispatch artifact builder now trains from a merged multi-scenario corpus, uses `scenario_seed` train/validation/test splits, and supports benchmark-weighted losses so congestion-heavy cases matter more than low-contention snapshots.

Integrated learning path currently implemented:

- end-to-end macro PPO over centralized integrated observations with planner-guided warm start and best-validation checkpoint retention

Training entry points:

```bash
python3 scripts/run_offline_policy_fitting.py train --config configs/offline_linear_fit.toml
python3 scripts/run_offline_policy_fitting.py train --config configs/offline_mlp_fit.toml
python3 scripts/run_offline_policy_fitting.py train --config configs/offline_graph_dispatch_fit.toml
python3 scripts/run_offline_policy_fitting.py train-rl --config configs/graph_dispatch_rl_fine_tune.toml
python3 scripts/run_offline_policy_fitting.py train-integrated-rl --config configs/integrated_macro_ppo_training.toml
```

## Planner Stack

Integrated non-learning policies currently supported:

- `prioritized_sipp_coordinator`
- `optimal_mapf_coordinator`
- `random_macro`

The integrated stack writes explicit robot trajectories, macro decisions, collision events, and planner-plan tables so planner behavior is inspectable beyond scalar summary metrics.

## Limitations / Honest Caveats

- The checked-in headline results currently support planner and congestion-aware heuristic claims, not broad learned-policy superiority claims.
- The canonical full-matrix configs expect trained artifact paths for `trained_linear_model`, `trained_mlp_model`, `trained_graph_dispatch_model`, and `trained_end_to_end_macro_ppo`. Generating that artifact bundle is the remaining step before the canonical suite can support full learning-vs-planning claims end to end.
- Global optimal MAPF over future task releases is still out of scope. The exact MAPF baseline is explicitly current-epoch and bounded to the current macro candidate surface.
- Battery/charging behavior and obstacle-aware free-space geometry are still not implemented.

## Documentation

- [Architecture overview](docs/architecture.md)
- [Benchmarks](docs/benchmarks.md)
- [Metric definitions](docs/metric_definitions.md)
- [Integrated coordination](docs/integrated_coordination.md)
- [Offline policy fitting](docs/offline_policy_fitting.md)
- [Policy models](docs/policy_models.md)
- [Observation datasets](docs/observation_datasets.md)
- [Repository audit](docs/repo_audit.md)
