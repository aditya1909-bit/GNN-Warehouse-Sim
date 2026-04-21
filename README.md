# GNN-Warehouse-Sim

Benchmark-first warehouse coordination research scaffold for comparing dispatch heuristics, learned dispatch scorers, integrated planners, and end-to-end macro controllers under shared synthetic scenario families.

## Problem

Warehouse coordination claims are cheap when dispatch, execution, planning, and learning are all benchmarked on different settings. This repository is built around one graph-backed warehouse model so dispatch policies, congestion-aware execution, integrated planners, and learned controllers can be compared under the same scenario definitions and machine-readable artifact contract.

## What This Repo Contributes

- Config-driven dispatch and integrated coordination experiments over the same warehouse graph abstraction
- Graph-conditioned dispatch learning, masked PPO fine-tuning, and an experimental end-to-end macro PPO controller
- Prioritized SIPP-style coordination, an exact current-epoch MAPF baseline, and continuous-time integrated execution
- Canonical metric naming, paired seed-wise deltas, distinctness audits, claim tables, config snapshots, seed bundles, and artifact manifests

## Current Claim Status

The checked-in artifacts support a narrower story than broad "learning wins" framing. Current status:

| family | current status | evidence |
| --- | --- | --- |
| Dispatch heuristics | Supported, but scenario-specific | `congestion_aware_nearest_robot_task` improves `open_low_load` p95 task completion time by `3.93%` versus `fifo`. See [`outputs/benchmarks/canonical_full_matrix/dispatch/benchmark_claims.csv`](outputs/benchmarks/canonical_full_matrix/dispatch/benchmark_claims.csv). |
| Dispatch learned models | Not supported as a headline claim | The learned dispatch family does not beat the strongest heuristic across the current canonical suite. |
| Integrated planners | Strongest supported result | `optimal_mapf_coordinator` improves `integrated_narrow_bottleneck` p95 task completion time by `11.00%` versus `random_macro`. See [`outputs/benchmarks/canonical_full_matrix/integrated/benchmark_claims.csv`](outputs/benchmarks/canonical_full_matrix/integrated/benchmark_claims.csv). |
| Integrated macro PPO | Benchmark-gated | The controller clears the safety and completion gates in the checked-in bundle, but its throughput ratio versus baseline is `0.649`, below the `0.9` threshold in [`outputs/canonical_artifacts/macro_ppo/claim_gate.json`](outputs/canonical_artifacts/macro_ppo/claim_gate.json). |

Interpretation constraints:

- Planner claims are currently the clearest positive result surface in the repository.
- Dispatch results are mixed and should be presented scenario by scenario, not as a blanket superiority claim.
- The integrated learned controller remains experimental until the benchmark gate is satisfied.

## Reproducibility Quickstart

Create a repo-local environment and install the supported test/runtime contract:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .[dev]
python3 -m pytest -q
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
- `benchmark_paired_deltas.csv`
- `policy_distinctness_audit.csv`
- `policy_collapse_diagnostics.csv`
- `policy_collapse_diagnostics.json`
- `benchmark_summary.json`
- `figures/`
- `manifest.json`
- `config_snapshot.toml`
- `seed_bundle.json`

The scripted path above is the authoritative reproduction route for benchmark claims. The notebooks under [`notebooks/`](notebooks/) are exploratory walkthroughs that reuse the same package APIs and write to repo-relative output roots, but they are not the source of record for headline results.

## Architecture Overview

- Dispatch stack: heuristics, linear/MLP scorers, graph-conditioned dispatch scoring, masked PPO fine-tuning
- Execution stack: idealized dispatch execution plus reservation-aware `reserved_edges` and `reserved_nodes` modes
- Integrated stack: continuous-time macro coordination, prioritized SIPP-style planning, exact current-epoch MAPF routing, optional free-space motion
- Reporting stack: stable metric schema, aggregate benchmark writer, paired-delta claim analysis, decision-distinctness audits, figure outputs, config snapshots, and manifest capture

## Benchmark Suite

The repository now includes canonical benchmark configs under [`configs/benchmarks/`](configs/benchmarks/):

- `canonical_dispatch_benchmark.toml`
- `canonical_integrated_benchmark.toml`
- `canonical_full_matrix.toml`
- `canonical_full_matrix_smoke.toml`
- `canonical_dispatch_benchmark_heavy.toml`
- `canonical_integrated_benchmark_heavy.toml`
- `canonical_full_matrix_heavy.toml`

The canonical scenario family includes:

- `open_low_load`
- `open_high_load`
- `narrow_bottleneck`
- `dense_crossing`
- `high_fleet_density`
- `dispatch_due_pressure`
- `dispatch_due_pressure_heavy`
- `integrated_reserved_edges`
- `integrated_reserved_nodes`
- `integrated_narrow_bottleneck`
- `integrated_tight_chokepoint`
- `integrated_tight_chokepoint_heavy`
- `integrated_high_fleet_density_heavy`
- `integrated_free_space`
- `unseen_layout_generalization`
- `unseen_demand_generalization`

The smoke suite keeps the current quick-turn runtime for CI and local sanity checks. The heavier research suite expands to twenty-five shared seeds per scenario, larger `8x8`/`10x10`/`12x12` layouts, longer `1800s-3000s` horizons, higher fleet counts, and more conflict-heavy dispatch and planner regimes.

The benchmark bundle now writes a visual and diagnostic package centered on:

- claim forest plots,
- per-seed paired-improvement dots,
- scenario-family small multiples,
- seen-vs-unseen gap plots,
- integrated bottleneck mechanism figures,
- congestion heatmaps,
- dispatch decision explainers.
- policy-collapse diagnostics.

The canonical metric schema now also exposes due-time and planner-conflict metrics:

- `on_time_completion_rate`
- `mean_tardiness`
- `p95_tardiness`
- `overdue_task_count`
- `path_conflict_count_before_resolution`
- `sipp_wait_insertion_count`
- `planner_wait_time_total`

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

The canonical dispatch artifact builder now trains from a merged multi-scenario corpus, uses `scenario_seed` train/validation/test splits, supports benchmark-weighted losses so congestion-heavy cases matter more than low-contention snapshots, and writes one relocatable artifact manifest that canonical benchmarks can consume without hardcoded artifact paths.

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

The integrated stack writes explicit robot trajectories, macro decisions, collision events, planner-plan tables, and charging execution tables so planner and energy behavior are inspectable beyond scalar summary metrics.

## Near-Term Priority

The next research cycle should strengthen the planner-first story rather than expand the learning surface:

- harden reproducibility and benchmark automation first,
- expand planner-facing analysis around the integrated scenarios that already show a measurable win,
- defer new learned-control claims until the current benchmark gate and reproducibility story are stronger.

## Limitations / Honest Caveats

- The checked-in headline results currently support planner and congestion-aware heuristic claims, not broad learned-policy superiority claims.
- The canonical full-matrix configs now resolve learned policies through the canonical artifact manifest. The artifact bundle still needs to be generated before the canonical suite can support full learning-vs-planning claims end to end.
- Global optimal MAPF over future task releases is still out of scope. The exact MAPF baseline is explicitly current-epoch and bounded to the current macro candidate surface.
- Dispatch and integrated runs now support battery-aware task filtering, charge actions, charging metrics, and explicit polygon obstacle geometry, but this is still a research scaffold rather than a warehouse-grade battery-management or CAD-geometry stack.

## Documentation

- [Architecture overview](docs/architecture.md)
- [Benchmarks](docs/benchmarks.md)
- [Metric definitions](docs/metric_definitions.md)
- [Integrated coordination](docs/integrated_coordination.md)
- [Offline policy fitting](docs/offline_policy_fitting.md)
- [Policy models](docs/policy_models.md)
- [Observation datasets](docs/observation_datasets.md)
- [Repository audit](docs/repo_audit.md)
