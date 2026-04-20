# Benchmarks

## Canonical Suite

The repository now has a canonical benchmark harness for benchmark-first comparisons:

- [`configs/benchmarks/canonical_dispatch_benchmark.toml`](../configs/benchmarks/canonical_dispatch_benchmark.toml)
- [`configs/benchmarks/canonical_integrated_benchmark.toml`](../configs/benchmarks/canonical_integrated_benchmark.toml)
- [`configs/benchmarks/canonical_full_matrix.toml`](../configs/benchmarks/canonical_full_matrix.toml)

Run the full suite with:

```bash
PYTHONPATH=src python3 scripts/build_canonical_artifacts.py \
  --output-dir outputs/canonical_artifacts
```

```bash
PYTHONPATH=src python3 scripts/run_canonical_benchmarks.py \
  --config configs/benchmarks/canonical_full_matrix.toml
```

That command runs the dispatch and integrated benchmark families, writes per-benchmark artifacts, and then combines their claim tables into one headline-results bundle.

Use that scripted workflow, not the notebooks, as the authoritative reproduction path for benchmark claims.

The artifact-builder step now:

- exports a merged dispatch corpus from the canonical training scenarios,
- keeps scenario-seed train/validation/test splits reproducible,
- trains the linear, MLP, and graph dispatch scorers with optional benchmark weighting,
- trains the integrated macro controller with planner-guided warm start and best-checkpoint selection.

## Claim Matrix

The checked-in canonical artifacts support the following claim status:

| family | current status | interpretation |
| --- | --- | --- |
| Dispatch heuristics | Supported, but scenario-specific | The clearest current dispatch claim is the `open_low_load` win for `congestion_aware_nearest_robot_task`; other dispatch scenarios are mixed or effectively tied. |
| Dispatch learned models | Inconclusive for headline use | The learned dispatch family is present in the suite, but it does not beat the strongest heuristic across the checked-in canonical results. |
| Integrated planners | Supported | `optimal_mapf_coordinator` has the clearest checked-in positive result, especially on `integrated_narrow_bottleneck`. |
| Integrated macro PPO | Benchmark-gated | Treat as experimental until the throughput gate in `outputs/canonical_artifacts/macro_ppo/claim_gate.json` is satisfied. |

## Scenario Families

The canonical scenario matrix is organized around fixed named regimes:

- `open_low_load`
- `open_high_load`
- `narrow_bottleneck`
- `dense_crossing`
- `high_fleet_density`
- `dispatch_due_pressure`
- `integrated_reserved_edges`
- `integrated_reserved_nodes`
- `integrated_narrow_bottleneck`
- `integrated_tight_chokepoint`
- `integrated_free_space`
- `unseen_layout_generalization`
- `unseen_demand_generalization`

Across the suite, those scenarios vary fleet size, demand rate, topology pressure, execution model, coordination mode, and now include:

- a due-time-pressure dispatch case with multiple source and sink zones,
- a tighter integrated chokepoint case designed to amplify path-conflict pressure.

The canonical manifests now use ten shared seeds per scenario so paired policy comparisons have enough power for scenario-level claim language.

## Artifact Contract

Each benchmark root writes:

- `benchmark_summary.csv`
- `benchmark_policy_aggregates.csv`
- `benchmark_claims.csv`
- `benchmark_claims.json`
- `benchmark_paired_deltas.csv`
- `policy_distinctness_audit.csv`
- `benchmark_summary.json`
- `figures/`
- `manifest.json`
- `config_snapshot.toml`
- `seed_bundle.json`

The aggregate tables use one stable metric schema, include mean/std/95% CI columns, keep demand seeds aligned across policies whenever the compared policies can share the same scenario config, and now add paired seed-wise deltas for claim rows.

## Visual Outputs

The benchmark figure bundle is now designed to explain both outcome and mechanism:

- `claim_forest_plot.png` for scenario-by-scenario deltas versus baseline with CI whiskers,
- `paired_seed_dot_plot.png` for seed-level robustness,
- `throughput_small_multiples.png` grouped by scenario family,
- `seen_vs_unseen_gap.png` for generalization gaps,
- `integrated_narrow_bottleneck_mechanism.png` for planner-vs-baseline trajectory, occupancy, and CDF comparisons,
- `integrated_narrow_bottleneck_congestion_heatmap.png` for graph-level congestion intensity,
- `dispatch_decision_explainer.png` for one representative dispatch event,
- `policy_distinctness_heatmap.png` for policy-collapse checks.

## Reproducibility

Every benchmark report now captures:

- the metric schema version,
- a config snapshot covering the benchmark manifest and scenario manifests,
- a machine-readable seed bundle,
- trace-backed artifact paths for each run so raw executions, dispatch traces, macro decisions, and trajectories are inspectable from benchmark rows,
- the current git commit hash in `manifest.json`.

That is the minimum bundle required to regenerate a headline claim from a saved artifact root.

## Legacy Benchmarks

The earlier benchmark manifests remain available for narrower checks:

- [`configs/policy_benchmark.toml`](../configs/policy_benchmark.toml)
- [`configs/congestion_policy_benchmark.toml`](../configs/congestion_policy_benchmark.toml)
- [`configs/integrated_coordination_benchmark.toml`](../configs/integrated_coordination_benchmark.toml)
- [`configs/integrated_optimal_mapf_benchmark.toml`](../configs/integrated_optimal_mapf_benchmark.toml)

They now emit the same reproducibility bundle and canonical metric names as the canonical suite.

## Near-Term Focus

The next benchmark-focused research cycle should prioritize the integrated planner story:

- keep improving reproducibility and benchmark automation,
- add planner-facing analysis where the integrated stack already shows a measurable win,
- keep using distinctness audits to verify learned policies are not collapsing onto baselines,
- avoid broad learned-policy claims until the current benchmark gates and repeated-seed evidence are stronger.
