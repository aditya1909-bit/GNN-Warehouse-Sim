# Benchmarks

## Canonical Suite

The repository now has a canonical benchmark harness for benchmark-first comparisons:

- [`configs/benchmarks/canonical_dispatch_benchmark.toml`](/Users/adityadutta/Desktop/GitHub/GNN-Warehouse-Sim/configs/benchmarks/canonical_dispatch_benchmark.toml)
- [`configs/benchmarks/canonical_integrated_benchmark.toml`](/Users/adityadutta/Desktop/GitHub/GNN-Warehouse-Sim/configs/benchmarks/canonical_integrated_benchmark.toml)
- [`configs/benchmarks/canonical_full_matrix.toml`](/Users/adityadutta/Desktop/GitHub/GNN-Warehouse-Sim/configs/benchmarks/canonical_full_matrix.toml)

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

The artifact-builder step now:

- exports a merged dispatch corpus from the canonical training scenarios,
- keeps scenario-seed train/validation/test splits reproducible,
- trains the linear, MLP, and graph dispatch scorers with optional benchmark weighting,
- trains the integrated macro controller with planner-guided warm start and best-checkpoint selection.

## Scenario Families

The canonical scenario matrix is organized around fixed named regimes:

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

Across the suite, those scenarios vary fleet size, demand rate, topology pressure, execution model, and coordination mode.

## Artifact Contract

Each benchmark root writes:

- `benchmark_summary.csv`
- `benchmark_policy_aggregates.csv`
- `benchmark_claims.csv`
- `benchmark_claims.json`
- `benchmark_summary.json`
- `figures/`
- `manifest.json`
- `config_snapshot.toml`
- `seed_bundle.json`

The aggregate tables use one stable metric schema, include mean/std/95% CI columns, and keep demand seeds aligned across policies whenever the compared policies can share the same scenario config.

## Reproducibility

Every benchmark report now captures:

- the metric schema version,
- a config snapshot covering the benchmark manifest and scenario manifests,
- a machine-readable seed bundle,
- the current git commit hash in `manifest.json`.

That is the minimum bundle required to regenerate a headline claim from a saved artifact root.

## Legacy Benchmarks

The earlier benchmark manifests remain available for narrower checks:

- [`configs/policy_benchmark.toml`](/Users/adityadutta/Desktop/GitHub/GNN-Warehouse-Sim/configs/policy_benchmark.toml)
- [`configs/congestion_policy_benchmark.toml`](/Users/adityadutta/Desktop/GitHub/GNN-Warehouse-Sim/configs/congestion_policy_benchmark.toml)
- [`configs/integrated_coordination_benchmark.toml`](/Users/adityadutta/Desktop/GitHub/GNN-Warehouse-Sim/configs/integrated_coordination_benchmark.toml)
- [`configs/integrated_optimal_mapf_benchmark.toml`](/Users/adityadutta/Desktop/GitHub/GNN-Warehouse-Sim/configs/integrated_optimal_mapf_benchmark.toml)

They now emit the same reproducibility bundle and canonical metric names as the canonical suite.
