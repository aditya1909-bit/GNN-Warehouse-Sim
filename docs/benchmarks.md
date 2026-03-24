# Benchmarks

## Scope

Benchmarks still run config-driven scenario/policy combinations, but the scenario set now includes contention-focused presets and congestion-sensitive metrics.

## Included Benchmark Manifests

- `configs/policy_benchmark.toml`: general baseline comparison across the original synthetic scenarios
- `configs/congestion_policy_benchmark.toml`: contention-focused comparison across interaction-heavy scenarios

## Contention-Focused Scenarios

- `narrow_bottleneck`: forced chokepoint in the middle of the layout
- `high_fleet_density`: more robots relative to grid size
- `asymmetric_flow`: directional corridor pressure
- `station_queueing`: repeated arrivals into a shared destination area using node reservations

These are still synthetic research scenarios, not calibrated operational warehouse models.

## Benchmark Outputs

Each run still writes its per-run experiment artifacts plus:

- `benchmark_summary.csv`
- `benchmark_summary.json`

The aggregate rows now include congestion-aware metrics such as:

- realized travel time total
- realized travel distance total
- congestion delay total
- average congestion delay per completed task
- blocked traversal events total

## Repeated Seeds

The benchmark manifest now supports optional `benchmark.seeds`. When provided, the runner repeats each scenario/policy combination across those demand seeds while keeping the rest of the scenario config fixed.
