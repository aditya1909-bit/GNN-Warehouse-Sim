# Benchmarks

## Stage 5 Scope

Stage 5 adds richer scenario presets and a benchmark workflow for comparing baseline policies.

- scenario preset configs under `configs/scenarios/`
- a benchmark manifest under `configs/policy_benchmark.toml`
- a benchmark runner that executes all scenario-policy combinations
- aggregate comparison outputs in CSV and JSON

## Included Scenarios

The current preset set is intentionally small but varied.

- `peak_load`: higher arrival intensity with a rush window
- `one_way_flow`: one-way aisle segments in the grid
- `blocked_cross_aisle`: static blocked cells that force detours

These are still synthetic scenarios. They are not yet calibrated to a real warehouse.

## Benchmark Outputs

A benchmark run writes:

- per-run experiment artifacts under scenario/policy subdirectories
- `benchmark_summary.csv`
- `benchmark_summary.json`

The aggregate report records throughput, waiting time, turnaround time, travel distance, queue length, makespan, and the path to each per-run summary.

