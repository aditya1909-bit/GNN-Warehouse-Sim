# Architecture Overview

## Current Scope

The repository now has ten concrete layers:

- Stage 1: validated stochastic demand/input model
- Stage 2: explicit warehouse graph, environment, and task-domain primitives
- Stage 3: discrete-event simulation baseline
- Stage 4: config-driven experiment runs with machine-readable reporting
- Stage 5: scenario presets and multi-policy benchmarks
- Stage 6: graph featurization and dispatch-time observation contracts
- Stage 7: observation-dataset export
- Stage 8: the first observation-driven linear scoring policy
- Stage 9: interaction-aware execution with route materialization and simplified congestion reservations
- Stage 10: offline policy fitting and evaluation for candidate-scoring dispatch models

```text
src/
  warehouse_sim/
    demand/       # validated stochastic input model
    config/       # experiment and benchmark configuration models/loaders
    environment/  # warehouse environment and named zone abstractions
    tasks/        # task objects, queues, and demand adapters
    agents/       # robot specifications and runtime state
    graph/        # warehouse topology, shortest paths, and explicit path utilities
    simulation/   # discrete-event engine, execution models, benchmarks
    policies/     # heuristic baselines, observation hooks, scoring policies
    metrics/      # summary metrics, datasets, plots, and report writers
    learning/     # offline dataset loading, splits, fit/eval, model artifacts
    utils/        # future shared helpers
```

## Design Principles

- Keep business logic under `src/warehouse_sim/` and keep scripts thin.
- Extend existing modules instead of creating redundant parallel stacks.
- Preserve baseline workflows by making new fidelity modes opt-in or default-safe.
- Prefer explicit typed contracts so simulation, reporting, and policy code stay aligned.
- Keep the framing honest: simplified reservations are not MAPF, and observation hooks are not learned policies.
- Treat offline fitting as candidate scoring over exported dispatch observations, not as a disguised GNN or RL stack.

## Interaction-Aware Execution Placement

Stage 9 intentionally lands inside existing layers rather than beside them:

- Graph/environment now expose explicit path node and edge sequences.
- Simulation execution can materialize routes and optionally reserve edges or nodes.
- Metrics/reporting reuse the existing summary and CSV outputs with added congestion fields.
- Dispatch observations reuse the same candidate-building path and now expose small, interpretable congestion features.

## Offline Policy Fitting Placement

Stage 10 also lands inside existing layers instead of beside them:

- Dataset loading is built around the existing `dispatch_observations.csv` export contract.
- Fitted artifacts reuse the same named candidate-feature contract as live scoring policies.
- The simulator loads trained artifacts through the existing policy/config path.
- Benchmark aggregation extends the current benchmark report layer with repeated-seed statistics.

## Still Out Of Scope

- Full multi-agent path finding
- Continuous collision geometry
- Battery and charging behavior
- Learned GNN dispatch policies
- RL training pipelines
