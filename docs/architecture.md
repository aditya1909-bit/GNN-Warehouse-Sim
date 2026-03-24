# Architecture Overview

## Current Scope

The repository now has nine concrete layers:

- Stage 1: validated stochastic demand/input model
- Stage 2: explicit warehouse graph, environment, and task-domain primitives
- Stage 3: discrete-event simulation baseline
- Stage 4: config-driven experiment runs with machine-readable reporting
- Stage 5: scenario presets and multi-policy benchmarks
- Stage 6: graph featurization and dispatch-time observation contracts
- Stage 7: observation-dataset export
- Stage 8: the first observation-driven linear scoring policy
- Stage 9: interaction-aware execution with route materialization and simplified congestion reservations

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
    utils/        # future shared helpers
```

## Design Principles

- Keep business logic under `src/warehouse_sim/` and keep scripts thin.
- Extend existing modules instead of creating redundant parallel stacks.
- Preserve baseline workflows by making new fidelity modes opt-in or default-safe.
- Prefer explicit typed contracts so simulation, reporting, and policy code stay aligned.
- Keep the framing honest: simplified reservations are not MAPF, and observation hooks are not learned policies.

## Interaction-Aware Execution Placement

Stage 9 intentionally lands inside existing layers rather than beside them:

- Graph/environment now expose explicit path node and edge sequences.
- Simulation execution can materialize routes and optionally reserve edges or nodes.
- Metrics/reporting reuse the existing summary and CSV outputs with added congestion fields.
- Dispatch observations reuse the same candidate-building path and now expose small, interpretable congestion features.

## Still Out Of Scope

- Full multi-agent path finding
- Continuous collision geometry
- Battery and charging behavior
- Learned GNN dispatch policies
