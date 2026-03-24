# Architecture Overview

## Current Scope

The repository now has eight concrete layers:

- Stage 1: validated stochastic demand/input model
- Stage 2: explicit warehouse graph, environment, and task-domain primitives
- Stage 3: a minimal discrete-event simulation baseline with robots, dispatch policies, and metrics
- Stage 4: config-driven experiment runs with machine-readable reporting outputs
- Stage 5: scenario presets and multi-policy benchmark comparison workflows
- Stage 6: graph featurization and dispatch-time observation contracts for future learned policies
- Stage 7: observation-dataset export for future learned-policy experiments
- Stage 8: the first observation-driven policy model integrated into experiments

```text
src/
  warehouse_sim/
    demand/       # validated stochastic input model
    config/       # experiment configuration models and TOML loader
    environment/  # warehouse environment and named zone abstractions
    tasks/        # task objects, queues, and demand adapters
    agents/       # robot specifications and runtime state
    simulation/   # discrete-event baseline engine and run models
    graph/        # warehouse topology and path utilities
    policies/     # baseline dispatch APIs and observation hooks
    metrics/      # simulation metrics collection and reporting
    utils/        # future shared helpers
```

## Design Principles

- Keep the demand model correct, reproducible, and testable before expanding scope.
- Keep scripts thin and move business logic into `src/`.
- Use explicit typed interfaces so later simulation and learning layers can compose around stable contracts.
- Preserve backward compatibility for the original demand-generation workflow where practical.
- Prefer honest placeholders over speculative implementations.
- Keep the environment and task model independent from any premature simulation-engine decisions.

## ODD Alignment

This structure is chosen so future documentation can map cleanly onto an ODD-style model description.

- Overview:
  package-level description of purpose, entities, state variables, and scales
- Design Concepts:
  stochasticity, interaction, objectives, sensing, observation, and adaptation hooks
- Details:
  initialization, input data, submodels, process scheduling, and metrics

## Next Stages

### Optional Follow-On

- Add richer non-linear or learned policy models only after the linear observation-driven baseline proves useful
- Add evaluation utilities that compare future learned policies against the existing baselines and linear scorer
