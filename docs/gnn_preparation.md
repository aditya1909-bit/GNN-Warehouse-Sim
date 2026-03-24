# GNN Preparation Layer

## Purpose

This stage does not add a learned policy, training loop, or fake GNN pipeline.
It adds the stable observation and featurization contracts that a later learned
policy layer will need.

## Implemented Now

- Static graph featurization with explicit node features and directed travel arcs
- Dispatch-time context builder that packages:
  - graph features
  - robot observations
  - task observations
  - global queue and fleet state
- Policy API hook via `DispatchPolicy.select_assignment_from_context(...)`
- Backward compatibility for existing baseline policies through the legacy
  `select_assignment(...)` adapter

## Feature Scope

Node features currently expose:

- `node_id`
- coordinates
- `node_type`
- `zone_id`
- inbound and outbound degree

Arc features currently expose:

- `source_id`
- `target_id`
- path distance
- travel time

Task observations currently expose:

- release timing and readiness
- pickup and dropoff endpoints
- zone references
- priority and service estimate
- pickup-to-dropoff travel estimates

Robot observations currently expose:

- current node and zone
- availability timing
- speed multiplier
- cumulative busy, idle, and travel counters
- completed-task count

Global observations currently expose:

- current simulation time
- pending, ready, and future task counts
- idle and busy robot counts
- mean age of ready tasks
- maximum robot availability time

## Not Implemented Yet

- Torch, PyG, DGL, or any other ML framework integration
- Learned dispatch policies
- Offline dataset generation for policy training
- State-action replay buffers
- Graph neural network training or inference

## Design Rationale

- Keep the graph features static and explainable.
- Keep dynamic policy observations separate from the simulation engine’s core
  state updates.
- Preserve the baseline dispatch API so non-learning policies stay simple.
- Expose enough typed structure that later learned policies can consume the same
  simulator state without reaching into private engine internals.
