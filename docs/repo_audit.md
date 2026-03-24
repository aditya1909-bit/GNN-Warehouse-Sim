# Repository Audit

## Implemented Now

The repository now has nine implemented layers:

- Synthetic task arrivals from a non-homogeneous Poisson process
- Explicit warehouse graphs, zones, tasks, queues, and robot state
- Discrete-event dispatch simulation
- Config-driven experiments and machine-readable reporting
- Scenario presets and benchmark workflows
- Graph and dispatch observation contracts
- Observation-dataset export
- Observation-driven linear scoring policy
- Interaction-aware execution with explicit routes, simplified reservations, and congestion-aware metrics

Concrete additions in the latest stage:

- shortest-path node and edge materialization
- per-task route-level execution details
- configurable `idealized`, `reserved_edges`, and `reserved_nodes` execution modes
- realized waiting from shared resource contention
- congestion-aware task and robot metrics
- congestion-sensitive observation features
- congestion-aware heuristic baseline
- contention-focused scenario presets and benchmark manifest

## Not Implemented Yet

The repository still does not implement a full warehouse simulator.

- No full MAPF or optimal joint path planning
- No continuous collision geometry
- No battery or charging behavior
- No learned policies
- No GNN training or inference pipeline
- No automatic fitting/training for the linear policy weights

## Current Framing

This codebase is now strong enough for experiments on dispatch decisions under simplified interaction-aware execution.

It should still be described as a modular research scaffold, not as a finished warehouse-operations simulator.
