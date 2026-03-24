# Repository Audit

## Implemented Now

The repository now has ten implemented layers:

- Synthetic task arrivals from a non-homogeneous Poisson process
- Explicit warehouse graphs, zones, tasks, queues, and robot state
- Discrete-event dispatch simulation
- Config-driven experiments and machine-readable reporting
- Scenario presets and benchmark workflows
- Graph and dispatch observation contracts
- Observation-dataset export
- Observation-driven linear scoring policy
- Interaction-aware execution with explicit routes, simplified reservations, and congestion-aware metrics
- Offline fitting and evaluation for candidate-scoring dispatch models

Concrete additions in the latest stage:

- shortest-path node and edge materialization
- per-task route-level execution details
- configurable `idealized`, `reserved_edges`, and `reserved_nodes` execution modes
- realized waiting from shared resource contention
- congestion-aware task and robot metrics
- congestion-sensitive observation features
- congestion-aware heuristic baseline
- contention-focused scenario presets and benchmark manifest
- grouped dispatch-dataset loading and split support
- fitted linear scorer artifact serialization and live loading
- one modest nonlinear learned baseline over candidate features
- repeated-seed aggregate benchmark reporting

## Not Implemented Yet

The repository still does not implement a full warehouse simulator.

- No full MAPF or optimal joint path planning
- No continuous collision geometry
- No battery or charging behavior
- No true graph-neural dispatch model
- No RL training pipeline
- No end-to-end learned coordination system

## Current Framing

This codebase is now strong enough for experiments on dispatch decisions under simplified interaction-aware execution, plus honest offline fitting and evaluation of candidate-scoring models on exported dispatch observations.

It should still be described as a modular research scaffold, not as a finished warehouse-operations simulator.
