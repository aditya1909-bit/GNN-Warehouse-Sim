# Repository Audit

## Implemented Now

The repository now has twelve implemented layers:

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
- Graph-conditioned dispatch learning with optional PPO fine-tuning
- Integrated continuous-time coordination with prioritized SIPP-style planning, exact current-epoch MAPF routing, and macro PPO

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
- dispatch-indexed node and arc observation export
- PyG message-passing graph dispatch scorer
- masked PPO fine-tuning at dispatch-event boundaries
- graph-dispatch artifact loading inside live simulation runs
- integrated coordination mode with continuous graph execution
- prioritized SIPP-style planner outputs and collision-event reporting
- exact current-epoch MAPF routing baseline for integrated mode
- end-to-end macro PPO artifact training and loading

## Not Implemented Yet

The repository still does not implement a full warehouse simulator.

- No battery or charging behavior
- No global warehouse-level optimal MAPF guarantees across dynamic task allocation and future releases
- No obstacle-aware free-space off-graph geometry beyond the current open-plane motion mode

## Current Framing

This codebase is now strong enough for both dispatch-centric coordination experiments and an integrated continuous-time coordination stack with centralized planning and experimental end-to-end macro RL.

It should still be described as a modular research scaffold, not as a finished warehouse-operations simulator. Stronger learned end-to-end coordination claims remain benchmark-gated.
