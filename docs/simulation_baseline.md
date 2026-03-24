# Simulation Baseline

## Stage 3 Scope

Stage 3 adds the first working simulation loop. It is intentionally minimal, but it is no longer placeholder code.

- explicit robot definitions
- baseline dispatch policies
- a discrete-event execution loop
- run-level metrics
- a thin CLI for baseline simulation runs

## Why Discrete-Event

This project now uses a discrete-event baseline instead of a time-stepped loop because the current process is dominated by sparse state changes:

- task releases
- robot assignment starts
- robot availability after travel and service

Advancing directly between these events keeps the baseline explainable and avoids unnecessary per-timestep bookkeeping before congestion and richer dynamics are introduced.

## Process Overview And Scheduling

For the current baseline:

1. Tasks exist with explicit release times.
2. Robots become idle when their assigned work completes.
3. At each event time, all ready tasks and idle robots are considered.
4. The chosen dispatch policy emits one robot-task pairing at a time.
5. The engine computes travel to pickup, service time, and travel to dropoff.
6. The robot becomes unavailable until task completion.
7. The simulation advances to the next task release or robot availability event.

This keeps the baseline deterministic once the demand sample, layout, robot definitions, and policy seed are fixed.

## Baseline Policies

Implemented now:

- `fifo`
- `random`
- `nearest_robot_task`
- `nearest_task_for_idle_robot`

These are non-learning baselines intended to remain available after future GNN policies are added.

## Current Simplifications

The current engine still omits several warehouse dynamics by design.

- no collisions or congestion
- no battery constraints
- no path reservation or MAPF logic
- no robot failures or task preemption
- no partial task execution state
- no stochastic service-time realization separate from the task estimate

Those belong in later stages once the baseline interfaces stabilize.

