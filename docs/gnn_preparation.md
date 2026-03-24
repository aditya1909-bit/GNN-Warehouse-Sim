# GNN Preparation Layer

## Purpose

This stage still does not add a learned policy or fake GNN pipeline. It extends the observation contracts so later learned policies can see a more realistic multi-robot coordination state.

## Implemented Now

- static graph featurization with explicit directed travel arcs
- dispatch-time context builder for robot, task, and global observations
- policy API hook via `DispatchPolicy.select_assignment_from_context(...)`
- candidate observations shared by live policies and exported datasets
- congestion-sensitive observation fields derived from active reservations

## Dynamic Observation Scope

Robot observations expose:

- current node and zone
- availability timing
- speed multiplier
- cumulative busy, idle, and travel counters
- completed-task count

Task observations expose:

- release timing and readiness
- pickup and dropoff endpoints
- priority and service estimate
- pickup-to-dropoff travel estimates

Global observations expose:

- current simulation time
- pending, ready, and future task counts
- idle and busy robot counts
- mean ready-task age
- max and average robot time-until-available
- active reserved edge/node counts
- current execution model label

Candidate features now also support interpretable congestion proxies such as:

- estimated pickup congestion delay
- estimated dropoff congestion delay
- estimated pickup blocked segments
- estimated dropoff blocked segments

## Still Not Implemented

- Torch, PyG, DGL, or other ML framework integration
- learned dispatch policies
- offline training pipelines
- GNN training or inference
