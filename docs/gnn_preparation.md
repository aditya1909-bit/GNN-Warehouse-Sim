# GNN Preparation Layer

## Purpose

This layer now does more than prepare future learning. It extends the observation contracts with graph-state exports that feed the implemented graph-conditioned dispatch model, while still keeping the framing honest about the limits of that model.

## Implemented Now

- static graph featurization with explicit directed travel arcs
- structural graph export features such as shortest-path transit and traversal counts
- dispatch-indexed node and arc observation export for graph-conditioned learning
- dispatch-time context builder for robot, task, and global observations
- policy API hook via `DispatchPolicy.select_assignment_from_context(...)`
- candidate observations shared by live policies and exported datasets
- congestion-sensitive observation fields derived from active reservations
- graph-aware candidate features derived from endpoint degrees and shortest-path structure
- PyG graph encoder consuming dispatch-time node and arc state

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

They also now support graph-aware structural signals such as:

- pickup and dropoff node in/out degree
- mean and max transit counts along pickup and dropoff paths
- mean and max arc traversal counts along pickup and dropoff paths

## Scope Limits

- the current graph model uses a global graph embedding plus candidate features
- it does not yet use endpoint-conditioned graph readouts or task/robot subgraph encoders
- it is still a dispatch scorer, not a full learned multi-robot coordination stack
