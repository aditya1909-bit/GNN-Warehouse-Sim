# Interaction-Aware Execution

## Purpose

Stage 9 adds the next honest fidelity step for this repository: realized route execution with simplified shared-resource contention.

It is meant to answer questions like:

- what happens when several robots need the same corridor?
- how much delay comes from shared travel resources?
- how should dispatch heuristics react to visible congestion?

It is not full MAPF, global conflict-based search, or a continuous-motion simulator.

## Execution Models

The `[simulation]` config section supports:

- `execution_model = "idealized"`
- `execution_model = "reserved_edges"`
- `execution_model = "reserved_nodes"`

`idealized` preserves the earlier baseline exactly in spirit: travel is shortest-path time with no interaction penalty.

`reserved_edges` uses directed-edge reservations. A robot waits if its next edge segment is still reserved by another robot.

`reserved_nodes` uses single-node occupancy. A robot waits before entering a node that is still occupied, and pickup/dropoff service can extend that occupancy.

## Path Materialization

Each assignment now materializes:

- path from current node to pickup
- service at pickup
- path from pickup to dropoff

The execution record stores explicit node and arc sequences plus ideal travel time, realized travel time, and waiting introduced by contention.

## Reservation Behavior

The reservation logic is intentionally simple:

- shortest paths are planned once at assignment time
- reservations are applied in route order
- waiting is inserted locally when the next resource is unavailable
- tie behavior stays deterministic through stable ordering and deterministic policy evaluation

The system does not globally replan all robots after every conflict.

## Metrics And Outputs

The existing reporting files remain the same:

- `summary.json`
- `executions.csv`
- `queue_snapshots.csv`
- `robot_metrics.csv`

They now include congestion-sensitive fields such as:

- realized travel totals
- congestion delay totals
- blocked traversal counters
- route-level execution details in task execution rows

## Observation Layer Effects

Dispatch-time observations now expose a small congestion slice:

- active reserved edge count
- active reserved node count
- average robot time-until-available
- estimated pickup/dropoff congestion delay
- estimated pickup/dropoff blocked segments

These features are intentionally interpretable and are shared between live dispatch and dataset export.
