# Domain Model

## Stage 2 Scope

Stage 2 introduces explicit warehouse domain primitives without claiming a full simulator exists.

- `WarehouseGraph` stores nodes, weighted edges, and shortest-path utilities.
- `SyntheticGridLayoutConfig` and `build_synthetic_grid_layout()` create reproducible baseline layouts.
- `WarehouseEnvironment` groups graph nodes into named zones.
- `Task` captures the minimal correct task specification for warehouse experiments.
- `TaskQueue` provides release-time-aware FIFO queue behavior.
- `tasks_from_demand_records()` bridges the validated demand layer into explicit task objects.

## Graph Layer

The graph layer is intentionally small but technically serious.

- Nodes have ids, coordinates, node types, and optional zone labels.
- Edges store both distance and travel time.
- Travel can be undirected or explicitly one-way on selected grid edges.
- Obstacles can be represented as isolated obstacle nodes or omitted via blocked cells.
- Shortest-path queries support both `distance` and `travel_time` weights.

## Environment Layer

The environment wraps the topology and names logical areas of the warehouse.

- Zones are explicit named node sets.
- Zones can be derived from node metadata or passed in directly.
- Stage 2 chooses the lexicographically first node in a zone as the default representative for adapters.
- Later stages can add richer node-selection and routing policies without changing the core graph API.

## Task Layer

Tasks are now explicit domain objects rather than implicit CSV rows.

- Required fields: `task_id`, `release_time`, `pickup_node`, `dropoff_node`
- Optional operational fields: `task_type`, `priority`, `service_time_estimate`, `due_time`
- Optional context fields: `source_zone`, `destination_zone`, `metadata`

The current queue is intentionally conservative.

- FIFO ordering is preserved within equal release times.
- Tasks are not released until `current_time >= release_time`.
- Assignment, execution, and completion state will belong to the simulation layer, not the immutable task specification.

## Simulation Layer Added In Stage 3

The simulation layer now consumes these domain objects directly.

- `RobotSpec` defines initial robot placement and speed scaling.
- `run_simulation()` executes a discrete-event baseline.
- `TaskExecution` records assignment, travel, and completion timing.
- `SimulationMetrics` summarizes throughput, waiting time, turnaround time, utilization, and queue behavior.

## What Is Still Missing

The repository still does not include:

- collision-aware movement or congestion effects
- battery/charging dynamics beyond the node type vocabulary
- advanced dispatch heuristics
- learning-based policies or GNN featurization/training
