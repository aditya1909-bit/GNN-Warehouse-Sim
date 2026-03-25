# Observation Datasets

## Purpose

This stage exports and consumes training-ready observation data from the existing baseline simulator. The learning pipeline still starts from what the simulator actually saw at each dispatch decision. It does not introduce a disconnected replay format or a separate policy interface.

## Files Written

When observation-dataset export is enabled, experiment runs now write:

- `graph_nodes.csv`: static node features
- `graph_arcs.csv`: directed travel arcs with distance and travel time
- `dispatch_node_observations.csv`: one row per dispatch event and node with dynamic graph-state features
- `dispatch_arc_observations.csv`: one row per dispatch event and directed arc with dynamic reservation-state features
- `dispatch_observations.csv`: one row per idle-robot and ready-task candidate
  pair at each dispatch event, labeled with `is_selected`
- `dataset_manifest.json`: dataset metadata and file manifest

## Dispatch Observation Schema

Each dispatch row includes:

- dispatch index and decision time
- selected robot/task ids
- candidate robot/task ids
- `is_selected` label
- robot state counters and location
- task timing, zones, and service estimate
- robot-to-pickup and pickup-to-dropoff travel estimates
- global queue and fleet counts at the decision time
- graph-aware endpoint and shortest-path structure features
- execution-model and reservation summary fields
- estimated congestion delay and blocked-segment features for candidate routes

## Design Rationale

- Export candidate pairs instead of only chosen assignments so later policies
  can be trained or evaluated against the full action set considered at a
  dispatch event.
- Keep graph features separate from dynamic dispatch rows because the graph is
  static within an experiment run.
- Reuse the same observation contracts already used by the policy API so the
  dataset format stays aligned with live simulation state.
- Keep dataset manifests lightweight and explicit so multiple runs can be
  combined later for grouped splitting by dispatch event, run, or scenario.

## Offline Learning Contract

Stage 10 builds the canonical offline loader around this dispatch-row contract:

- labels: `is_selected`
- dispatch grouping: `dispatch_group_id` derived from run metadata plus `dispatch_index`
- metadata: ids, nodes, zones, timing, scenario/run metadata
- numeric candidate features: the same named feature set used by live scoring policies

The loader accepts a single `dispatch_observations.csv`, a single
`dataset_manifest.json`, or a directory tree containing multiple manifests.
That makes it possible to fit or evaluate a model across repeated runs without
inventing a second dataset format.

The static graph files now also carry lightweight structural counts:

- `graph_nodes.csv`: node coordinates, type, degree, and shortest-path transit count
- `graph_arcs.csv`: directed travel arcs plus shortest-path traversal count

The dispatch-indexed graph tables carry the dynamic state used by the graph-conditioned model:

- `dispatch_node_observations.csv`: robot occupancy, ready-task endpoints, reservation status, and reserved-time remaining
- `dispatch_arc_observations.csv`: reservation status and reserved-time remaining by directed arc

## Current Limits

- No full episode replay buffer beyond the dispatch-indexed tables
- No end-to-end warehouse-control action space beyond dispatch candidate choice
- The graph-conditioned model uses global graph pooling rather than richer endpoint-conditioned readouts
