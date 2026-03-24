# Observation Datasets

## Purpose

This stage exports and consumes training-ready observation data from the existing baseline simulator. The learning pipeline still starts from what the simulator actually saw at each dispatch decision. It does not introduce a disconnected replay format or a separate policy interface.

## Files Written

When observation-dataset export is enabled, experiment runs now write:

- `graph_nodes.csv`: static node features
- `graph_arcs.csv`: directed travel arcs with distance and travel time
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

## Current Limits

- No replay buffer or episode serialization
- No action masking beyond the exported candidate rows
- No graph-tensor training format yet
- No graph-neural dispatch model
