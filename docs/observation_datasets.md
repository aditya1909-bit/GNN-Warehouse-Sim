# Observation Datasets

## Purpose

This stage exports training-ready observation data from the existing baseline
simulator. It does not train a model. It captures what the simulator actually
saw at each dispatch decision so later learned policies can start from a stable,
auditable dataset contract.

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

## Design Rationale

- Export candidate pairs instead of only chosen assignments so later policies
  can be trained or evaluated against the full action set considered at a
  dispatch event.
- Keep graph features separate from dynamic dispatch rows because the graph is
  static within an experiment run.
- Reuse the same observation contracts already used by the policy API so the
  dataset format stays aligned with live simulation state.

## Current Limits

- No replay buffer or episode serialization
- No action masking beyond the exported candidate rows
- No tensor serialization format yet
- No learned-policy trainer or evaluator yet
