# Offline Policy Fitting

## Scope

Stage 10 adds an offline fitting and evaluation pipeline for dispatch candidate scorers. It does not add a GNN, and it does not add RL. The learning target is still the dispatch-time candidate set exported by the simulator.

## Dataset Contract

The canonical offline input remains the exported observation dataset:

- `dispatch_observations.csv`: one row per candidate robot-task pair at a dispatch event
- `dataset_manifest.json`: run metadata plus file locations

Important columns:

- label: `is_selected`
- dispatch grouping: `dispatch_index`, promoted to `dispatch_group_id` with run metadata
- candidate ids: `candidate_robot_id`, `candidate_task_id`
- chosen ids: `selected_robot_id`, `selected_task_id`
- numeric features: the same named candidate features used by live dispatch scoring

The loader accepts:

- a single `dispatch_observations.csv`
- a single `dataset_manifest.json`
- a directory tree containing multiple manifests or dispatch CSVs

## Group-Aware Splits

Offline splits are grouped by dispatch event by default so rows from the same choice set never leak across train, validation, and test.

Supported split units:

- `dispatch_group`
- `run`
- `scenario`

That makes repeated-run or repeated-scenario evaluation possible without turning grouped decisions into i.i.d. rows.

## Implemented Models

The current fitted models are intentionally modest:

- grouped linear scorer trained with dispatch-group softmax cross-entropy
- one-hidden-layer MLP trained with the same grouped objective

Both models consume the same exported candidate features. Neither model is a GNN.

## Offline Metrics

Offline evaluation writes JSON summaries and CSV predictions for each split.

Primary grouped metrics:

- top-1 selection accuracy within dispatch groups
- mean reciprocal rank
- group log loss

Secondary candidate-level metrics:

- accuracy
- precision
- recall
- binary log loss

Linear artifacts also carry a learned weight summary for inspection.

## Artifact Format

Trained models are saved as JSON artifacts with:

- artifact version
- model type
- objective
- ordered feature names
- model parameters
- metadata such as learned weights or normalization statistics

## Live Simulator Integration

Trained artifacts can be loaded back into the simulator through experiment configs:

```toml
[simulation]
policy = "trained_linear_model"

[policy_model]
artifact_path = "artifacts/models/linear_dispatch_model.json"
```

Supported trained policy names:

- `trained_linear_model`
- `trained_mlp_model`

That means the simulator can now compare:

- heuristic policies
- hand-configured linear scoring
- fitted linear scoring
- one modest nonlinear learned baseline

inside the same experiment and benchmark stack.

## CLI Workflow

Fit a linear model:

```bash
python3 scripts/run_offline_policy_fitting.py train --config configs/offline_linear_fit.toml
```

Fit the MLP baseline:

```bash
python3 scripts/run_offline_policy_fitting.py train --config configs/offline_mlp_fit.toml
```

Evaluate an artifact:

```bash
PYTHONPATH=src python3 -m warehouse_sim.learning.cli evaluate \
  --artifact outputs/offline/offline_linear_dispatch_fit/model_artifact.json \
  --dataset outputs/linear_assignment_policy/dataset_manifest.json \
  --output-dir outputs/offline/evaluation
```

## Still Out Of Scope

- graph-neural dispatch models
- graph encoders over `graph_nodes.csv` and `graph_arcs.csv`
- RL
- claims of end-to-end learned multi-robot coordination
