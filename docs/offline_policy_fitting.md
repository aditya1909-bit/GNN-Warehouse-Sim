# Offline Policy Fitting

## Scope

Stage 10 added offline fitting and evaluation for dispatch candidate scorers. Stage 11 extends that pipeline with a graph-conditioned message-passing dispatch model and optional masked PPO fine-tuning. The learning target is still the dispatch-time candidate set exported by the simulator.

## Dataset Contract

The canonical offline input remains the exported observation dataset:

- `dispatch_observations.csv`: one row per candidate robot-task pair at a dispatch event
- `dispatch_node_observations.csv`: one row per dispatch event and node with dynamic graph-state features
- `dispatch_arc_observations.csv`: one row per dispatch event and directed arc with dynamic reservation-state features
- `graph_nodes.csv`: static node features
- `graph_arcs.csv`: static directed arc features
- `dataset_manifest.json`: run metadata plus file locations

Important columns:

- label: `is_selected`
- dispatch grouping: `dispatch_index`, promoted to `dispatch_group_id` with run metadata
- candidate ids: `candidate_robot_id`, `candidate_task_id`
- chosen ids: `selected_robot_id`, `selected_task_id`
- numeric features: the same named candidate features used by live dispatch scoring

For graph-conditioned fitting, the loader joins:

- static graph topology from `graph_nodes.csv` and `graph_arcs.csv`
- dispatch-time node state from `dispatch_node_observations.csv`
- dispatch-time arc state from `dispatch_arc_observations.csv`
- candidate features and labels from `dispatch_observations.csv`

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
- PyG graph-conditioned dispatch scorer with directed message passing, global mean pooling, and a candidate-scoring head

The linear and MLP baselines consume only candidate features. The graph-conditioned model also consumes dispatch-time graph tensors reconstructed from the exported node and arc tables.

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

Graph-model reports also include the configured node, edge, and candidate feature sets plus the learned parameter count.

## Artifact Format

Trained models are saved as JSON artifacts with:

- artifact version
- model type
- objective
- ordered feature names
- model parameters
- metadata such as learned weights or normalization statistics

Graph-conditioned artifacts save the JSON manifest plus a separate PyTorch state-dict file.

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
- `trained_graph_dispatch_model`

That means the simulator can now compare:

- heuristic policies
- hand-configured linear scoring
- fitted linear scoring
- one modest nonlinear learned baseline
- graph-conditioned message-passing dispatch scoring

inside the same experiment and benchmark stack.

## RL Fine-Tuning

Stage 11 also adds an optional dispatch-event RL wrapper and a compact masked PPO fine-tuning loop.

Important scope limits:

- one RL step is one dispatch decision
- action space is still the valid candidate set for that dispatch event
- PPO initializes from the pretrained graph scorer rather than training from scratch
- the learned policy still scores dispatch candidates; it is not a MAPF solver or an end-to-end warehouse controller

Default shaped reward:

- `+1.0 * tasks_completed_delta`
- `-0.01 * waiting_time_delta`
- `-0.02 * congestion_delay_delta`
- `-0.05 * blocked_traversal_events_delta`

## CLI Workflow

Fit a linear model:

```bash
python3 scripts/run_offline_policy_fitting.py train --config configs/offline_linear_fit.toml
```

Fit the MLP baseline:

```bash
python3 scripts/run_offline_policy_fitting.py train --config configs/offline_mlp_fit.toml
```

Fit the graph-conditioned scorer:

```bash
python3 scripts/run_offline_policy_fitting.py train --config configs/offline_graph_dispatch_fit.toml
```

Fine-tune with PPO:

```bash
python3 scripts/run_offline_policy_fitting.py train-rl --config configs/graph_dispatch_rl_fine_tune.toml
```

Evaluate an artifact:

```bash
PYTHONPATH=src python3 -m warehouse_sim.learning.cli evaluate \
  --artifact outputs/offline/offline_linear_dispatch_fit/model_artifact.json \
  --dataset outputs/linear_assignment_policy/dataset_manifest.json \
  --output-dir outputs/offline/evaluation
```

## Still Out Of Scope

- full MAPF or joint multi-robot path planning
- end-to-end learned warehouse coordination
- claims that the current RL loop solves full warehouse control beyond dispatch decisions
