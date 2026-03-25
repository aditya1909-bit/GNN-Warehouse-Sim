"""Offline evaluation for graph-conditioned dispatch scorers."""

from __future__ import annotations

import numpy as np
import torch

from warehouse_sim.learning.evaluation import OfflineEvaluationResult
from warehouse_sim.learning.graph_data import GraphDispatchDataset
from warehouse_sim.learning.graph_model import load_graph_dispatch_model


def evaluate_graph_dispatch_artifact(
    dataset: GraphDispatchDataset,
    artifact_path,
) -> OfflineEvaluationResult:
    """Evaluate a graph dispatch artifact on a graph dispatch dataset."""

    loaded = load_graph_dispatch_model(artifact_path)
    model = loaded.model
    model.eval()
    prediction_rows: list[dict[str, object]] = []
    top1_hits = 0
    reciprocal_ranks: list[float] = []
    selected_ranks: list[int] = []
    group_log_losses: list[float] = []
    candidate_log_losses: list[float] = []
    candidate_correct = 0
    candidate_total = 0

    with torch.no_grad():
        for example in dataset.examples:
            logits, _ = model(
                node_features=torch.tensor(example.node_features, dtype=torch.float32),
                edge_index=torch.tensor(example.edge_index, dtype=torch.long),
                edge_features=torch.tensor(example.edge_features, dtype=torch.float32),
                candidate_features=torch.tensor(example.candidate_features, dtype=torch.float32),
            )
            scores = logits.detach().cpu().numpy()
            probabilities = torch.softmax(logits, dim=0).detach().cpu().numpy()
            binary_probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            target_index = int(np.argmax(example.labels))
            ranked = np.argsort(-scores)
            rank = int(np.where(ranked == target_index)[0][0]) + 1
            top1_hits += 1 if rank == 1 else 0
            reciprocal_ranks.append(1.0 / rank)
            selected_ranks.append(rank)
            group_log_losses.append(-np.log(max(float(probabilities[target_index]), 1e-12)))
            candidate_log_losses.extend(
                [
                    -(
                        label * np.log(max(float(probability), 1e-12))
                        + (1 - label) * np.log(max(float(1.0 - probability), 1e-12))
                    )
                    for label, probability in zip(example.labels.tolist(), binary_probabilities.tolist(), strict=True)
                ]
            )
            predicted_positive = binary_probabilities >= 0.5
            candidate_correct += int(np.sum(predicted_positive == example.labels))
            candidate_total += len(example.labels)
            for index, ((robot_id, task_id), label, score, probability) in enumerate(
                zip(
                    zip(
                        example.metadata["candidate_robot_ids"],
                        example.metadata["candidate_task_ids"],
                        strict=True,
                    ),
                    example.labels.tolist(),
                    scores.tolist(),
                    probabilities.tolist(),
                    strict=True,
                )
            ):
                prediction_rows.append(
                    {
                        "dispatch_group_id": example.dispatch_group_id,
                        "dispatch_index": example.dispatch_index,
                        "candidate_robot_id": robot_id,
                        "candidate_task_id": task_id,
                        "is_selected": int(label),
                        "score": float(score),
                        "group_probability": float(probability),
                        "rank": int(np.where(ranked == index)[0][0]) + 1,
                    }
                )

    metrics: dict[str, float | int | None] = {
        "candidate_rows": sum(example.candidate_count for example in dataset.examples),
        "dispatch_groups": len(dataset.examples),
        "candidate_accuracy": candidate_correct / candidate_total if candidate_total else None,
        "candidate_precision": None,
        "candidate_recall": None,
        "candidate_log_loss": float(np.mean(candidate_log_losses)) if candidate_log_losses else None,
        "group_top_1_accuracy": top1_hits / len(dataset.examples) if dataset.examples else None,
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None,
        "mean_selected_rank": float(np.mean(selected_ranks)) if selected_ranks else None,
        "group_log_loss": float(np.mean(group_log_losses)) if group_log_losses else None,
        "parameter_count": int(loaded.artifact.metadata["training"]["parameter_count"]),
    }
    return OfflineEvaluationResult(metrics=metrics, prediction_rows=tuple(prediction_rows))
