"""Offline evaluation helpers for grouped dispatch candidate scorers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from warehouse_sim.learning.artifacts import DispatchModelArtifact
from warehouse_sim.learning.datasets import DispatchObservationDataset


@dataclass(frozen=True)
class OfflineEvaluationResult:
    """Summary metrics and per-candidate predictions for an offline evaluation pass."""

    metrics: dict[str, float | int | None]
    prediction_rows: tuple[dict[str, object], ...]


def evaluate_dispatch_model(
    dataset: DispatchObservationDataset,
    artifact: DispatchModelArtifact,
) -> OfflineEvaluationResult:
    """Evaluate a fitted scorer against grouped dispatch choices."""

    if dataset.row_count == 0:
        return OfflineEvaluationResult(
            metrics={
                "candidate_rows": 0,
                "dispatch_groups": 0,
                "candidate_accuracy": None,
                "candidate_precision": None,
                "candidate_recall": None,
                "candidate_log_loss": None,
                "group_top_1_accuracy": None,
                "mean_reciprocal_rank": None,
                "mean_selected_rank": None,
                "group_log_loss": None,
            },
            prediction_rows=(),
        )

    feature_matrix = _select_feature_matrix(dataset, artifact.feature_names)
    scores = artifact.score_matrix(feature_matrix)
    group_probabilities = _group_softmax_probabilities(scores, dataset.group_ids)
    candidate_probabilities = _sigmoid(scores)

    group_indices = dataset.iter_group_indices()
    reciprocal_ranks: list[float] = []
    top1_hits = 0
    group_log_losses: list[float] = []
    selected_ranks: list[int] = []

    for indices in group_indices:
        group_scores = scores[indices]
        group_labels = dataset.labels[indices]
        positive_index = int(np.argmax(group_labels))
        ordered = np.argsort(-group_scores)
        rank = int(np.where(ordered == positive_index)[0][0]) + 1
        reciprocal_ranks.append(1.0 / rank)
        top1_hits += 1 if rank == 1 else 0
        selected_ranks.append(rank)
        selected_probability = float(group_probabilities[indices][positive_index])
        group_log_losses.append(-np.log(max(selected_probability, 1e-12)))

    predicted_positive = candidate_probabilities >= 0.5
    true_positive = dataset.labels == 1
    tp = int(np.sum(predicted_positive & true_positive))
    fp = int(np.sum(predicted_positive & ~true_positive))
    fn = int(np.sum(~predicted_positive & true_positive))
    tn = int(np.sum(~predicted_positive & ~true_positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / len(dataset.labels) if len(dataset.labels) else 0.0
    binary_log_loss = -np.mean(
        dataset.labels * np.log(np.clip(candidate_probabilities, 1e-12, 1.0))
        + (1 - dataset.labels) * np.log(np.clip(1.0 - candidate_probabilities, 1e-12, 1.0))
    )

    prediction_rows = tuple(
        {
            "dispatch_group_id": dataset.metadata["dispatch_group_id"][row_index],
            "dispatch_index": dataset.metadata["dispatch_index"][row_index],
            "decision_time": dataset.metadata["decision_time"][row_index],
            "candidate_robot_id": dataset.metadata["candidate_robot_id"][row_index],
            "candidate_action_type": dataset.metadata["candidate_action_type"][row_index],
            "candidate_task_id": dataset.metadata["candidate_task_id"][row_index],
            "candidate_charging_node_id": dataset.metadata["candidate_charging_node_id"][row_index],
            "selected_robot_id": dataset.metadata["selected_robot_id"][row_index],
            "selected_action_type": dataset.metadata["selected_action_type"][row_index],
            "selected_task_id": dataset.metadata["selected_task_id"][row_index],
            "selected_charging_node_id": dataset.metadata["selected_charging_node_id"][row_index],
            "is_selected": int(dataset.labels[row_index]),
            "score": float(scores[row_index]),
            "candidate_probability": float(candidate_probabilities[row_index]),
            "group_probability": float(group_probabilities[row_index]),
        }
        for row_index in range(dataset.row_count)
    )

    metrics: dict[str, float | int | None] = {
        "candidate_rows": dataset.row_count,
        "dispatch_groups": dataset.group_count,
        "candidate_accuracy": accuracy,
        "candidate_precision": precision,
        "candidate_recall": recall,
        "candidate_log_loss": float(binary_log_loss),
        "group_top_1_accuracy": top1_hits / len(group_indices) if group_indices else 0.0,
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "mean_selected_rank": float(np.mean(selected_ranks)) if selected_ranks else None,
        "group_log_loss": float(np.mean(group_log_losses)) if group_log_losses else None,
    }
    return OfflineEvaluationResult(metrics=metrics, prediction_rows=prediction_rows)


def write_offline_evaluation_report(
    output_dir: Path,
    split_name: str,
    result: OfflineEvaluationResult,
) -> dict[str, Path]:
    """Write JSON summary and per-candidate predictions for one dataset split."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{split_name}_evaluation.json"
    predictions_path = output_dir / f"{split_name}_predictions.csv"

    summary_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    _write_csv(predictions_path, list(result.prediction_rows))
    return {
        f"{split_name}_evaluation": summary_path,
        f"{split_name}_predictions": predictions_path,
    }


def _select_feature_matrix(
    dataset: DispatchObservationDataset,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    index_by_name = {name: index for index, name in enumerate(dataset.feature_names)}
    missing = [name for name in feature_names if name not in index_by_name]
    if missing:
        raise ValueError(f"Dataset is missing artifact features: {missing}")
    indices = np.asarray([index_by_name[name] for name in feature_names], dtype=int)
    return dataset.feature_matrix[:, indices]


def _group_softmax_probabilities(scores: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    probabilities = np.zeros_like(scores, dtype=float)
    grouped_positions: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids.tolist()):
        grouped_positions.setdefault(str(group_id), []).append(index)
    for indices in grouped_positions.values():
        group_scores = scores[indices]
        shifted = group_scores - np.max(group_scores)
        exponentiated = np.exp(shifted)
        probabilities[np.asarray(indices, dtype=int)] = exponentiated / np.sum(exponentiated)
    return probabilities


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
