"""Shared optimization helpers for offline dispatch scorers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from warehouse_sim.learning.artifacts import DispatchModelArtifact


@dataclass(frozen=True)
class DispatchTrainingResult:
    """Result of fitting one offline dispatch scorer."""

    artifact: DispatchModelArtifact
    training_history: tuple[dict[str, float], ...]
    best_epoch: int
    best_validation_loss: float
    training_metadata: dict[str, object] = field(default_factory=dict)


def standardize_feature_matrix(
    feature_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score feature columns with zero-variance protection."""

    means = feature_matrix.mean(axis=0)
    scales = feature_matrix.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardized = (feature_matrix - means) / scales
    return standardized, means, scales


def grouped_softmax_loss(
    scores: np.ndarray,
    labels: np.ndarray,
    group_indices: tuple[np.ndarray, ...],
) -> float:
    """Average grouped negative log-likelihood across dispatch events."""

    if not group_indices:
        return 0.0
    total = 0.0
    for indices in group_indices:
        group_scores = scores[indices]
        shifted = group_scores - np.max(group_scores)
        log_denom = float(np.log(np.sum(np.exp(shifted))))
        total += -float(np.dot(labels[indices], shifted - log_denom))
    return total / len(group_indices)
