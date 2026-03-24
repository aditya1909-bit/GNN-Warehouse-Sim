"""Grouped linear fitting for offline dispatch candidate scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from warehouse_sim.learning.artifacts import DispatchModelArtifact
from warehouse_sim.learning.splits import DatasetSplit
from warehouse_sim.learning.training import (
    DispatchTrainingResult,
    grouped_softmax_loss,
    standardize_feature_matrix,
)


@dataclass(frozen=True)
class GroupedLinearFitConfig:
    """Training hyperparameters for the fitted linear scorer."""

    learning_rate: float = 0.05
    max_epochs: int = 300
    l2_regularization: float = 1e-4
    patience: int = 25

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be > 0.")
        if self.l2_regularization < 0:
            raise ValueError("l2_regularization must be >= 0.")
        if self.patience <= 0:
            raise ValueError("patience must be > 0.")


def fit_grouped_linear_model(
    train_split: DatasetSplit,
    validation_split: DatasetSplit | None = None,
    config: GroupedLinearFitConfig | None = None,
) -> DispatchTrainingResult:
    """Fit a grouped-softmax linear scorer over candidate features."""

    config = config or GroupedLinearFitConfig()
    validation_split = validation_split or train_split

    x_train_scaled, means, scales = standardize_feature_matrix(train_split.dataset.feature_matrix)
    y_train = train_split.dataset.labels.astype(float)
    train_groups = train_split.dataset.iter_group_indices()

    x_validation_scaled = (validation_split.dataset.feature_matrix - means) / scales
    y_validation = validation_split.dataset.labels.astype(float)
    validation_groups = validation_split.dataset.iter_group_indices()

    weights = np.zeros(x_train_scaled.shape[1], dtype=float)
    bias = 0.0
    best_weights = weights.copy()
    best_bias = bias
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        train_loss, gradient_weights, gradient_bias = _linear_loss_and_gradient(
            x_train_scaled,
            y_train,
            train_groups,
            weights,
            bias,
            config.l2_regularization,
        )
        weights -= config.learning_rate * gradient_weights
        bias -= config.learning_rate * gradient_bias

        validation_scores = x_validation_scaled @ weights + bias
        validation_loss = grouped_softmax_loss(validation_scores, y_validation, validation_groups)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
            }
        )
        if validation_loss + 1e-9 < best_validation_loss:
            best_validation_loss = float(validation_loss)
            best_weights = weights.copy()
            best_bias = bias
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    raw_weights = best_weights / scales
    raw_bias = float(best_bias - np.dot(means / scales, best_weights))
    artifact = DispatchModelArtifact(
        artifact_version=1,
        model_type="grouped_linear",
        objective="dispatch_group_softmax_cross_entropy",
        feature_names=train_split.dataset.feature_names,
        parameters={
            "weights": raw_weights.tolist(),
            "bias": raw_bias,
        },
        metadata={
            "training": {
                "feature_means": means.tolist(),
                "feature_scales": scales.tolist(),
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
            },
            "weight_summary": {
                feature_name: float(weight)
                for feature_name, weight in sorted(
                    zip(train_split.dataset.feature_names, raw_weights.tolist(), strict=True),
                    key=lambda item: abs(item[1]),
                    reverse=True,
                )
            },
        },
    )
    return DispatchTrainingResult(
        artifact=artifact,
        training_history=tuple(history),
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        training_metadata={
            "feature_means": means.tolist(),
            "feature_scales": scales.tolist(),
        },
    )


def _linear_loss_and_gradient(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    group_indices: tuple[np.ndarray, ...],
    weights: np.ndarray,
    bias: float,
    l2_regularization: float,
) -> tuple[float, np.ndarray, float]:
    scores = feature_matrix @ weights + bias
    gradients = np.zeros_like(weights)
    gradient_bias = 0.0
    total_loss = 0.0

    for indices in group_indices:
        group_features = feature_matrix[indices]
        group_labels = labels[indices]
        group_scores = scores[indices]
        shifted = group_scores - np.max(group_scores)
        exponentiated = np.exp(shifted)
        probabilities = exponentiated / np.sum(exponentiated)
        total_loss += -float(np.dot(group_labels, shifted - np.log(np.sum(exponentiated))))
        difference = probabilities - group_labels
        gradients += group_features.T @ difference
        gradient_bias += float(np.sum(difference))

    normalizer = max(len(group_indices), 1)
    total_loss = total_loss / normalizer + 0.5 * l2_regularization * float(np.dot(weights, weights))
    gradients = gradients / normalizer + l2_regularization * weights
    gradient_bias /= normalizer
    return total_loss, gradients, gradient_bias
