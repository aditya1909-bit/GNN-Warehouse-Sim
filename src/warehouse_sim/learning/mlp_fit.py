"""Small grouped MLP baseline for offline dispatch candidate scoring."""

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
class GroupedMLPFitConfig:
    """Training hyperparameters for the nonlinear learned baseline."""

    hidden_dim: int = 16
    learning_rate: float = 0.01
    max_epochs: int = 400
    l2_regularization: float = 1e-4
    patience: int = 30
    seed: int = 0
    benchmark_weighting: bool = False

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be > 0.")
        if self.l2_regularization < 0:
            raise ValueError("l2_regularization must be >= 0.")
        if self.patience <= 0:
            raise ValueError("patience must be > 0.")


def fit_grouped_mlp_model(
    train_split: DatasetSplit,
    validation_split: DatasetSplit | None = None,
    config: GroupedMLPFitConfig | None = None,
) -> DispatchTrainingResult:
    """Fit a one-hidden-layer grouped ranking model over candidate features."""

    config = config or GroupedMLPFitConfig()
    validation_split = validation_split or train_split

    x_train_scaled, means, scales = standardize_feature_matrix(train_split.dataset.feature_matrix)
    y_train = train_split.dataset.labels.astype(float)
    train_groups = train_split.dataset.iter_group_indices()
    train_group_weights = train_split.dataset.group_weights() if config.benchmark_weighting else None

    x_validation_scaled = (validation_split.dataset.feature_matrix - means) / scales
    y_validation = validation_split.dataset.labels.astype(float)
    validation_groups = validation_split.dataset.iter_group_indices()
    validation_group_weights = (
        validation_split.dataset.group_weights() if config.benchmark_weighting else None
    )

    rng = np.random.default_rng(config.seed)
    input_dim = x_train_scaled.shape[1]
    hidden_weights = rng.normal(0.0, 0.25, size=(input_dim, config.hidden_dim))
    hidden_bias = np.zeros(config.hidden_dim, dtype=float)
    output_weights = rng.normal(0.0, 0.25, size=config.hidden_dim)
    output_bias = 0.0

    best_params = (
        hidden_weights.copy(),
        hidden_bias.copy(),
        output_weights.copy(),
        output_bias,
    )
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        train_loss, gradients = _mlp_loss_and_gradient(
            x_train_scaled,
            y_train,
            train_groups,
            train_group_weights,
            hidden_weights,
            hidden_bias,
            output_weights,
            output_bias,
            config.l2_regularization,
        )
        gradient_hidden_weights, gradient_hidden_bias, gradient_output_weights, gradient_output_bias = gradients
        hidden_weights -= config.learning_rate * gradient_hidden_weights
        hidden_bias -= config.learning_rate * gradient_hidden_bias
        output_weights -= config.learning_rate * gradient_output_weights
        output_bias -= config.learning_rate * gradient_output_bias

        validation_scores = _mlp_scores(
            x_validation_scaled,
            hidden_weights,
            hidden_bias,
            output_weights,
            output_bias,
        )
        validation_loss = grouped_softmax_loss(
            validation_scores,
            y_validation,
            validation_groups,
            validation_group_weights,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
            }
        )
        if validation_loss + 1e-9 < best_validation_loss:
            best_validation_loss = float(validation_loss)
            best_epoch = epoch
            stale_epochs = 0
            best_params = (
                hidden_weights.copy(),
                hidden_bias.copy(),
                output_weights.copy(),
                output_bias,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    best_hidden_weights, best_hidden_bias, best_output_weights, best_output_bias = best_params
    artifact = DispatchModelArtifact(
        artifact_version=2,
        model_type="grouped_mlp",
        objective="dispatch_group_softmax_cross_entropy",
        feature_names=train_split.dataset.feature_names,
        parameters={
            "normalization": {
                "means": means.tolist(),
                "scales": scales.tolist(),
            },
            "hidden_weights": best_hidden_weights.tolist(),
            "hidden_bias": best_hidden_bias.tolist(),
            "output_weights": best_output_weights.tolist(),
            "output_bias": float(best_output_bias),
        },
        metadata={
            "training": {
                "hidden_dim": config.hidden_dim,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "benchmark_weighting": config.benchmark_weighting,
            }
        },
    )
    return DispatchTrainingResult(
        artifact=artifact,
        training_history=tuple(history),
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        training_metadata={
            "hidden_dim": config.hidden_dim,
            "feature_means": means.tolist(),
            "feature_scales": scales.tolist(),
        },
    )


def _mlp_scores(
    feature_matrix: np.ndarray,
    hidden_weights: np.ndarray,
    hidden_bias: np.ndarray,
    output_weights: np.ndarray,
    output_bias: float,
) -> np.ndarray:
    hidden_linear = feature_matrix @ hidden_weights + hidden_bias
    hidden_activation = np.maximum(hidden_linear, 0.0)
    return hidden_activation @ output_weights + output_bias


def _mlp_loss_and_gradient(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    group_indices: tuple[np.ndarray, ...],
    group_weights: np.ndarray | None,
    hidden_weights: np.ndarray,
    hidden_bias: np.ndarray,
    output_weights: np.ndarray,
    output_bias: float,
    l2_regularization: float,
) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    gradient_hidden_weights = np.zeros_like(hidden_weights)
    gradient_hidden_bias = np.zeros_like(hidden_bias)
    gradient_output_weights = np.zeros_like(output_weights)
    gradient_output_bias = 0.0
    total_loss = 0.0
    weights_by_group = (
        np.asarray(group_weights, dtype=float)
        if group_weights is not None
        else np.ones(len(group_indices), dtype=float)
    )
    total_weight = 0.0

    for group_offset, indices in enumerate(group_indices):
        group_features = feature_matrix[indices]
        group_labels = labels[indices]
        hidden_linear = group_features @ hidden_weights + hidden_bias
        hidden_activation = np.maximum(hidden_linear, 0.0)
        scores = hidden_activation @ output_weights + output_bias
        shifted = scores - np.max(scores)
        exponentiated = np.exp(shifted)
        probabilities = exponentiated / np.sum(exponentiated)
        weight = float(weights_by_group[group_offset])
        total_loss += weight * -float(np.dot(group_labels, shifted - np.log(np.sum(exponentiated))))
        difference = probabilities - group_labels

        gradient_output_weights += weight * (hidden_activation.T @ difference)
        gradient_output_bias += weight * float(np.sum(difference))
        hidden_gradient = np.outer(difference, output_weights)
        relu_gradient = hidden_gradient * (hidden_linear > 0.0)
        gradient_hidden_weights += weight * (group_features.T @ relu_gradient)
        gradient_hidden_bias += weight * np.sum(relu_gradient, axis=0)
        total_weight += weight

    normalizer = max(total_weight, 1e-12)
    total_loss = total_loss / normalizer
    total_loss += 0.5 * l2_regularization * (
        float(np.sum(hidden_weights * hidden_weights)) + float(np.dot(output_weights, output_weights))
    )
    gradient_hidden_weights = gradient_hidden_weights / normalizer + l2_regularization * hidden_weights
    gradient_hidden_bias /= normalizer
    gradient_output_weights = gradient_output_weights / normalizer + l2_regularization * output_weights
    gradient_output_bias /= normalizer
    return (
        total_loss,
        (
            gradient_hidden_weights,
            gradient_hidden_bias,
            gradient_output_weights,
            gradient_output_bias,
        ),
    )
