"""Offline learning configuration models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from warehouse_sim.config.models import ConfigValidationError


@dataclass(frozen=True)
class OfflineDatasetConfig:
    """Dataset-loading configuration for offline policy fitting."""

    source: Path
    feature_names: tuple[str, ...] | None = None
    node_feature_names: tuple[str, ...] | None = None
    edge_feature_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class OfflineSplitConfig:
    """Grouped split settings for offline policy fitting."""

    split_unit: str = "dispatch_group"
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    test_fraction: float = 0.15

    def __post_init__(self) -> None:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ConfigValidationError("offline split fractions must sum to 1.0.")


@dataclass(frozen=True)
class OfflineModelConfig:
    """Model and optimizer settings for offline policy fitting."""

    type: str
    learning_rate: float = 0.05
    max_epochs: int = 300
    l2_regularization: float = 1e-4
    patience: int = 25
    hidden_dim: int = 16
    message_passing_layers: int = 2
    dropout: float = 0.0
    batch_size: int = 8
    benchmark_weighting: bool = False

    def __post_init__(self) -> None:
        if self.type not in {"linear", "mlp", "graph_dispatch"}:
            raise ConfigValidationError("offline model.type must be one of: linear, mlp, graph_dispatch")


@dataclass(frozen=True)
class OfflineReportingConfig:
    """Output settings for offline policy fitting runs."""

    output_dir: Path


@dataclass(frozen=True)
class OfflineTrainingConfig:
    """Top-level configuration for offline model fitting and evaluation."""

    name: str
    seed: int
    dataset: OfflineDatasetConfig
    split: OfflineSplitConfig
    model: OfflineModelConfig
    reporting: OfflineReportingConfig

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigValidationError("offline training name must be non-empty.")
