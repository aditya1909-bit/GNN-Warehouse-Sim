"""TOML loader for offline learning configuration manifests."""

from __future__ import annotations

import tomllib
from pathlib import Path

from warehouse_sim.config.learning_models import (
    OfflineDatasetConfig,
    OfflineModelConfig,
    OfflineReportingConfig,
    OfflineSplitConfig,
    OfflineTrainingConfig,
)


def load_offline_training_config(path: Path) -> OfflineTrainingConfig:
    """Load an offline fitting config from TOML."""

    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    dataset = raw["dataset"]
    split = raw.get("split", {})
    model = raw["model"]
    reporting = raw.get("reporting", {})

    dataset_source = (path.parent / Path(str(dataset["source"]))).resolve()
    output_dir = (path.parent / Path(str(reporting.get("output_dir", "outputs/offline_training")))).resolve()

    return OfflineTrainingConfig(
        name=str(raw["name"]),
        seed=int(raw.get("seed", 0)),
        dataset=OfflineDatasetConfig(
            source=dataset_source,
            feature_names=None
            if dataset.get("feature_names") is None
            else tuple(str(item) for item in dataset["feature_names"]),
        ),
        split=OfflineSplitConfig(
            split_unit=str(split.get("split_unit", "dispatch_group")),
            train_fraction=float(split.get("train_fraction", 0.7)),
            validation_fraction=float(split.get("validation_fraction", 0.15)),
            test_fraction=float(split.get("test_fraction", 0.15)),
        ),
        model=OfflineModelConfig(
            type=str(model["type"]),
            learning_rate=float(model.get("learning_rate", 0.05)),
            max_epochs=int(model.get("max_epochs", 300)),
            l2_regularization=float(model.get("l2_regularization", 1e-4)),
            patience=int(model.get("patience", 25)),
            hidden_dim=int(model.get("hidden_dim", 16)),
        ),
        reporting=OfflineReportingConfig(output_dir=output_dir),
    )
