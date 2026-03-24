"""Grouping-aware train/validation/test splits for dispatch datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from warehouse_sim.learning.datasets import DispatchObservationDataset
from warehouse_sim.learning.features import SUPPORTED_SPLIT_UNITS


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for deterministic grouped dataset splits."""

    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    split_unit: str = "dispatch_group"
    seed: int = 0

    def __post_init__(self) -> None:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if self.split_unit not in SUPPORTED_SPLIT_UNITS:
            raise ValueError(f"split_unit must be one of {SUPPORTED_SPLIT_UNITS}")
        if self.train_fraction <= 0:
            raise ValueError("train_fraction must be > 0.")
        if self.validation_fraction < 0 or self.test_fraction < 0:
            raise ValueError("validation_fraction and test_fraction must be >= 0.")
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Split fractions must sum to 1.0.")


@dataclass(frozen=True)
class DatasetSplit:
    """One named split of a dispatch-observation dataset."""

    name: str
    dataset: DispatchObservationDataset
    indices: np.ndarray
    split_units: tuple[str, ...]

    @property
    def row_count(self) -> int:
        return int(self.indices.size)

    @property
    def group_count(self) -> int:
        return len(tuple(dict.fromkeys(self.dataset.group_ids.tolist())))


@dataclass(frozen=True)
class DatasetSplits:
    """Full train/validation/test split bundle."""

    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit
    config: SplitConfig


def split_dispatch_observation_dataset(
    dataset: DispatchObservationDataset,
    config: SplitConfig | None = None,
) -> DatasetSplits:
    """Split a dataset without leaking rows across the chosen grouping unit."""

    config = config or SplitConfig()
    split_values = dataset.split_values(config.split_unit)
    unique_units = list(dict.fromkeys(str(value) for value in split_values.tolist()))
    if not unique_units:
        raise ValueError("Cannot split an empty dataset.")

    rng = np.random.default_rng(config.seed)
    shuffled_units = list(unique_units)
    rng.shuffle(shuffled_units)
    train_units, validation_units, test_units = _partition_units(shuffled_units, config)

    return DatasetSplits(
        train=_build_named_split("train", dataset, split_values, train_units),
        validation=_build_named_split("validation", dataset, split_values, validation_units),
        test=_build_named_split("test", dataset, split_values, test_units),
        config=config,
    )


def _partition_units(
    units: list[str],
    config: SplitConfig,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    count = len(units)
    train_count = int(round(count * config.train_fraction))
    validation_count = int(round(count * config.validation_fraction))
    test_count = count - train_count - validation_count

    if count >= 3:
        if validation_count == 0 and config.validation_fraction > 0:
            validation_count = 1
            train_count = max(train_count - 1, 1)
        if test_count == 0 and config.test_fraction > 0:
            test_count = 1
            if train_count > validation_count and train_count > 1:
                train_count -= 1
            elif validation_count > 1:
                validation_count -= 1
    elif count == 2:
        train_count = 1
        validation_count = 1 if config.validation_fraction > 0 else 0
        test_count = 1 if config.validation_fraction == 0 and config.test_fraction > 0 else 0
    else:
        train_count = 1
        validation_count = 0
        test_count = 0

    while train_count + validation_count + test_count > count:
        if train_count >= validation_count and train_count >= test_count and train_count > 1:
            train_count -= 1
        elif validation_count >= test_count and validation_count > 0:
            validation_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break
    while train_count + validation_count + test_count < count:
        train_count += 1

    return (
        tuple(units[:train_count]),
        tuple(units[train_count : train_count + validation_count]),
        tuple(units[train_count + validation_count : train_count + validation_count + test_count]),
    )


def _build_named_split(
    name: str,
    dataset: DispatchObservationDataset,
    split_values: np.ndarray,
    units: tuple[str, ...],
) -> DatasetSplit:
    unit_set = set(units)
    indices = np.asarray(
        [index for index, split_value in enumerate(split_values.tolist()) if str(split_value) in unit_set],
        dtype=int,
    )
    return DatasetSplit(
        name=name,
        dataset=dataset.subset(indices),
        indices=indices,
        split_units=units,
    )
