"""Reusable loaders for exported dispatch-observation datasets."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from warehouse_sim.learning.features import (
    LABEL_COLUMN,
    METADATA_COLUMNS,
    REQUIRED_DISPATCH_COLUMNS,
    candidate_feature_names_from_columns,
    validate_candidate_feature_names,
)
from warehouse_sim.learning.objective_weighting import benchmark_weight_from_row


def load_dispatch_observation_dataset(
    source: Path,
    feature_names: tuple[str, ...] | list[str] | None = None,
) -> "DispatchObservationDataset":
    """Load one or more exported observation datasets into a reusable table."""

    resolved_source = source.resolve()
    source_entries = _resolve_dataset_sources(resolved_source)
    raw_rows: list[dict[str, object]] = []

    for dataset_index, entry in enumerate(source_entries):
        manifest_payload = entry.manifest_payload or {}
        if manifest_payload and int(manifest_payload.get("dataset_schema_version", 0)) != 2:
            raise ValueError(
                f"Dispatch dataset manifest {entry.manifest_path} is not schema version 2."
            )
        dispatch_rows = _read_csv_rows(entry.dispatch_observations_path)
        dataset_id = str(
            manifest_payload.get("run_id")
            or manifest_payload.get("dataset_id")
            or f"dataset_{dataset_index}"
        )
        run_id = str(manifest_payload.get("run_id") or dataset_id)
        scenario_name = str(
            manifest_payload.get("scenario_name")
            or manifest_payload.get("experiment_name")
            or dataset_id
        )
        experiment_name = str(manifest_payload.get("experiment_name") or scenario_name)
        source_policy_name = str(manifest_payload.get("policy_name") or "unknown_policy")
        demand_seed = _optional_int(manifest_payload.get("demand_seed"))
        source_manifest_path = "" if entry.manifest_path is None else str(entry.manifest_path)

        for row in dispatch_rows:
            dispatch_index = int(row["dispatch_index"])
            scenario_seed = (
                f"{scenario_name}::seed_{demand_seed}"
                if demand_seed is not None
                else f"{scenario_name}::{run_id}"
            )
            enriched = dict(row)
            enriched.update(
                {
                    "dataset_id": dataset_id,
                    "run_id": run_id,
                    "scenario_name": scenario_name,
                    "experiment_name": experiment_name,
                    "source_policy_name": source_policy_name,
                    "demand_seed": demand_seed,
                    "scenario_seed": scenario_seed,
                    "source_manifest_path": source_manifest_path,
                    "benchmark_weight": benchmark_weight_from_row(row),
                    "dispatch_group_id": f"{run_id}::dispatch_{dispatch_index}",
                }
            )
            raw_rows.append(enriched)

    if not raw_rows:
        raise ValueError(f"No dispatch observations found under {resolved_source}")

    columns = tuple(raw_rows[0].keys())
    _validate_required_columns(columns)
    selected_features = (
        candidate_feature_names_from_columns(columns)
        if feature_names is None
        else validate_candidate_feature_names(feature_names)
    )
    missing_features = [name for name in selected_features if name not in raw_rows[0]]
    if missing_features:
        raise ValueError(f"Missing requested feature columns: {missing_features}")

    feature_matrix = np.asarray(
        [
            [float(row[feature_name]) for feature_name in selected_features]
            for row in raw_rows
        ],
        dtype=float,
    )
    labels = np.asarray([1 if bool(row[LABEL_COLUMN]) else 0 for row in raw_rows], dtype=int)
    group_ids = np.asarray([str(row["dispatch_group_id"]) for row in raw_rows], dtype=object)

    metadata_columns = tuple(column for column in METADATA_COLUMNS if column in raw_rows[0])
    metadata = {
        column: tuple(row.get(column) for row in raw_rows)
        for column in metadata_columns
    }

    return DispatchObservationDataset(
        source=resolved_source,
        row_count=len(raw_rows),
        feature_names=selected_features,
        metadata_columns=metadata_columns,
        feature_matrix=feature_matrix,
        labels=labels,
        group_ids=group_ids,
        metadata=metadata,
    )


@dataclass(frozen=True)
class DispatchObservationDataset:
    """In-memory table of candidate rows grouped by dispatch events."""

    source: Path
    row_count: int
    feature_names: tuple[str, ...]
    metadata_columns: tuple[str, ...]
    feature_matrix: np.ndarray
    labels: np.ndarray
    group_ids: np.ndarray
    metadata: dict[str, tuple[object, ...]]

    @property
    def dispatch_groups(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(group_id) for group_id in self.group_ids))

    @property
    def group_count(self) -> int:
        return len(self.dispatch_groups)

    def split_values(self, split_unit: str) -> np.ndarray:
        """Return the split-unit values for each row."""

        if split_unit == "dispatch_group":
            return self.group_ids
        if split_unit == "run":
            return np.asarray(self.metadata["run_id"], dtype=object)
        if split_unit == "scenario":
            return np.asarray(self.metadata["scenario_name"], dtype=object)
        if split_unit == "scenario_seed":
            return np.asarray(self.metadata["scenario_seed"], dtype=object)
        raise ValueError(f"Unsupported split unit: {split_unit}")

    def subset(self, indices: np.ndarray) -> "DispatchObservationDataset":
        """Return a row-filtered view of the dataset."""

        return DispatchObservationDataset(
            source=self.source,
            row_count=int(indices.size),
            feature_names=self.feature_names,
            metadata_columns=self.metadata_columns,
            feature_matrix=self.feature_matrix[indices],
            labels=self.labels[indices],
            group_ids=self.group_ids[indices],
            metadata={
                column: tuple(values[index] for index in indices.tolist())
                for column, values in self.metadata.items()
            },
        )

    def iter_group_indices(self) -> tuple[np.ndarray, ...]:
        """Return row indices grouped by dispatch event."""

        positions: dict[str, list[int]] = {}
        for index, group_id in enumerate(self.group_ids.tolist()):
            positions.setdefault(str(group_id), []).append(index)
        return tuple(np.asarray(indices, dtype=int) for indices in positions.values())

    def group_weights(self) -> np.ndarray:
        """Return one deployment-aware weight per dispatch group in group order."""

        weights = []
        for indices in self.iter_group_indices():
            selected_index = next(
                (int(index) for index in indices.tolist() if int(self.labels[index]) == 1),
                int(indices[0]),
            )
            if "benchmark_weight" in self.metadata:
                weights.append(float(self.metadata["benchmark_weight"][selected_index]))
            else:
                weights.append(1.0)
        return np.asarray(weights, dtype=float)


@dataclass(frozen=True)
class _DatasetSource:
    dispatch_observations_path: Path
    manifest_path: Path | None
    manifest_payload: dict[str, object] | None


def _resolve_dataset_sources(source: Path) -> tuple[_DatasetSource, ...]:
    if source.is_file():
        if source.name == "dataset_manifest.json":
            payload = json.loads(source.read_text(encoding="utf-8"))
            dispatch_name = str(payload["files"]["dispatch_observations"])
            return (
                _DatasetSource(
                    dispatch_observations_path=(source.parent / dispatch_name).resolve(),
                    manifest_path=source,
                    manifest_payload=payload,
                ),
            )
        if source.suffix == ".csv":
            return (_DatasetSource(dispatch_observations_path=source, manifest_path=None, manifest_payload=None),)
        raise ValueError(f"Unsupported dataset source file: {source}")

    if not source.is_dir():
        raise ValueError(f"Dataset source does not exist: {source}")

    manifest_paths = tuple(sorted(source.glob("**/dataset_manifest.json")))
    if manifest_paths:
        resolved_sources: list[_DatasetSource] = []
        for manifest_path in manifest_paths:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            dispatch_name = str(payload["files"]["dispatch_observations"])
            resolved_sources.append(
                _DatasetSource(
                    dispatch_observations_path=(manifest_path.parent / dispatch_name).resolve(),
                    manifest_path=manifest_path.resolve(),
                    manifest_payload=payload,
                )
            )
        return tuple(resolved_sources)

    direct_dispatch = source / "dispatch_observations.csv"
    direct_manifest = source / "dataset_manifest.json"
    if direct_dispatch.exists():
        payload = json.loads(direct_manifest.read_text(encoding="utf-8")) if direct_manifest.exists() else None
        return (
            _DatasetSource(
                dispatch_observations_path=direct_dispatch.resolve(),
                manifest_path=direct_manifest.resolve() if direct_manifest.exists() else None,
                manifest_payload=payload,
            ),
        )

    dispatch_paths = tuple(sorted(source.glob("**/dispatch_observations.csv")))
    if dispatch_paths:
        return tuple(
            _DatasetSource(
                dispatch_observations_path=dispatch_path.resolve(),
                manifest_path=None,
                manifest_payload=None,
            )
            for dispatch_path in dispatch_paths
        )

    raise ValueError(f"Could not find observation datasets under {source}")


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Dispatch observation CSV is empty: {path}")
        rows = []
        for raw_row in reader:
            row = {key: _parse_scalar(value) for key, value in raw_row.items() if key is not None}
            rows.append(row)
        return rows


def _parse_scalar(value: str | None) -> object:
    if value is None:
        return ""
    stripped = value.strip()
    if stripped == "":
        return ""
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." not in stripped and "e" not in lowered:
            return int(stripped)
        return float(stripped)
    except ValueError:
        return stripped


def _optional_int(value: object | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _validate_required_columns(columns: tuple[str, ...]) -> None:
    missing = [column for column in REQUIRED_DISPATCH_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"Dispatch observations are missing required columns: {missing}")
