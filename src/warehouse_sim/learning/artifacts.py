"""Serialization for offline-fitted dispatch candidate scorers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from warehouse_sim.learning.features import validate_candidate_feature_names


@dataclass(frozen=True)
class DispatchModelArtifact:
    """Serializable trained dispatch scorer."""

    artifact_version: int
    model_type: str
    objective: str
    feature_names: tuple[str, ...]
    parameters: dict[str, object]
    metadata: dict[str, object] = field(default_factory=dict)

    def score_matrix(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Score a batch of candidate rows."""

        if self.model_type == "grouped_linear":
            weights = np.asarray(self.parameters["weights"], dtype=float)
            bias = float(self.parameters["bias"])
            return feature_matrix @ weights + bias
        if self.model_type == "grouped_mlp":
            normalization = self.parameters["normalization"]
            means = np.asarray(normalization["means"], dtype=float)
            scales = np.asarray(normalization["scales"], dtype=float)
            hidden_weights = np.asarray(self.parameters["hidden_weights"], dtype=float)
            hidden_bias = np.asarray(self.parameters["hidden_bias"], dtype=float)
            output_weights = np.asarray(self.parameters["output_weights"], dtype=float)
            output_bias = float(self.parameters["output_bias"])
            normalized = (feature_matrix - means) / scales
            hidden = np.maximum(normalized @ hidden_weights + hidden_bias, 0.0)
            return hidden @ output_weights + output_bias
        raise ValueError(f"Unsupported model_type: {self.model_type}")


def write_dispatch_model_artifact(artifact: DispatchModelArtifact, path: Path) -> Path:
    """Write a dispatch model artifact to JSON."""

    validate_candidate_feature_names(artifact.feature_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_version": artifact.artifact_version,
        "model_type": artifact.model_type,
        "objective": artifact.objective,
        "feature_names": list(artifact.feature_names),
        "parameters": _to_jsonable(artifact.parameters),
        "metadata": _to_jsonable(artifact.metadata),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_dispatch_model_artifact(path: Path) -> DispatchModelArtifact:
    """Load a saved dispatch model artifact."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return DispatchModelArtifact(
        artifact_version=int(payload["artifact_version"]),
        model_type=str(payload["model_type"]),
        objective=str(payload["objective"]),
        feature_names=validate_candidate_feature_names(tuple(payload["feature_names"])),
        parameters=dict(payload["parameters"]),
        metadata=dict(payload.get("metadata", {})),
    )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value
