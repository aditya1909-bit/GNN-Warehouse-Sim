"""Tests for loading trained dispatch artifacts into live simulation policies."""

from __future__ import annotations

from pathlib import Path

from warehouse_sim.learning.artifacts import DispatchModelArtifact, write_dispatch_model_artifact
from warehouse_sim.simulation import run_experiment_from_path


def test_run_experiment_with_trained_linear_model_artifact(tmp_path: Path) -> None:
    artifact_path = write_dispatch_model_artifact(
        DispatchModelArtifact(
            artifact_version=2,
            model_type="grouped_linear",
            objective="dispatch_group_softmax_cross_entropy",
            feature_names=("travel_to_pickup_time", "task_age"),
            parameters={
                "weights": [-2.0, 0.5],
                "bias": 0.0,
            },
        ),
        tmp_path / "trained_linear_artifact.json",
    )
    config_path = tmp_path / "trained_linear_experiment.toml"
    config_path.write_text(
        f"""
name = "trained_linear_policy"

[layout]
rows = 3
columns = 3

[demand]
horizon_seconds = 300.0
mean_interval = 120.0
seed = 7

[robots]
count = 2

[tasks]
default_service_time_estimate = 30.0

[simulation]
policy = "trained_linear_model"
horizon_seconds = 300.0
continue_until_all_tasks_complete = true

[policy_model]
artifact_path = "{artifact_path}"

[reporting]
output_dir = "outputs/trained_linear_policy"
write_plots = false
write_observation_dataset = true
""".strip(),
        encoding="utf-8",
    )

    result, written = run_experiment_from_path(
        config_path=config_path,
        output_dir_override=tmp_path / "outputs",
        force_write_plots=False,
        force_write_observation_dataset=True,
    )

    assert result.policy_name == "trained_linear_model"
    assert written["dispatch_observations"].exists()
    assert written["dataset_manifest"].exists()
