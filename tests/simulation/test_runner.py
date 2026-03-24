"""Tests for config-driven experiment execution."""

from __future__ import annotations

from pathlib import Path

from warehouse_sim.simulation import run_experiment_from_path


def test_run_experiment_from_config_path(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "baseline_experiment.toml"

    result, written = run_experiment_from_path(
        config_path=config_path,
        output_dir_override=tmp_path,
        force_write_plots=False,
        force_write_observation_dataset=True,
    )

    assert result.metrics.tasks_generated >= result.metrics.tasks_completed
    assert written["summary"].exists()
    assert written["executions"].exists()
    assert written["queue_snapshots"].exists()
    assert written["robot_metrics"].exists()
    assert written["graph_nodes"].exists()
    assert written["graph_arcs"].exists()
    assert written["dispatch_observations"].exists()
