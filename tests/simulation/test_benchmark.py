"""Tests for benchmark comparisons across scenarios and policies."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from warehouse_sim.simulation import run_benchmark_from_path


def test_run_benchmark_from_manifest(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "policy_benchmark.toml"

    written = run_benchmark_from_path(
        benchmark_config_path=config_path,
        benchmark_root_override=tmp_path,
        force_write_plots=False,
    )

    assert written["summary_csv"].exists()
    assert written["summary_json"].exists()
    assert written["paired_deltas_csv"].exists()
    assert written["distinctness_audit_csv"].exists()

    payload = json.loads(written["summary_json"].read_text(encoding="utf-8"))
    assert payload["benchmark_name"] == "baseline_policy_benchmark"
    assert len(payload["runs"]) == 15
    assert "paired_deltas" in payload
    assert "policy_distinctness_audit" in payload
    assert set(payload["best_by_scenario"]) == {
        "peak_load",
        "one_way_flow",
        "blocked_cross_aisle",
    }


def test_run_congestion_benchmark_manifest(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "congestion_policy_benchmark.toml"

    written = run_benchmark_from_path(
        benchmark_config_path=config_path,
        benchmark_root_override=tmp_path,
        force_write_plots=False,
    )

    payload = json.loads(written["summary_json"].read_text(encoding="utf-8"))
    assert payload["benchmark_name"] == "congestion_policy_benchmark"
    assert len(payload["runs"]) == 12
    assert "narrow_bottleneck" in payload["best_by_scenario"]


def test_run_integrated_benchmark_manifest(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "integrated_coordination_benchmark.toml"

    written = run_benchmark_from_path(
        benchmark_config_path=config_path,
        benchmark_root_override=tmp_path,
        force_write_plots=False,
    )

    payload = json.loads(written["summary_json"].read_text(encoding="utf-8"))
    assert payload["benchmark_name"] == "integrated_coordination_benchmark"
    assert any(row["coordination_mode"] == "integrated" for row in payload["runs"])


def test_run_spatial_realism_integrated_benchmark_manifest(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "benchmarks" / "spatial_realism_integrated_benchmark.toml"

    written = run_benchmark_from_path(
        benchmark_config_path=config_path,
        benchmark_root_override=tmp_path,
        force_write_plots=False,
    )

    payload = json.loads(written["summary_json"].read_text(encoding="utf-8"))
    assert payload["benchmark_name"] == "spatial_realism_integrated_benchmark"
    assert set(payload["best_by_scenario"]) == {
        "integrated_blocked_cross_aisle",
        "integrated_obstacle_slalom",
        "integrated_unseen_geometry_generalization",
    }


def test_run_benchmark_script_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "policy_benchmark.toml"
    output_dir = tmp_path / "script_benchmark"

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_benchmark.py"),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "benchmark_summary.json").exists()
    assert (output_dir / "manifest.json").exists()
