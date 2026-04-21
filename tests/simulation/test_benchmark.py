"""Tests for benchmark comparisons across scenarios and policies."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from warehouse_sim.simulation import run_benchmark_from_path
from warehouse_sim.simulation.runner import default_parallel_worker_count, resolve_runtime_device


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
    assert written["run_state"].exists()
    assert written["execution_summary"].exists()


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


def test_run_benchmark_resume_skips_completed_runs(tmp_path: Path) -> None:
    config_path = _tiny_benchmark_config(tmp_path)

    first = run_benchmark_from_path(
        benchmark_config_path=config_path,
        benchmark_root_override=tmp_path / "resume_benchmark",
        force_write_plots=False,
        parallel_workers_override=2,
        resume_override=True,
    )
    second = run_benchmark_from_path(
        benchmark_config_path=config_path,
        benchmark_root_override=tmp_path / "resume_benchmark",
        force_write_plots=False,
        parallel_workers_override=2,
        resume_override=True,
    )

    first_summary = json.loads(first["execution_summary"].read_text(encoding="utf-8"))
    second_summary = json.loads(second["execution_summary"].read_text(encoding="utf-8"))

    assert first_summary["completed_jobs"] == 2
    assert second_summary["skipped_jobs"] == 2
    assert second_summary["completed_jobs"] == 2


def test_parallel_and_serial_benchmark_runs_have_same_row_order(tmp_path: Path) -> None:
    config_path = _tiny_benchmark_config(tmp_path)

    serial = run_benchmark_from_path(
        benchmark_config_path=config_path,
        benchmark_root_override=tmp_path / "serial",
        force_write_plots=False,
        parallel_workers_override=1,
        resume_override=False,
    )
    parallel = run_benchmark_from_path(
        benchmark_config_path=config_path,
        benchmark_root_override=tmp_path / "parallel",
        force_write_plots=False,
        parallel_workers_override=2,
        resume_override=False,
    )

    serial_payload = json.loads(serial["summary_json"].read_text(encoding="utf-8"))
    parallel_payload = json.loads(parallel["summary_json"].read_text(encoding="utf-8"))

    assert [
        (row["scenario_id"], row["seed"], row["policy"])
        for row in serial_payload["runs"]
    ] == [
        (row["scenario_id"], row["seed"], row["policy"])
        for row in parallel_payload["runs"]
    ]


def test_resolve_runtime_device_only_uses_mps_for_opted_in_learned_policies(monkeypatch) -> None:
    monkeypatch.setattr("warehouse_sim.simulation.runner.mps_available", lambda: True)

    assert resolve_runtime_device("fifo", use_mps_for_learned_policies=True) == "cpu"
    assert resolve_runtime_device("trained_linear_model", use_mps_for_learned_policies=True) == "cpu"
    assert resolve_runtime_device("trained_graph_dispatch_model", use_mps_for_learned_policies=True) == "mps"
    assert resolve_runtime_device("trained_end_to_end_macro_ppo", use_mps_for_learned_policies=True) == "mps"
    assert resolve_runtime_device("trained_graph_dispatch_model", use_mps_for_learned_policies=False) == "cpu"


def test_default_parallel_worker_count_caps_at_eight() -> None:
    assert default_parallel_worker_count(logical_cpu_count=11) == 8
    assert default_parallel_worker_count(logical_cpu_count=4) == 3
    assert default_parallel_worker_count(logical_cpu_count=1) == 1


def _tiny_benchmark_config(tmp_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = tmp_path / "tiny_benchmark.toml"
    config_path.write_text(
        f"""
[benchmark]
name = "tiny_benchmark"
scenario_family = "tiny"
scenario_configs = ["{repo_root / 'configs' / 'scenarios' / 'open_low_load.toml'}"]
policies = ["fifo", "nearest_robot_task"]
output_dir = "outputs/tiny"
write_plots = false
write_manifest = true
seeds = [7]
""".strip(),
        encoding="utf-8",
    )
    return config_path
