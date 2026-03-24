"""Tests for benchmark comparisons across scenarios and policies."""

from __future__ import annotations

import json
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

    payload = json.loads(written["summary_json"].read_text(encoding="utf-8"))
    assert payload["benchmark_name"] == "baseline_policy_benchmark"
    assert len(payload["runs"]) == 12
    assert set(payload["best_by_scenario"]) == {
        "peak_load",
        "one_way_flow",
        "blocked_cross_aisle",
    }
