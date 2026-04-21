"""Smoke tests for canonical-style benchmark-suite orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from warehouse_sim.reporting.canonical_suite import run_canonical_suite_from_path


def test_run_canonical_suite_from_path_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dispatch_benchmark_path = tmp_path / "dispatch_benchmark.toml"
    dispatch_benchmark_path.write_text(
        f"""
[benchmark]
name = "dispatch_smoke"
scenario_family = "dispatch_smoke"
scenario_configs = ["{repo_root / 'configs' / 'scenarios' / 'open_low_load.toml'}"]
policies = ["fifo", "nearest_robot_task"]
output_dir = "outputs/dispatch"
write_plots = false
write_manifest = true
seeds = [7]
""".strip(),
        encoding="utf-8",
    )
    integrated_benchmark_path = tmp_path / "integrated_benchmark.toml"
    integrated_benchmark_path.write_text(
        f"""
[benchmark]
name = "integrated_spatial_smoke"
scenario_family = "integrated_spatial_smoke"
scenario_configs = ["{repo_root / 'configs' / 'scenarios' / 'integrated_obstacle_slalom.toml'}"]
policies = ["random_macro", "prioritized_sipp_coordinator"]
output_dir = "outputs/integrated"
write_plots = false
write_manifest = true
seeds = [7]
""".strip(),
        encoding="utf-8",
    )
    suite_config_path = tmp_path / "suite.toml"
    suite_config_path.write_text(
        """
[suite]
name = "suite_smoke"
dispatch_benchmark = "dispatch_benchmark.toml"
integrated_benchmark = "integrated_benchmark.toml"
output_dir = "outputs/suite"
analyze_after_run = true
parallel_workers = 2
resume = true
concurrent_benchmarks = true
""".strip(),
        encoding="utf-8",
    )

    written = run_canonical_suite_from_path(suite_config_path)

    assert written["headline_results_csv"].exists()
    assert written["headline_results_json"].exists()

    payload = json.loads(written["headline_results_json"].read_text(encoding="utf-8"))
    assert {row["scenario_name"] for row in payload} == {
        "open_low_load",
        "integrated_obstacle_slalom",
    }
    assert written["dispatch_run_state"].exists()
    assert written["integrated_run_state"].exists()
