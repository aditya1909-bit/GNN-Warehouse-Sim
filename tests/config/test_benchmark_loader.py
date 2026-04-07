"""Tests for benchmark manifest loading."""

from __future__ import annotations

from pathlib import Path

from warehouse_sim.config import load_benchmark_config


def test_load_policy_benchmark_manifest() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "policy_benchmark.toml"
    config = load_benchmark_config(config_path)

    assert config.name == "baseline_policy_benchmark"
    assert len(config.scenario_configs) == 3
    assert "nearest_robot_task" in config.policies
    assert "congestion_aware_nearest_robot_task" in config.policies
    assert config.seeds is None
    assert config.scenario_family == "custom"
    assert config.write_manifest is True


def test_load_benchmark_manifest_with_policy_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "benchmark.toml"
    config_path.write_text(
        """
[benchmark]
name = "trained_policy_benchmark"
scenario_configs = ["scenario.toml"]
policies = ["fifo", "trained_linear_model"]
output_dir = "outputs/benchmark"
seeds = [7, 11]

[benchmark.policy_artifacts]
trained_linear_model = "artifacts/linear.json"
""".strip(),
        encoding="utf-8",
    )

    config = load_benchmark_config(config_path)

    assert config.policy_artifacts["trained_linear_model"] == Path("artifacts/linear.json")
    assert config.seeds == (7, 11)


def test_load_integrated_benchmark_manifest() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "integrated_coordination_benchmark.toml"
    config = load_benchmark_config(config_path)

    assert config.name == "integrated_coordination_benchmark"
    assert "prioritized_sipp_coordinator" in config.policies
    assert "random_macro" in config.policies


def test_load_integrated_optimal_mapf_benchmark_manifest() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "integrated_optimal_mapf_benchmark.toml"
    config = load_benchmark_config(config_path)

    assert config.name == "integrated_optimal_mapf_benchmark"
    assert "optimal_mapf_coordinator" in config.policies


def test_load_canonical_dispatch_benchmark_manifest() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "benchmarks" / "canonical_dispatch_benchmark.toml"
    config = load_benchmark_config(config_path)

    assert config.name == "canonical_dispatch_benchmark"
    assert config.scenario_family == "canonical_dispatch"
    assert "trained_graph_dispatch_model" in config.policies
    assert config.seeds == (7, 11, 13, 17, 19)


def test_load_spatial_realism_integrated_benchmark_manifest() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "benchmarks" / "spatial_realism_integrated_benchmark.toml"
    config = load_benchmark_config(config_path)

    assert config.name == "spatial_realism_integrated_benchmark"
    assert config.scenario_family == "spatial_realism_integrated"
    assert "prioritized_sipp_coordinator" in config.policies
    assert config.seeds == (7,)
