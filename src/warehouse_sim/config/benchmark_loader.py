"""TOML loader for benchmark configuration manifests."""

from __future__ import annotations

import tomllib
from pathlib import Path

from warehouse_sim.config.benchmark_models import BenchmarkConfig
from warehouse_sim.config.models import ConfigValidationError


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    """Load a benchmark manifest from TOML."""

    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    try:
        benchmark = raw["benchmark"]
    except KeyError as exc:
        raise ConfigValidationError("Missing required config section: benchmark") from exc

    scenario_configs = tuple(Path(str(item)) for item in benchmark.get("scenario_configs", []))
    policies = tuple(str(item) for item in benchmark.get("policies", ()))
    output_dir = Path(str(benchmark.get("output_dir", "outputs/benchmark")))
    raw_seeds = benchmark.get("seeds")
    seeds = None if raw_seeds is None else tuple(int(item) for item in raw_seeds)
    policy_artifacts = {
        str(policy_name): Path(str(artifact_path))
        for policy_name, artifact_path in benchmark.get("policy_artifacts", {}).items()
    }

    return BenchmarkConfig(
        name=str(benchmark["name"]),
        scenario_configs=scenario_configs,
        policies=policies,
        output_dir=output_dir,
        write_plots=bool(benchmark.get("write_plots", False)),
        seeds=seeds,
        policy_artifacts=policy_artifacts,
    )
