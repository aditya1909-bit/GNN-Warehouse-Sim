"""Benchmark configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from warehouse_sim.config.models import ConfigValidationError


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for multi-scenario policy benchmarking."""

    name: str
    scenario_configs: tuple[Path, ...]
    policies: tuple[str, ...]
    output_dir: Path
    scenario_family: str = "custom"
    write_plots: bool = False
    write_manifest: bool = True
    seeds: tuple[int, ...] | None = None
    policy_artifacts: dict[str, Path] = field(default_factory=dict)
    artifact_manifest: Path | None = None
    parallel_workers: int | None = None
    resume: bool = True
    fail_fast: bool = False
    use_mps_for_learned_policies: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigValidationError("benchmark.name must be non-empty.")
        if not self.scenario_configs:
            raise ConfigValidationError("benchmark.scenario_configs must be non-empty.")
        if not self.policies:
            raise ConfigValidationError("benchmark.policies must be non-empty.")
        if not self.scenario_family:
            raise ConfigValidationError("benchmark.scenario_family must be non-empty.")
        if self.seeds is not None and not self.seeds:
            raise ConfigValidationError("benchmark.seeds must be non-empty.")
        if self.parallel_workers is not None and self.parallel_workers <= 0:
            raise ConfigValidationError("benchmark.parallel_workers must be > 0 when provided.")
