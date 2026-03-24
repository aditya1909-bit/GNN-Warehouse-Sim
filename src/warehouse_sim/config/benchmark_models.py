"""Benchmark configuration models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from warehouse_sim.config.models import ConfigValidationError


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for multi-scenario policy benchmarking."""

    name: str
    scenario_configs: tuple[Path, ...]
    policies: tuple[str, ...]
    output_dir: Path
    write_plots: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigValidationError("benchmark.name must be non-empty.")
        if not self.scenario_configs:
            raise ConfigValidationError("benchmark.scenario_configs must be non-empty.")
        if not self.policies:
            raise ConfigValidationError("benchmark.policies must be non-empty.")

