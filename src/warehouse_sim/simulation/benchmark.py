"""Benchmark runner for comparing baseline policies across scenarios."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from warehouse_sim.config import BenchmarkConfig, ExperimentConfig, load_benchmark_config, load_experiment_config
from warehouse_sim.metrics import write_benchmark_report
from warehouse_sim.simulation.runner import run_experiment_from_config


def run_benchmark_from_config(
    benchmark_config: BenchmarkConfig,
    benchmark_root_override: Path | None = None,
    force_write_plots: bool | None = None,
) -> dict[str, Path]:
    """Run a policy benchmark from a loaded benchmark config."""

    benchmark_root = benchmark_root_override or benchmark_config.output_dir
    summary_rows: list[dict[str, object]] = []

    for scenario_path in benchmark_config.scenario_configs:
        experiment_config = load_experiment_config(scenario_path)
        for policy in benchmark_config.policies:
            policy_config = _override_policy(experiment_config, policy)
            run_output_dir = benchmark_root / experiment_config.name / policy
            result, written_paths = run_experiment_from_config(
                config=policy_config,
                output_dir_override=run_output_dir,
                force_write_plots=(
                    benchmark_config.write_plots if force_write_plots is None else force_write_plots
                ),
            )
            summary_rows.append(
                {
                    "scenario_name": experiment_config.name,
                    "scenario_config": str(scenario_path),
                    "policy": policy,
                    "tasks_generated": result.metrics.tasks_generated,
                    "tasks_completed": result.metrics.tasks_completed,
                    "tasks_unassigned": result.metrics.tasks_unassigned,
                    "average_waiting_time": result.metrics.average_waiting_time,
                    "average_turnaround_time": result.metrics.average_turnaround_time,
                    "average_travel_distance_per_task": result.metrics.average_travel_distance_per_task,
                    "average_queue_length": result.metrics.average_queue_length,
                    "throughput_per_hour": result.metrics.throughput_per_hour,
                    "makespan": result.metrics.makespan,
                    "summary_path": str(written_paths["summary"]),
                }
            )

    aggregate_paths = write_benchmark_report(
        output_dir=benchmark_root,
        benchmark_name=benchmark_config.name,
        rows=summary_rows,
    )
    return aggregate_paths


def run_benchmark_from_path(
    benchmark_config_path: Path,
    benchmark_root_override: Path | None = None,
    force_write_plots: bool | None = None,
) -> dict[str, Path]:
    """Load a benchmark config and run it."""

    benchmark_config = load_benchmark_config(benchmark_config_path)
    benchmark_config = _resolve_benchmark_paths(benchmark_config, benchmark_config_path.parent)
    return run_benchmark_from_config(
        benchmark_config=benchmark_config,
        benchmark_root_override=benchmark_root_override,
        force_write_plots=force_write_plots,
    )


def _override_policy(config: ExperimentConfig, policy: str) -> ExperimentConfig:
    return replace(config, simulation=replace(config.simulation, policy=policy))


def _resolve_benchmark_paths(config: BenchmarkConfig, config_dir: Path) -> BenchmarkConfig:
    resolved_paths = tuple(
        path if path.is_absolute() else (config_dir / path).resolve()
        for path in config.scenario_configs
    )
    resolved_output = config.output_dir if config.output_dir.is_absolute() else (config_dir / config.output_dir).resolve()
    return replace(config, scenario_configs=resolved_paths, output_dir=resolved_output)

