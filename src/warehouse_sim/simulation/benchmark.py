"""Benchmark runner for comparing baseline policies across scenarios."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from warehouse_sim.config import BenchmarkConfig, ExperimentConfig, load_benchmark_config, load_experiment_config
from warehouse_sim.config.models import PolicyModelConfig
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
        seeds = benchmark_config.seeds or (experiment_config.demand.seed,)
        for seed in seeds:
            seeded_config = _override_seed(experiment_config, seed)
            for policy in benchmark_config.policies:
                policy_config = _override_policy(
                    seeded_config,
                    policy,
                    benchmark_config.policy_artifacts.get(policy),
                )
                run_output_dir = benchmark_root / experiment_config.name / f"seed_{seed}" / policy
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
                        "seed": seed,
                        "policy": policy,
                        "coordination_mode": policy_config.simulation.coordination_mode,
                        "execution_model": policy_config.simulation.execution_model,
                        "motion_model": (
                            policy_config.coordination.motion_model
                            if policy_config.coordination is not None
                            else "graph_embedded"
                        ),
                        "tasks_generated": result.metrics.tasks_generated,
                        "tasks_completed": result.metrics.tasks_completed,
                        "tasks_unassigned": result.metrics.tasks_unassigned,
                        "average_waiting_time": result.metrics.average_waiting_time,
                        "average_turnaround_time": result.metrics.average_turnaround_time,
                        "average_travel_distance_per_task": result.metrics.average_travel_distance_per_task,
                        "realized_travel_time_total": result.metrics.realized_travel_time_total,
                        "realized_travel_distance_total": result.metrics.realized_travel_distance_total,
                        "congestion_delay_total": result.metrics.congestion_delay_total,
                        "average_congestion_delay_per_completed_task": (
                            result.metrics.average_congestion_delay_per_completed_task
                        ),
                        "blocked_traversal_events_total": result.metrics.blocked_traversal_events_total,
                        "average_queue_length": result.metrics.average_queue_length,
                        "throughput_per_hour": result.metrics.throughput_per_hour,
                        "makespan": result.metrics.makespan,
                        "safety_violations_total": result.metrics.safety_violations_total,
                        "replans_total": result.metrics.replans_total,
                        "planner_failures_total": result.metrics.planner_failures_total,
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


def _override_policy(
    config: ExperimentConfig,
    policy: str,
    artifact_path: Path | None = None,
) -> ExperimentConfig:
    policy_model = config.policy_model
    if artifact_path is not None:
        policy_model = replace(policy_model, artifact_path=artifact_path) if policy_model is not None else PolicyModelConfig(artifact_path=artifact_path)
    return replace(config, simulation=replace(config.simulation, policy=policy), policy_model=policy_model)


def _override_seed(config: ExperimentConfig, seed: int) -> ExperimentConfig:
    return replace(config, demand=replace(config.demand, seed=seed))


def _resolve_benchmark_paths(config: BenchmarkConfig, config_dir: Path) -> BenchmarkConfig:
    resolved_paths = tuple(
        path if path.is_absolute() else (config_dir / path).resolve()
        for path in config.scenario_configs
    )
    resolved_output = config.output_dir if config.output_dir.is_absolute() else (config_dir / config.output_dir).resolve()
    resolved_policy_artifacts = {
        policy_name: artifact_path
        if artifact_path.is_absolute()
        else (config_dir / artifact_path).resolve()
        for policy_name, artifact_path in config.policy_artifacts.items()
    }
    return replace(
        config,
        scenario_configs=resolved_paths,
        output_dir=resolved_output,
        policy_artifacts=resolved_policy_artifacts,
    )
