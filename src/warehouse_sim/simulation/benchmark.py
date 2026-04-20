"""Benchmark runner for comparing baseline policies across scenarios."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from warehouse_sim.config import BenchmarkConfig, ExperimentConfig, load_benchmark_config, load_experiment_config
from warehouse_sim.config.models import PolicyModelConfig
from warehouse_sim.metrics import write_benchmark_report
from warehouse_sim.reporting import METRIC_SCHEMA_VERSION, build_simulation_metric_record, load_artifact_aliases
from warehouse_sim.simulation.runner import run_experiment_from_config


def run_benchmark_from_config(
    benchmark_config: BenchmarkConfig,
    benchmark_root_override: Path | None = None,
    force_write_plots: bool | None = None,
) -> dict[str, Path]:
    """Run a policy benchmark from a loaded benchmark config."""

    benchmark_root = benchmark_root_override or benchmark_config.output_dir
    summary_rows: list[dict[str, object]] = []
    scenario_seed_map: dict[str, list[int]] = {}
    config_sources: dict[str, str] = {}

    for scenario_path in benchmark_config.scenario_configs:
        experiment_config = load_experiment_config(scenario_path)
        config_sources[str(scenario_path)] = scenario_path.read_text(encoding="utf-8")
        seeds = benchmark_config.seeds or (experiment_config.demand.seed,)
        scenario_seed_map[experiment_config.name] = list(seeds)
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
                metrics = build_simulation_metric_record(result)
                summary_rows.append(
                    {
                        "metric_schema_version": METRIC_SCHEMA_VERSION,
                        "benchmark_name": benchmark_config.name,
                        "scenario_family": benchmark_config.scenario_family,
                        "scenario_id": experiment_config.name,
                        "scenario_name": experiment_config.name,
                        "scenario_config": str(scenario_path),
                        "seed": seed,
                        "policy": policy,
                        "policy_family": _policy_family(policy),
                        "policy_role": _policy_role(policy),
                        "coordination_mode": policy_config.simulation.coordination_mode,
                        "execution_model": policy_config.simulation.execution_model,
                        "motion_model": (
                            policy_config.coordination.motion_model
                            if policy_config.coordination is not None
                            else "graph_embedded"
                        ),
                        "fleet_size": experiment_config.robots.count,
                        "demand_mean_interval": experiment_config.demand.mean_interval,
                        "demand_horizon_seconds": experiment_config.demand.horizon_seconds,
                        "layout_rows": experiment_config.layout.rows,
                        "layout_columns": experiment_config.layout.columns,
                        "blocked_cell_count": len(experiment_config.layout.blocked_cells),
                        "directed_edge_count": len(experiment_config.layout.directed_edges),
                        "topology_difficulty": _topology_difficulty(experiment_config),
                        "summary_path": str(written_paths["summary"]),
                        **metrics,
                    }
                )

    aggregate_paths = write_benchmark_report(
        output_dir=benchmark_root,
        benchmark_name=benchmark_config.name,
        rows=summary_rows,
        config_sources={
            "benchmark": _benchmark_config_snapshot(benchmark_config),
            **config_sources,
        },
        seed_bundle={
            "benchmark_name": benchmark_config.name,
            "scenario_family": benchmark_config.scenario_family,
            "shared_across_policies": True,
            "scenario_seeds": scenario_seed_map,
        },
        write_manifest=benchmark_config.write_manifest,
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
    resolved_manifest = (
        None
        if config.artifact_manifest is None
        else (
            config.artifact_manifest
            if config.artifact_manifest.is_absolute()
            else (config_dir / config.artifact_manifest).resolve()
        )
    )
    alias_artifacts = {} if resolved_manifest is None else _policy_artifacts_from_manifest(resolved_manifest)
    resolved_policy_artifacts = {
        policy_name: artifact_path
        if artifact_path.is_absolute()
        else (config_dir / artifact_path).resolve()
        for policy_name, artifact_path in config.policy_artifacts.items()
    }
    for policy_name, artifact_path in alias_artifacts.items():
        resolved_policy_artifacts.setdefault(policy_name, artifact_path)
    return replace(
        config,
        scenario_configs=resolved_paths,
        output_dir=resolved_output,
        policy_artifacts=resolved_policy_artifacts,
        artifact_manifest=resolved_manifest,
    )


def _policy_family(policy: str) -> str:
    if policy in {
        "fifo",
        "random",
        "nearest_robot_task",
        "nearest_task_for_idle_robot",
        "congestion_aware_nearest_robot_task",
    }:
        return "heuristic_dispatch"
    if policy in {
        "trained_linear_model",
        "trained_mlp_model",
        "trained_graph_dispatch_model",
    }:
        return "learned_dispatch"
    if policy == "random_macro":
        return "random_integrated"
    if policy in {"prioritized_sipp_coordinator", "optimal_mapf_coordinator"}:
        return "planner_integrated"
    if policy == "trained_end_to_end_macro_ppo":
        return "learned_integrated"
    return "custom"


def _policy_role(policy: str) -> str:
    if policy in {"fifo", "random", "nearest_robot_task", "nearest_task_for_idle_robot"}:
        return "dispatch_baseline"
    if policy == "congestion_aware_nearest_robot_task":
        return "dispatch_advanced_baseline"
    if policy in {"trained_linear_model", "trained_mlp_model", "trained_graph_dispatch_model"}:
        return "dispatch_learned"
    if policy == "random_macro":
        return "integrated_baseline"
    if policy in {"prioritized_sipp_coordinator", "optimal_mapf_coordinator"}:
        return "integrated_planner"
    if policy == "trained_end_to_end_macro_ppo":
        return "integrated_learned"
    return "custom"


def _topology_difficulty(config: ExperimentConfig) -> str:
    name = config.name
    if "open" in name or len(config.layout.blocked_cells) == 0:
        return "open"
    if "bottleneck" in name:
        return "bottleneck"
    if "crossing" in name:
        return "crossing"
    if "high_fleet" in name or config.robots.count >= max(config.layout.rows, config.layout.columns):
        return "high_fleet_density"
    if "unseen" in name:
        return "generalization"
    return "structured"


def _benchmark_config_snapshot(config: BenchmarkConfig) -> str:
    lines = [
        "[benchmark]",
        f'name = "{config.name}"',
        f'scenario_family = "{config.scenario_family}"',
        "scenario_configs = [",
        *[f'  "{path}",' for path in config.scenario_configs],
        "]",
        "policies = [",
        *[f'  "{policy}",' for policy in config.policies],
        "]",
        f'output_dir = "{config.output_dir}"',
        f"write_plots = {'true' if config.write_plots else 'false'}",
        f"write_manifest = {'true' if config.write_manifest else 'false'}",
    ]
    if config.artifact_manifest is not None:
        lines.append(f'artifact_manifest = "{config.artifact_manifest}"')
    if config.seeds is not None:
        seeds = ", ".join(str(seed) for seed in config.seeds)
        lines.append(f"seeds = [{seeds}]")
    if config.policy_artifacts:
        lines.append("")
        lines.append("[benchmark.policy_artifacts]")
        for policy_name, artifact_path in sorted(config.policy_artifacts.items()):
            lines.append(f'{policy_name} = "{artifact_path}"')
    return "\n".join(lines) + "\n"


def _policy_artifacts_from_manifest(artifact_manifest: Path) -> dict[str, Path]:
    aliases = load_artifact_aliases(artifact_manifest)
    return {
        policy_name: aliases[policy_name]
        for policy_name in (
            "trained_linear_model",
            "trained_mlp_model",
            "trained_graph_dispatch_model",
            "trained_end_to_end_macro_ppo",
        )
        if policy_name in aliases
    }
