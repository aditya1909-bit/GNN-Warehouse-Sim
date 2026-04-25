"""Helpers for building the trained artifact bundle used by canonical benchmarks."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
import shutil
import tomllib

from warehouse_sim.config import (
    load_experiment_config,
    load_integrated_rl_training_config,
    load_offline_training_config,
)
from warehouse_sim.learning.cli import run_offline_training_from_config
from warehouse_sim.reporting import write_artifact_manifest, write_config_snapshot
from warehouse_sim.simulation import run_experiment_from_config
from warehouse_sim.utils.dependencies import require_dependency
from warehouse_sim.utils.progress import ProgressTracker


def build_canonical_artifacts(
    *,
    repo_root: Path,
    dispatch_corpus_config_path: Path,
    linear_config_path: Path,
    mlp_config_path: Path,
    graph_config_path: Path,
    macro_config_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Build the canonical dataset and trained artifact bundle."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_output_dir = output_dir / "dispatch_dataset"
    written: dict[str, Path] = {}
    stage_progress = ProgressTracker(label="canonical_artifacts", total=5, unit="stage")

    corpus_written = _build_dispatch_corpus(
        config_path=dispatch_corpus_config_path,
        output_dir=dataset_output_dir,
    )
    written.update({f"dataset_{label}": path for label, path in corpus_written.items()})
    stage_progress.update(1, extra="dispatch dataset ready", force=True)

    for stage_index, (prefix, config_path) in enumerate(
        (
            ("linear", linear_config_path),
            ("mlp", mlp_config_path),
            ("graph", graph_config_path),
        ),
        start=2,
    ):
        loaded = load_offline_training_config(config_path)
        outputs = run_offline_training_from_config(
            replace(
                loaded,
                dataset=replace(loaded.dataset, source=dataset_output_dir),
                reporting=replace(loaded.reporting, output_dir=output_dir / f"dispatch_{prefix}"),
            )
        )
        written.update({f"{prefix}_{label}": path for label, path in outputs.items()})
        stage_progress.update(stage_index, extra=f"{prefix} artifact ready", force=True)

    loaded_macro = load_integrated_rl_training_config(macro_config_path)
    require_dependency("torch", feature="Canonical integrated PPO artifact generation")
    require_dependency("torch_geometric", feature="Canonical integrated PPO artifact generation")
    from warehouse_sim.learning.integrated_rl import run_integrated_rl_training_from_config

    stage_progress.update(4, extra="starting macro PPO", force=True)
    macro_outputs = run_integrated_rl_training_from_config(
        replace(
            loaded_macro,
            output_dir=output_dir / "macro_ppo",
        )
    )
    written.update({f"macro_{label}": path for label, path in macro_outputs.items()})
    stage_progress.update(5, extra="macro PPO ready", force=True)

    config_snapshot = write_config_snapshot(
        output_dir / "config_snapshot.toml",
        {
            str(path.relative_to(repo_root)): path.read_text(encoding="utf-8")
            for path in (
                dispatch_corpus_config_path,
                linear_config_path,
                mlp_config_path,
                graph_config_path,
                macro_config_path,
            )
        },
    )
    written["config_snapshot"] = config_snapshot
    written["manifest"] = write_artifact_manifest(
        output_dir / "manifest.json",
        benchmark_name="canonical_artifacts",
        generated_paths=written,
        config_snapshot_path=config_snapshot,
        extra_metadata={
            "artifact_aliases": {
                "dispatch_dataset": _relative_to_manifest(output_dir / "manifest.json", written["dataset_corpus_root"]),
                "trained_linear_model": _relative_to_manifest(output_dir / "manifest.json", written["linear_artifact"]),
                "trained_mlp_model": _relative_to_manifest(output_dir / "manifest.json", written["mlp_artifact"]),
                "trained_graph_dispatch_model": _relative_to_manifest(output_dir / "manifest.json", written["graph_artifact"]),
                "trained_end_to_end_macro_ppo": _relative_to_manifest(output_dir / "manifest.json", written["macro_artifact"]),
                "trained_conflict_graph_macro_ppo": _relative_to_manifest(output_dir / "manifest.json", written["macro_artifact"]),
            },
            "artifact_bundle": {
                "dispatch_linear": str(written["linear_artifact"]),
                "dispatch_mlp": str(written["mlp_artifact"]),
                "dispatch_graph": str(written["graph_artifact"]),
                "macro_ppo": str(written["macro_artifact"]),
                "conflict_graph_macro_ppo": str(written["macro_artifact"]),
            }
        },
    )
    stage_progress.close(extra="artifact bundle complete")
    return written


def _build_dispatch_corpus(
    *,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    corpus = raw["corpus"]
    scenario_paths = tuple(
        (config_path.parent / Path(str(item))).resolve()
        for item in corpus["scenario_configs"]
    )
    seeds = tuple(int(seed) for seed in corpus["seeds"])
    teacher_policy = str(corpus.get("policy", "congestion_aware_nearest_robot_task"))
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    total_runs = len(scenario_paths) * len(seeds)
    progress = ProgressTracker(label="dispatch_corpus", total=total_runs, unit="run")
    completed_runs = 0
    for scenario_path in scenario_paths:
        base_config = load_experiment_config(scenario_path)
        for seed in seeds:
            scenario_config = replace(
                base_config,
                demand=replace(base_config.demand, seed=seed),
                simulation=replace(base_config.simulation, policy=teacher_policy),
            )
            run_output_dir = output_dir / base_config.name / f"seed_{seed}"
            _result, run_written = run_experiment_from_config(
                config=scenario_config,
                output_dir_override=run_output_dir,
                force_write_plots=False,
                force_write_observation_dataset=True,
            )
            written[f"{base_config.name}_seed_{seed}_dataset_manifest"] = run_written["dataset_manifest"]
            completed_runs += 1
            progress.update(completed_runs, extra=f"{base_config.name} seed {seed}")
    written["corpus_root"] = output_dir
    progress.close(extra="dataset corpus complete")
    return written


def _relative_to_manifest(manifest_path: Path, target_path: Path) -> str:
    return os.path.relpath(target_path.resolve(), manifest_path.parent.resolve())
