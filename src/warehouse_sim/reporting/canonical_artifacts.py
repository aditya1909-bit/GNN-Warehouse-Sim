"""Helpers for building the trained artifact bundle used by canonical benchmarks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import tomllib

from warehouse_sim.config import load_experiment_config, load_integrated_rl_training_config
from warehouse_sim.learning.cli import run_offline_training_from_path
from warehouse_sim.learning.integrated_rl import run_integrated_rl_training_from_config
from warehouse_sim.reporting import write_artifact_manifest, write_config_snapshot
from warehouse_sim.simulation import run_experiment_from_config


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

    corpus_written = _build_dispatch_corpus(
        config_path=dispatch_corpus_config_path,
        output_dir=dataset_output_dir,
    )
    written.update({f"dataset_{label}": path for label, path in corpus_written.items()})

    for prefix, config_path in (
        ("linear", linear_config_path),
        ("mlp", mlp_config_path),
        ("graph", graph_config_path),
    ):
        outputs = run_offline_training_from_path(config_path)
        written.update({f"{prefix}_{label}": path for label, path in outputs.items()})

    macro_outputs = run_integrated_rl_training_from_config(
        load_integrated_rl_training_config(macro_config_path)
    )
    written.update({f"macro_{label}": path for label, path in macro_outputs.items()})

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
            "artifact_bundle": {
                "dispatch_linear": str(written["linear_artifact"]),
                "dispatch_mlp": str(written["mlp_artifact"]),
                "dispatch_graph": str(written["graph_artifact"]),
                "macro_ppo": str(written["macro_artifact"]),
            }
        },
    )
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
    written["corpus_root"] = output_dir
    return written
