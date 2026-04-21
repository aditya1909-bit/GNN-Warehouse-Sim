"""Canonical benchmark-suite orchestration and headline analysis."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
import json
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from warehouse_sim.config import load_benchmark_config
from warehouse_sim.reporting import write_artifact_manifest, write_config_snapshot, write_seed_bundle
from warehouse_sim.simulation import run_benchmark_from_config
from warehouse_sim.simulation.benchmark import _effective_parallel_workers, _resolve_benchmark_paths


@dataclass(frozen=True)
class CanonicalBenchmarkSuiteConfig:
    """Configuration for the dispatch-plus-integrated canonical benchmark suite."""

    name: str
    dispatch_benchmark: Path
    integrated_benchmark: Path
    output_dir: Path
    analyze_after_run: bool = True
    artifact_manifest: Path | None = None
    parallel_workers: int | None = None
    resume: bool = True
    fail_fast: bool = False
    use_mps_for_learned_policies: bool = False
    concurrent_benchmarks: bool = True


def load_canonical_suite_config(path: Path) -> CanonicalBenchmarkSuiteConfig:
    """Load a canonical benchmark-suite config from TOML."""

    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    suite = raw["suite"]
    return CanonicalBenchmarkSuiteConfig(
        name=str(suite["name"]),
        dispatch_benchmark=(path.parent / str(suite["dispatch_benchmark"])).resolve(),
        integrated_benchmark=(path.parent / str(suite["integrated_benchmark"])).resolve(),
        output_dir=(path.parent / str(suite["output_dir"])).resolve(),
        analyze_after_run=bool(suite.get("analyze_after_run", True)),
        artifact_manifest=(
            None
            if suite.get("artifact_manifest") is None
            else (path.parent / str(suite["artifact_manifest"])).resolve()
        ),
        parallel_workers=(
            None
            if suite.get("parallel_workers") in {None, "auto"}
            else int(suite["parallel_workers"])
        ),
        resume=bool(suite.get("resume", True)),
        fail_fast=bool(suite.get("fail_fast", False)),
        use_mps_for_learned_policies=bool(suite.get("use_mps_for_learned_policies", False)),
        concurrent_benchmarks=bool(suite.get("concurrent_benchmarks", True)),
    )


def run_canonical_suite_from_path(
    config_path: Path,
    *,
    parallel_workers_override: int | None = None,
    resume_override: bool | None = None,
    fail_fast_override: bool | None = None,
    use_mps_for_learned_policies_override: bool | None = None,
) -> dict[str, Path]:
    """Run the canonical dispatch and integrated benchmarks and combine their headline outputs."""

    suite = load_canonical_suite_config(config_path)
    suite = replace(
        suite,
        parallel_workers=suite.parallel_workers if parallel_workers_override is None else parallel_workers_override,
        resume=suite.resume if resume_override is None else resume_override,
        fail_fast=suite.fail_fast if fail_fast_override is None else fail_fast_override,
        use_mps_for_learned_policies=(
            suite.use_mps_for_learned_policies
            if use_mps_for_learned_policies_override is None
            else use_mps_for_learned_policies_override
        ),
    )
    suite.output_dir.mkdir(parents=True, exist_ok=True)
    dispatch_output = suite.output_dir / "dispatch"
    integrated_output = suite.output_dir / "integrated"

    written: dict[str, Path] = {
        "dispatch_root": dispatch_output,
        "integrated_root": integrated_output,
    }
    dispatch_benchmark = _load_suite_benchmark(
        suite.dispatch_benchmark,
        suite_root_override=dispatch_output,
        artifact_manifest=suite.artifact_manifest,
        suite=suite,
    )
    integrated_benchmark = _load_suite_benchmark(
        suite.integrated_benchmark,
        suite_root_override=integrated_output,
        artifact_manifest=suite.artifact_manifest,
        suite=suite,
    )
    worker_budget = _effective_parallel_workers(suite.parallel_workers)
    if suite.concurrent_benchmarks and worker_budget > 1:
        dispatch_workers = max(1, worker_budget // 2)
        integrated_workers = max(1, worker_budget - dispatch_workers)
        with ThreadPoolExecutor(max_workers=2) as executor:
            dispatch_future = executor.submit(
                run_benchmark_from_config,
                benchmark_config=replace(dispatch_benchmark, parallel_workers=dispatch_workers),
                benchmark_root_override=dispatch_output,
                parallel_workers_override=dispatch_workers,
                resume_override=suite.resume,
                fail_fast_override=suite.fail_fast,
                use_mps_for_learned_policies_override=suite.use_mps_for_learned_policies,
            )
            integrated_future = executor.submit(
                run_benchmark_from_config,
                benchmark_config=replace(integrated_benchmark, parallel_workers=integrated_workers),
                benchmark_root_override=integrated_output,
                parallel_workers_override=integrated_workers,
                resume_override=suite.resume,
                fail_fast_override=suite.fail_fast,
                use_mps_for_learned_policies_override=suite.use_mps_for_learned_policies,
            )
            dispatch_written = dispatch_future.result()
            integrated_written = integrated_future.result()
    else:
        dispatch_written = run_benchmark_from_config(
            benchmark_config=replace(dispatch_benchmark, parallel_workers=worker_budget),
            benchmark_root_override=dispatch_output,
            parallel_workers_override=worker_budget,
            resume_override=suite.resume,
            fail_fast_override=suite.fail_fast,
            use_mps_for_learned_policies_override=suite.use_mps_for_learned_policies,
        )
        integrated_written = run_benchmark_from_config(
            benchmark_config=replace(integrated_benchmark, parallel_workers=worker_budget),
            benchmark_root_override=integrated_output,
            parallel_workers_override=worker_budget,
            resume_override=suite.resume,
            fail_fast_override=suite.fail_fast,
            use_mps_for_learned_policies_override=suite.use_mps_for_learned_policies,
        )
    written.update({f"dispatch_{label}": path for label, path in dispatch_written.items()})
    written.update({f"integrated_{label}": path for label, path in integrated_written.items()})
    if suite.analyze_after_run:
        written.update(
            {
                f"headline_{label}": path
                for label, path in analyze_canonical_suite(
                    output_dir=suite.output_dir,
                    suite_name=suite.name,
                    dispatch_claims_path=written["dispatch_claims_csv"],
                    integrated_claims_path=written["integrated_claims_csv"],
                    config_path=config_path,
                ).items()
            }
        )
    return written


def analyze_canonical_suite(
    *,
    output_dir: Path,
    suite_name: str,
    dispatch_claims_path: Path,
    integrated_claims_path: Path,
    config_path: Path,
) -> dict[str, Path]:
    """Combine benchmark-level claim tables into one canonical headline results bundle."""

    headline_csv = output_dir / "headline_results.csv"
    headline_json = output_dir / "headline_results.json"
    seed_bundle_path = output_dir / "seed_bundle.json"
    config_snapshot_path = output_dir / "config_snapshot.toml"
    manifest_path = output_dir / "manifest.json"

    rows = [*_read_csv(dispatch_claims_path), *_read_csv(integrated_claims_path)]
    _write_csv(
        headline_csv,
        rows,
        fieldnames=(
            "scenario_name",
            "baseline_policy",
            "challenger_policy",
            "primary_metric",
            "uplift_percent",
            "improvement_ci95_low",
            "improvement_ci95_high",
            "artifact_path",
            "claim_text",
        ),
    )
    headline_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    written = {
        "results_csv": headline_csv,
        "results_json": headline_json,
        "seed_bundle": write_seed_bundle(
            seed_bundle_path,
            {
                "suite_name": suite_name,
                "source_claim_tables": [str(dispatch_claims_path), str(integrated_claims_path)],
            },
        ),
        "config_snapshot": write_config_snapshot(
            config_snapshot_path,
            {
                str(config_path): config_path.read_text(encoding="utf-8"),
            },
        ),
    }
    written["manifest"] = write_artifact_manifest(
        manifest_path,
        benchmark_name=suite_name,
        generated_paths=written,
        config_snapshot_path=written["config_snapshot"],
        seed_bundle_path=written["seed_bundle"],
    )
    return written


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: tuple[str, ...]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(
            {
                fieldname: row.get(fieldname, "")
                for fieldname in fieldnames
            }
            for row in rows
        )


def _load_suite_benchmark(
    benchmark_path: Path,
    *,
    suite_root_override: Path,
    artifact_manifest: Path | None,
    suite: CanonicalBenchmarkSuiteConfig,
):
    benchmark = _resolve_benchmark_paths(load_benchmark_config(benchmark_path), benchmark_path.parent)
    benchmark = replace(
        benchmark,
        output_dir=suite_root_override,
        parallel_workers=suite.parallel_workers,
        resume=suite.resume,
        fail_fast=suite.fail_fast,
        use_mps_for_learned_policies=suite.use_mps_for_learned_policies,
    )
    if artifact_manifest is not None and benchmark.artifact_manifest is None:
        benchmark = replace(benchmark, artifact_manifest=artifact_manifest)
    return benchmark
