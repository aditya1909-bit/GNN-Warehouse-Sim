"""Aggregate report writers for reproducible policy benchmarks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from warehouse_sim.metrics.plots import _load_matplotlib_pyplot
from warehouse_sim.reporting import (
    METRIC_DEFINITIONS_BY_NAME,
    METRIC_NAMES,
    METRIC_SCHEMA_VERSION,
    ordered_aggregate_fields,
    ordered_run_fields,
    validate_benchmark_aggregate_row,
    validate_benchmark_claim_row,
    validate_benchmark_run_row,
    write_artifact_manifest,
    write_config_snapshot,
    write_seed_bundle,
)


def write_benchmark_report(
    output_dir: Path,
    benchmark_name: str,
    rows: list[dict[str, object]],
    *,
    config_sources: Mapping[str, str] | None = None,
    seed_bundle: Mapping[str, object] | None = None,
    write_manifest: bool = True,
) -> dict[str, Path]:
    """Write aggregate benchmark comparison artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "benchmark_summary.csv"
    summary_json = output_dir / "benchmark_summary.json"
    aggregate_csv = output_dir / "benchmark_policy_aggregates.csv"
    claims_csv = output_dir / "benchmark_claims.csv"
    claims_json = output_dir / "benchmark_claims.json"
    figures_dir = output_dir / "figures"
    config_snapshot_path = output_dir / "config_snapshot.toml"
    seed_bundle_path = output_dir / "seed_bundle.json"
    manifest_path = output_dir / "manifest.json"

    if rows:
        extra_run_fields = tuple(
            key for key in rows[0].keys() if key not in set(ordered_run_fields())
        )
        for row in rows:
            validate_benchmark_run_row(row)
        _write_csv(summary_csv, rows, ordered_run_fields(extra_run_fields))
        aggregate_rows = _aggregate_rows(rows)
        extra_aggregate_fields = tuple(
            key for key in aggregate_rows[0].keys() if key not in set(ordered_aggregate_fields())
        )
        for row in aggregate_rows:
            validate_benchmark_aggregate_row(row)
        _write_csv(aggregate_csv, aggregate_rows, ordered_aggregate_fields(extra_aggregate_fields))
    else:
        summary_csv.write_text("", encoding="utf-8")
        aggregate_csv.write_text("", encoding="utf-8")
        claims_csv.write_text("", encoding="utf-8")
        claims_json.write_text("[]\n", encoding="utf-8")
        aggregate_rows = []

    best_by_scenario = _best_by_scenario(aggregate_rows)
    per_seed_breakdown: dict[str, dict[str, list[dict[str, object]]]] = {}
    for row in rows:
        per_seed_breakdown.setdefault(str(row["scenario_id"]), {}).setdefault(str(row["policy"]), []).append(row)

    claim_rows = _build_claim_rows(aggregate_rows, aggregate_csv)
    if claim_rows:
        for row in claim_rows:
            validate_benchmark_claim_row(row)
        _write_csv(claims_csv, claim_rows, claim_rows[0].keys())
    else:
        claims_csv.write_text("", encoding="utf-8")
    claims_json.write_text(json.dumps(claim_rows, indent=2), encoding="utf-8")

    written: dict[str, Path] = {
        "summary_csv": summary_csv,
        "aggregate_csv": aggregate_csv,
        "summary_json": summary_json,
        "claims_csv": claims_csv,
        "claims_json": claims_json,
    }
    if aggregate_rows:
        written.update(_write_benchmark_figures(figures_dir, aggregate_rows))

    if config_sources:
        written["config_snapshot"] = write_config_snapshot(config_snapshot_path, config_sources)
    if seed_bundle is not None:
        written["seed_bundle"] = write_seed_bundle(seed_bundle_path, seed_bundle)
    if write_manifest:
        written["manifest"] = write_artifact_manifest(
            manifest_path,
            benchmark_name=benchmark_name,
            generated_paths=written,
            config_snapshot_path=written.get("config_snapshot"),
            seed_bundle_path=written.get("seed_bundle"),
            extra_metadata={"metric_schema_version": METRIC_SCHEMA_VERSION},
        )

    payload = {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "benchmark_name": benchmark_name,
        "runs": rows,
        "aggregates": aggregate_rows,
        "claims": claim_rows,
        "best_by_scenario": best_by_scenario,
        "per_seed_breakdown": per_seed_breakdown,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return written


def _aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    grouping_fields = (
        "metric_schema_version",
        "benchmark_name",
        "scenario_family",
        "scenario_id",
        "scenario_name",
        "policy",
        "policy_family",
        "policy_role",
        "coordination_mode",
        "execution_model",
        "motion_model",
        "fleet_size",
        "demand_mean_interval",
        "demand_horizon_seconds",
        "layout_rows",
        "layout_columns",
        "blocked_cell_count",
        "directed_edge_count",
        "topology_difficulty",
    )
    for row in rows:
        key = tuple(row[field] for field in grouping_fields)
        grouped.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, object]] = []
    for key, grouped_rows in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        aggregate_row = {
            field_name: field_value
            for field_name, field_value in zip(grouping_fields, key, strict=True)
        }
        aggregate_row["run_count"] = len(grouped_rows)
        aggregate_row["seeds"] = ",".join(str(row["seed"]) for row in grouped_rows)
        for metric_name in METRIC_NAMES:
            values = np.asarray(
                [
                    float(row[metric_name])
                    for row in grouped_rows
                    if row.get(metric_name) is not None
                ],
                dtype=float,
            )
            if values.size == 0:
                aggregate_row[f"{metric_name}_mean"] = None
                aggregate_row[f"{metric_name}_std"] = None
                aggregate_row[f"{metric_name}_ci95_low"] = None
                aggregate_row[f"{metric_name}_ci95_high"] = None
                continue
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ci_halfwidth = float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
            aggregate_row[f"{metric_name}_mean"] = mean
            aggregate_row[f"{metric_name}_std"] = std
            aggregate_row[f"{metric_name}_ci95_low"] = mean - ci_halfwidth
            aggregate_row[f"{metric_name}_ci95_high"] = mean + ci_halfwidth
        aggregate_rows.append(aggregate_row)
    _attach_generalization_gap_metric(aggregate_rows)
    return aggregate_rows


def _best_by_scenario(aggregate_rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    best_by_scenario: dict[str, dict[str, object]] = {}
    for row in aggregate_rows:
        scenario_id = str(row["scenario_id"])
        current_best = best_by_scenario.get(scenario_id)
        if current_best is None or _aggregate_ranking_tuple(row) < _aggregate_ranking_tuple(current_best):
            best_by_scenario[scenario_id] = row
    return best_by_scenario


def _build_claim_rows(
    aggregate_rows: list[dict[str, object]],
    aggregate_csv_path: Path,
) -> list[dict[str, object]]:
    rows_by_scenario: dict[str, list[dict[str, object]]] = {}
    for row in aggregate_rows:
        rows_by_scenario.setdefault(str(row["scenario_id"]), []).append(row)

    claim_rows: list[dict[str, object]] = []
    for scenario_id, scenario_rows in sorted(rows_by_scenario.items()):
        baseline_rows = [
            row
            for row in scenario_rows
            if str(row["policy_role"]) in {"dispatch_baseline", "integrated_baseline"}
        ]
        challenger_rows = [
            row
            for row in scenario_rows
            if str(row["policy_role"]) not in {"dispatch_baseline", "integrated_baseline"}
        ]
        if baseline_rows and challenger_rows:
            baseline = min(baseline_rows, key=_aggregate_ranking_tuple)
            challenger = min(challenger_rows, key=_aggregate_ranking_tuple)
        elif len(scenario_rows) >= 2:
            ranked_rows = sorted(scenario_rows, key=_aggregate_ranking_tuple)
            challenger = ranked_rows[0]
            baseline = ranked_rows[1]
        else:
            continue
        primary_metric = _primary_claim_metric(baseline, challenger)
        winner = _winner_for_metric(baseline, challenger, primary_metric)
        uplift_absolute, uplift_percent, ci_low, ci_high = _improvement_interval(
            baseline,
            challenger,
            primary_metric,
        )
        claim_rows.append(
            {
                "metric_schema_version": METRIC_SCHEMA_VERSION,
                "benchmark_name": baseline["benchmark_name"],
                "scenario_family": baseline["scenario_family"],
                "scenario_id": scenario_id,
                "scenario_name": baseline["scenario_name"],
                "baseline_policy": baseline["policy"],
                "challenger_policy": challenger["policy"],
                "baseline_policy_family": baseline["policy_family"],
                "challenger_policy_family": challenger["policy_family"],
                "comparison_type": _comparison_type(baseline, challenger),
                "primary_metric": primary_metric,
                "winner": winner,
                "baseline_mean": baseline[f"{primary_metric}_mean"],
                "challenger_mean": challenger[f"{primary_metric}_mean"],
                "uplift_absolute": uplift_absolute,
                "uplift_percent": uplift_percent,
                "improvement_ci95_low": ci_low,
                "improvement_ci95_high": ci_high,
                "artifact_path": str(aggregate_csv_path),
                "claim_text": _claim_text(
                    scenario_name=str(baseline["scenario_name"]),
                    primary_metric=primary_metric,
                    baseline_policy=str(baseline["policy"]),
                    challenger_policy=str(challenger["policy"]),
                    uplift_percent=uplift_percent,
                    winner=winner,
                ),
            }
        )
    return claim_rows


def _comparison_type(baseline: dict[str, object], challenger: dict[str, object]) -> str:
    baseline_role = str(baseline["policy_role"])
    challenger_role = str(challenger["policy_role"])
    if baseline_role in {"dispatch_baseline", "integrated_baseline"}:
        return f"{challenger['policy_family']}_vs_{baseline['policy_family']}"
    return "head_to_head"


def _primary_claim_metric(baseline: dict[str, object], challenger: dict[str, object]) -> str:
    ordered_metrics = (
        "collision_event_count",
        "throughput",
        "p95_task_completion_time",
        "mean_task_completion_time",
        "makespan",
    )
    for metric_name in ordered_metrics:
        if baseline.get(f"{metric_name}_mean") is None or challenger.get(f"{metric_name}_mean") is None:
            continue
        if _winner_for_metric(baseline, challenger, metric_name) == "challenger":
            return metric_name
    return "throughput"


def _winner_for_metric(
    baseline: dict[str, object],
    challenger: dict[str, object],
    metric_name: str,
) -> str:
    definition = METRIC_DEFINITIONS_BY_NAME[metric_name]
    baseline_mean = baseline.get(f"{metric_name}_mean")
    challenger_mean = challenger.get(f"{metric_name}_mean")
    if baseline_mean is None or challenger_mean is None:
        return "inconclusive"
    if definition.direction == "maximize":
        return "challenger" if float(challenger_mean) > float(baseline_mean) else "baseline"
    return "challenger" if float(challenger_mean) < float(baseline_mean) else "baseline"


def _improvement_interval(
    baseline: dict[str, object],
    challenger: dict[str, object],
    metric_name: str,
) -> tuple[float | None, float | None, float | None, float | None]:
    definition = METRIC_DEFINITIONS_BY_NAME[metric_name]
    baseline_mean = baseline.get(f"{metric_name}_mean")
    challenger_mean = challenger.get(f"{metric_name}_mean")
    baseline_low = baseline.get(f"{metric_name}_ci95_low")
    baseline_high = baseline.get(f"{metric_name}_ci95_high")
    challenger_low = challenger.get(f"{metric_name}_ci95_low")
    challenger_high = challenger.get(f"{metric_name}_ci95_high")
    if (
        baseline_mean is None
        or challenger_mean is None
        or baseline_low is None
        or baseline_high is None
        or challenger_low is None
        or challenger_high is None
    ):
        return None, None, None, None

    baseline_mean = float(baseline_mean)
    challenger_mean = float(challenger_mean)
    baseline_low = float(baseline_low)
    baseline_high = float(baseline_high)
    challenger_low = float(challenger_low)
    challenger_high = float(challenger_high)

    if definition.direction == "maximize":
        improvement = challenger_mean - baseline_mean
        ci_low = challenger_low - baseline_high
        ci_high = challenger_high - baseline_low
        percent = improvement / baseline_mean * 100.0 if baseline_mean != 0 else None
    else:
        improvement = baseline_mean - challenger_mean
        ci_low = baseline_low - challenger_high
        ci_high = baseline_high - challenger_low
        percent = improvement / baseline_mean * 100.0 if baseline_mean != 0 else None
    return improvement, percent, ci_low, ci_high


def _claim_text(
    *,
    scenario_name: str,
    primary_metric: str,
    baseline_policy: str,
    challenger_policy: str,
    uplift_percent: float | None,
    winner: str,
) -> str:
    if uplift_percent is None:
        return (
            f"{scenario_name}: {challenger_policy} versus {baseline_policy} is inconclusive on {primary_metric}."
        )
    if primary_metric == "collision_event_count":
        if winner == "challenger":
            return (
                f"{scenario_name}: {challenger_policy} reduces {primary_metric} by "
                f"{uplift_percent:.2f}% versus {baseline_policy}."
            )
        return (
            f"{scenario_name}: {baseline_policy} retains the lead on {primary_metric}; "
            f"{challenger_policy} trails by {abs(uplift_percent):.2f}%."
        )
    if winner == "challenger":
        return (
            f"{scenario_name}: {challenger_policy} improves {primary_metric} by "
            f"{uplift_percent:.2f}% versus {baseline_policy}."
        )
    return (
        f"{scenario_name}: {baseline_policy} retains the lead on {primary_metric}; "
        f"{challenger_policy} trails by {abs(uplift_percent):.2f}%."
    )


def _aggregate_ranking_tuple(row: dict[str, object]) -> tuple[float, float, float, float, float, float]:
    collision_event_count = row.get("collision_event_count_mean")
    throughput = row.get("throughput_mean")
    p95_completion = row.get("p95_task_completion_time_mean")
    mean_completion = row.get("mean_task_completion_time_mean")
    makespan = row.get("makespan_mean")
    return (
        1.0 if collision_event_count is not None and float(collision_event_count) > 1e-9 else 0.0,
        float(collision_event_count) if collision_event_count is not None else 0.0,
        -float(throughput) if throughput is not None else float("inf"),
        float(p95_completion) if p95_completion is not None else float("inf"),
        float(mean_completion) if mean_completion is not None else float("inf"),
        float(makespan) if makespan is not None else float("inf"),
    )


def _attach_generalization_gap_metric(aggregate_rows: list[dict[str, object]]) -> None:
    rows_by_policy: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in aggregate_rows:
        rows_by_policy.setdefault((str(row["benchmark_name"]), str(row["policy"])), []).append(row)

    for policy_rows in rows_by_policy.values():
        seen_throughputs = [
            float(row["throughput_mean"])
            for row in policy_rows
            if not _is_generalization_row(row) and row.get("throughput_mean") is not None
        ]
        unseen_throughputs = [
            float(row["throughput_mean"])
            for row in policy_rows
            if _is_generalization_row(row) and row.get("throughput_mean") is not None
        ]
        if not seen_throughputs or not unseen_throughputs:
            continue
        gap = abs(float(np.mean(seen_throughputs)) - float(np.mean(unseen_throughputs)))
        for row in policy_rows:
            row["generalization_gap_seen_vs_unseen_mean"] = gap
            row["generalization_gap_seen_vs_unseen_std"] = 0.0
            row["generalization_gap_seen_vs_unseen_ci95_low"] = gap
            row["generalization_gap_seen_vs_unseen_ci95_high"] = gap


def _is_generalization_row(row: dict[str, object]) -> bool:
    scenario_id = str(row.get("scenario_id", ""))
    scenario_name = str(row.get("scenario_name", ""))
    topology_difficulty = str(row.get("topology_difficulty", ""))
    return (
        topology_difficulty == "generalization"
        or "unseen" in scenario_id
        or "unseen" in scenario_name
    )


def _write_benchmark_figures(
    figures_dir: Path,
    aggregate_rows: list[dict[str, object]],
) -> dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    throughput_path = figures_dir / "throughput_by_policy.png"
    latency_path = figures_dir / "p95_completion_time_by_policy.png"
    plt = _load_matplotlib_pyplot()

    labels = [f"{row['scenario_id']}\n{row['policy']}" for row in aggregate_rows]
    throughput_values = [
        float(row["throughput_mean"]) if row["throughput_mean"] is not None else 0.0
        for row in aggregate_rows
    ]
    latency_values = [
        float(row["p95_task_completion_time_mean"]) if row["p95_task_completion_time_mean"] is not None else 0.0
        for row in aggregate_rows
    ]

    plt.figure(figsize=(max(8, len(labels) * 0.5), 4.5))
    plt.bar(labels, throughput_values)
    plt.ylabel("Throughput")
    plt.title("Benchmark Throughput by Scenario and Policy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(throughput_path)
    plt.close()

    plt.figure(figsize=(max(8, len(labels) * 0.5), 4.5))
    plt.bar(labels, latency_values)
    plt.ylabel("P95 Task Completion Time")
    plt.title("Benchmark Tail Latency by Scenario and Policy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(latency_path)
    plt.close()

    return {
        "throughput_figure": throughput_path,
        "p95_completion_figure": latency_path,
    }


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
