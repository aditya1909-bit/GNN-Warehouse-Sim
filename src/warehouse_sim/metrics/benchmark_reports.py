"""Aggregate report writers for reproducible policy benchmarks."""

from __future__ import annotations

import csv
import json
import math
import tomllib
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
    paired_csv = output_dir / "benchmark_paired_deltas.csv"
    paired_json = output_dir / "benchmark_paired_deltas.json"
    distinctness_csv = output_dir / "policy_distinctness_audit.csv"
    distinctness_json = output_dir / "policy_distinctness_audit.json"
    collapse_csv = output_dir / "policy_collapse_diagnostics.csv"
    collapse_json = output_dir / "policy_collapse_diagnostics.json"
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
        paired_csv.write_text("", encoding="utf-8")
        paired_json.write_text("[]\n", encoding="utf-8")
        distinctness_csv.write_text("", encoding="utf-8")
        distinctness_json.write_text("[]\n", encoding="utf-8")
        collapse_csv.write_text("", encoding="utf-8")
        collapse_json.write_text("[]\n", encoding="utf-8")
        aggregate_rows = []

    best_by_scenario = _best_by_scenario(aggregate_rows)
    per_seed_breakdown: dict[str, dict[str, list[dict[str, object]]]] = {}
    for row in rows:
        per_seed_breakdown.setdefault(str(row["scenario_id"]), {}).setdefault(str(row["policy"]), []).append(row)

    claim_rows = _build_claim_rows(rows, aggregate_rows, aggregate_csv)
    if claim_rows:
        for row in claim_rows:
            validate_benchmark_claim_row(row)
        _write_csv(claims_csv, claim_rows, _all_fieldnames(claim_rows))
    else:
        claims_csv.write_text("", encoding="utf-8")
    claims_json.write_text(json.dumps(claim_rows, indent=2), encoding="utf-8")

    paired_rows = _build_paired_delta_rows(rows, claim_rows)
    if paired_rows:
        _write_csv(paired_csv, paired_rows, _all_fieldnames(paired_rows))
    else:
        paired_csv.write_text("", encoding="utf-8")
    paired_json.write_text(json.dumps(paired_rows, indent=2), encoding="utf-8")

    distinctness_rows = _build_policy_distinctness_rows(rows)
    if distinctness_rows:
        _write_csv(distinctness_csv, distinctness_rows, _all_fieldnames(distinctness_rows))
    else:
        distinctness_csv.write_text("", encoding="utf-8")
    distinctness_json.write_text(json.dumps(distinctness_rows, indent=2), encoding="utf-8")

    collapse_rows = _build_policy_collapse_diagnostics(rows, distinctness_rows)
    if collapse_rows:
        _write_csv(collapse_csv, collapse_rows, _all_fieldnames(collapse_rows))
    else:
        collapse_csv.write_text("", encoding="utf-8")
    collapse_json.write_text(json.dumps(collapse_rows, indent=2), encoding="utf-8")

    written: dict[str, Path] = {
        "summary_csv": summary_csv,
        "aggregate_csv": aggregate_csv,
        "summary_json": summary_json,
        "claims_csv": claims_csv,
        "claims_json": claims_json,
        "paired_deltas_csv": paired_csv,
        "paired_deltas_json": paired_json,
        "distinctness_audit_csv": distinctness_csv,
        "distinctness_audit_json": distinctness_json,
        "collapse_diagnostics_csv": collapse_csv,
        "collapse_diagnostics_json": collapse_json,
    }
    if aggregate_rows:
        written.update(
            _write_benchmark_figures(
                figures_dir=figures_dir,
                rows=rows,
                aggregate_rows=aggregate_rows,
                claim_rows=claim_rows,
                paired_rows=paired_rows,
                distinctness_rows=distinctness_rows,
                collapse_rows=collapse_rows,
            )
        )

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
        "paired_deltas": paired_rows,
        "policy_distinctness_audit": distinctness_rows,
        "policy_collapse_diagnostics": collapse_rows,
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
            mean, std, ci_low, ci_high = _mean_std_ci(values)
            aggregate_row[f"{metric_name}_mean"] = mean
            aggregate_row[f"{metric_name}_std"] = std
            aggregate_row[f"{metric_name}_ci95_low"] = ci_low
            aggregate_row[f"{metric_name}_ci95_high"] = ci_high
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
    rows: list[dict[str, object]],
    aggregate_rows: list[dict[str, object]],
    aggregate_csv_path: Path,
) -> list[dict[str, object]]:
    rows_by_scenario: dict[str, list[dict[str, object]]] = {}
    for row in aggregate_rows:
        rows_by_scenario.setdefault(str(row["scenario_id"]), []).append(row)

    raw_lookup: dict[tuple[str, str, int], dict[str, object]] = {}
    for row in rows:
        raw_lookup[(str(row["scenario_id"]), str(row["policy"]), int(row["seed"]))] = row

    claim_rows: list[dict[str, object]] = []
    for scenario_id, scenario_rows in sorted(rows_by_scenario.items()):
        baseline, challenger = _select_claim_pair(scenario_rows)
        if baseline is None or challenger is None:
            continue
        primary_metric = _primary_claim_metric(baseline, challenger)
        winner = _winner_for_metric(baseline, challenger, primary_metric)
        paired_deltas = _paired_improvements_for_metric(
            raw_lookup=raw_lookup,
            scenario_id=scenario_id,
            baseline_policy=str(baseline["policy"]),
            challenger_policy=str(challenger["policy"]),
            metric_name=primary_metric,
        )
        if paired_deltas:
            uplift_absolute = float(np.mean(paired_deltas))
            ci_low, ci_high = _ci_bounds(np.asarray(paired_deltas, dtype=float))
            baseline_mean = baseline.get(f"{primary_metric}_mean")
            uplift_percent = (
                uplift_absolute / float(baseline_mean) * 100.0
                if baseline_mean not in (None, 0)
                else None
            )
        else:
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
                "paired_seed_count": len(paired_deltas),
                "claim_supported": bool(ci_low is not None and ci_low > 0),
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


def _select_claim_pair(
    scenario_rows: list[dict[str, object]],
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
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
        return (
            min(baseline_rows, key=_aggregate_ranking_tuple),
            min(challenger_rows, key=_aggregate_ranking_tuple),
        )
    if len(scenario_rows) >= 2:
        ranked_rows = sorted(scenario_rows, key=_aggregate_ranking_tuple)
        return ranked_rows[1], ranked_rows[0]
    return None, None


def _build_paired_delta_rows(
    rows: list[dict[str, object]],
    claim_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    raw_lookup: dict[tuple[str, str, int], dict[str, object]] = {}
    for row in rows:
        raw_lookup[(str(row["scenario_id"]), str(row["policy"]), int(row["seed"]))] = row

    paired_rows: list[dict[str, object]] = []
    for claim in claim_rows:
        scenario_id = str(claim["scenario_id"])
        metric_name = str(claim["primary_metric"])
        baseline_policy = str(claim["baseline_policy"])
        challenger_policy = str(claim["challenger_policy"])
        baseline_seeds = {
            seed
            for (row_scenario, row_policy, seed) in raw_lookup
            if row_scenario == scenario_id and row_policy == baseline_policy
        }
        challenger_seeds = {
            seed
            for (row_scenario, row_policy, seed) in raw_lookup
            if row_scenario == scenario_id and row_policy == challenger_policy
        }
        shared_seeds = sorted(baseline_seeds & challenger_seeds)
        deltas: list[float] = []
        for seed in shared_seeds:
            baseline_value = raw_lookup[(scenario_id, baseline_policy, seed)].get(metric_name)
            challenger_value = raw_lookup[(scenario_id, challenger_policy, seed)].get(metric_name)
            if baseline_value is None or challenger_value is None:
                continue
            delta = _signed_improvement(
                float(baseline_value),
                float(challenger_value),
                metric_name,
            )
            deltas.append(delta)
            paired_rows.append(
                {
                    "metric_schema_version": METRIC_SCHEMA_VERSION,
                    "benchmark_name": claim["benchmark_name"],
                    "scenario_family": claim["scenario_family"],
                    "scenario_id": scenario_id,
                    "scenario_name": claim["scenario_name"],
                    "baseline_policy": baseline_policy,
                    "challenger_policy": challenger_policy,
                    "primary_metric": metric_name,
                    "seed": seed,
                    "paired_improvement": delta,
                }
            )
        if deltas:
            mean_delta, std_delta, ci_low, ci_high = _mean_std_ci(np.asarray(deltas, dtype=float))
            paired_rows.append(
                {
                    "metric_schema_version": METRIC_SCHEMA_VERSION,
                    "benchmark_name": claim["benchmark_name"],
                    "scenario_family": claim["scenario_family"],
                    "scenario_id": scenario_id,
                    "scenario_name": claim["scenario_name"],
                    "baseline_policy": baseline_policy,
                    "challenger_policy": challenger_policy,
                    "primary_metric": metric_name,
                    "seed": "aggregate",
                    "paired_improvement": mean_delta,
                    "paired_improvement_std": std_delta,
                    "paired_improvement_ci95_low": ci_low,
                    "paired_improvement_ci95_high": ci_high,
                }
            )
    return paired_rows


def _build_policy_distinctness_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["scenario_id"]), int(row["seed"])), []).append(row)

    distinctness_rows: list[dict[str, object]] = []
    for (scenario_id, seed), grouped_rows in sorted(grouped.items()):
        for index, left in enumerate(sorted(grouped_rows, key=lambda item: str(item["policy"]))):
            for right in sorted(grouped_rows, key=lambda item: str(item["policy"]))[index + 1:]:
                left_trace = _decision_trace(str(left.get("coordination_mode")), left)
                right_trace = _decision_trace(str(right.get("coordination_mode")), right)
                if not left_trace and not right_trace:
                    identical_rate = 1.0
                elif not left_trace or not right_trace:
                    identical_rate = 0.0
                else:
                    overlap = min(len(left_trace), len(right_trace))
                    identical = sum(
                        left_trace[position] == right_trace[position]
                        for position in range(overlap)
                    )
                    identical_rate = identical / overlap if overlap else 0.0
                distinctness_rows.append(
                    {
                        "metric_schema_version": METRIC_SCHEMA_VERSION,
                        "benchmark_name": left["benchmark_name"],
                        "scenario_family": left["scenario_family"],
                        "scenario_id": scenario_id,
                        "scenario_name": left["scenario_name"],
                        "seed": seed,
                        "left_policy": left["policy"],
                        "right_policy": right["policy"],
                        "left_policy_family": left["policy_family"],
                        "right_policy_family": right["policy_family"],
                        "decision_count_left": len(left_trace),
                        "decision_count_right": len(right_trace),
                        "identical_decision_rate": identical_rate,
                        "identical_trace": identical_rate == 1.0 and len(left_trace) == len(right_trace),
                        "audit_status": (
                            "potentially_collapsed"
                            if identical_rate == 1.0 and len(left_trace) == len(right_trace)
                            else "distinct"
                        ),
                    }
                )
    return distinctness_rows


def _build_policy_collapse_diagnostics(
    rows: list[dict[str, object]],
    distinctness_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    collapsed_pairs = {
        (
            str(row["scenario_id"]),
            int(row["seed"]),
            *sorted((str(row["left_policy"]), str(row["right_policy"]))),
        )
        for row in distinctness_rows
        if str(row.get("audit_status")) == "potentially_collapsed"
    }
    row_lookup = {
        (str(row["scenario_id"]), int(row["seed"]), str(row["policy"])): row
        for row in rows
    }
    diagnostics: list[dict[str, object]] = []
    for scenario_id, seed, left_policy, right_policy in sorted(collapsed_pairs):
        left_row = row_lookup.get((scenario_id, seed, left_policy))
        right_row = row_lookup.get((scenario_id, seed, right_policy))
        if left_row is None or right_row is None:
            continue
        if str(left_row.get("coordination_mode")) == "integrated":
            diagnostics.append(_integrated_collapse_diagnostic(left_row, right_row))
        else:
            diagnostics.append(_dispatch_collapse_diagnostic(left_row, right_row))
    return diagnostics


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


def _paired_improvements_for_metric(
    *,
    raw_lookup: dict[tuple[str, str, int], dict[str, object]],
    scenario_id: str,
    baseline_policy: str,
    challenger_policy: str,
    metric_name: str,
) -> list[float]:
    baseline_seeds = {
        seed for (row_scenario, row_policy, seed) in raw_lookup
        if row_scenario == scenario_id and row_policy == baseline_policy
    }
    challenger_seeds = {
        seed for (row_scenario, row_policy, seed) in raw_lookup
        if row_scenario == scenario_id and row_policy == challenger_policy
    }
    deltas: list[float] = []
    for seed in sorted(baseline_seeds & challenger_seeds):
        baseline_value = raw_lookup[(scenario_id, baseline_policy, seed)].get(metric_name)
        challenger_value = raw_lookup[(scenario_id, challenger_policy, seed)].get(metric_name)
        if baseline_value is None or challenger_value is None:
            continue
        deltas.append(
            _signed_improvement(float(baseline_value), float(challenger_value), metric_name)
        )
    return deltas


def _signed_improvement(baseline_value: float, challenger_value: float, metric_name: str) -> float:
    definition = METRIC_DEFINITIONS_BY_NAME[metric_name]
    if definition.direction == "maximize":
        return challenger_value - baseline_value
    return baseline_value - challenger_value


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
    *,
    figures_dir: Path,
    rows: list[dict[str, object]],
    aggregate_rows: list[dict[str, object]],
    claim_rows: list[dict[str, object]],
    paired_rows: list[dict[str, object]],
    distinctness_rows: list[dict[str, object]],
    collapse_rows: list[dict[str, object]],
) -> dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    written.update(_write_claim_forest_plot(figures_dir, claim_rows))
    written.update(_write_small_multiples(figures_dir, aggregate_rows))
    written.update(_write_per_seed_dot_plot(figures_dir, paired_rows))
    written.update(_write_seen_vs_unseen_gap_plot(figures_dir, aggregate_rows))
    written.update(_write_distinctness_heatmap(figures_dir, distinctness_rows))
    written.update(_write_collapse_diagnostics_figure(figures_dir, collapse_rows))
    written.update(_write_mechanism_figure(figures_dir, rows, claim_rows))
    written.update(_write_congestion_heatmap(figures_dir, rows))
    written.update(_write_dispatch_decision_explainer(figures_dir, rows))
    return written


def _write_claim_forest_plot(figures_dir: Path, claim_rows: list[dict[str, object]]) -> dict[str, Path]:
    if not claim_rows:
        return {}
    plot_path = figures_dir / "claim_forest_plot.png"
    plt = _load_matplotlib_pyplot()
    ordered_rows = sorted(
        claim_rows,
        key=lambda row: (
            str(row["scenario_family"]),
            str(row["scenario_id"]),
        ),
    )
    y_positions = np.arange(len(ordered_rows))
    means = np.asarray([float(row["uplift_absolute"] or 0.0) for row in ordered_rows], dtype=float)
    lower = np.asarray(
        [
            max(means[index] - float(row["improvement_ci95_low"]), 0.0)
            if row["improvement_ci95_low"] is not None
            else 0.0
            for index, row in enumerate(ordered_rows)
        ],
        dtype=float,
    )
    upper = np.asarray(
        [
            max(float(row["improvement_ci95_high"]) - means[index], 0.0)
            if row["improvement_ci95_high"] is not None
            else 0.0
            for index, row in enumerate(ordered_rows)
        ],
        dtype=float,
    )
    colors = ["#146356" if bool(row.get("claim_supported")) else "#9e9e9e" for row in ordered_rows]

    plt.figure(figsize=(10, max(4.5, 0.7 * len(ordered_rows))))
    plt.errorbar(means, y_positions, xerr=np.vstack([lower, upper]), fmt="none", ecolor="#4a4a4a", capsize=3)
    plt.scatter(means, y_positions, c=colors, s=55, zorder=3)
    plt.axvline(0.0, color="#444444", linestyle="--", linewidth=1)
    plt.yticks(
        y_positions,
        [
            f"{row['scenario_id']}: {row['challenger_policy']} vs {row['baseline_policy']}"
            for row in ordered_rows
        ],
    )
    plt.xlabel("Paired improvement on primary metric")
    plt.title("Claim Forest Plot")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return {"claim_forest_plot": plot_path}


def _write_small_multiples(figures_dir: Path, aggregate_rows: list[dict[str, object]]) -> dict[str, Path]:
    if not aggregate_rows:
        return {}
    plot_path = figures_dir / "throughput_small_multiples.png"
    plt = _load_matplotlib_pyplot()
    panel_groups = {}
    for row in aggregate_rows:
        panel_groups.setdefault(_scenario_panel_group(row), []).append(row)
    group_names = [name for name in ("open", "bottleneck", "generalization", "reserved", "other") if name in panel_groups]
    if not group_names:
        return {}

    figure, axes = plt.subplots(
        len(group_names),
        1,
        figsize=(12, max(4.5, 3.0 * len(group_names))),
        squeeze=False,
    )
    family_colors = {
        "heuristic_dispatch": "#4c78a8",
        "learned_dispatch": "#f58518",
        "planner_integrated": "#54a24b",
        "learned_integrated": "#e45756",
        "random_integrated": "#9d755d",
    }
    for axis, group_name in zip(axes.flatten(), group_names, strict=True):
        group_rows = sorted(
            panel_groups[group_name],
            key=lambda row: (str(row["scenario_id"]), str(row["policy"])),
        )
        scenario_labels = [f"{row['scenario_id']}\n{row['policy']}" for row in group_rows]
        values = [float(row["throughput_mean"] or 0.0) for row in group_rows]
        colors = [family_colors.get(str(row["policy_family"]), "#7f7f7f") for row in group_rows]
        axis.bar(np.arange(len(group_rows)), values, color=colors)
        axis.set_ylabel("Throughput")
        axis.set_title(group_name.replace("_", " ").title())
        axis.set_xticks(np.arange(len(group_rows)))
        axis.set_xticklabels(scenario_labels, rotation=45, ha="right")
    figure.suptitle("Scenario Family Small Multiples")
    figure.tight_layout()
    figure.savefig(plot_path)
    plt.close(figure)
    return {"throughput_small_multiples": plot_path}


def _write_per_seed_dot_plot(figures_dir: Path, paired_rows: list[dict[str, object]]) -> dict[str, Path]:
    seed_rows = [row for row in paired_rows if row["seed"] != "aggregate"]
    if not seed_rows:
        return {}
    plot_path = figures_dir / "paired_seed_dot_plot.png"
    plt = _load_matplotlib_pyplot()

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in seed_rows:
        key = f"{row['scenario_id']}|{row['challenger_policy']}|{row['baseline_policy']}"
        grouped.setdefault(key, []).append(row)
    ordered_keys = sorted(grouped)

    plt.figure(figsize=(12, max(4.5, 0.8 * len(ordered_keys))))
    for position, key in enumerate(ordered_keys):
        group = sorted(grouped[key], key=lambda row: int(row["seed"]))
        values = [float(row["paired_improvement"]) for row in group]
        x_positions = np.full(len(values), position, dtype=float)
        jitter = np.linspace(-0.12, 0.12, num=max(len(values), 1))
        plt.scatter(x_positions + jitter[: len(values)], values, color="#1f77b4", alpha=0.85, s=35)
        mean_value = float(np.mean(values))
        plt.plot([position - 0.2, position + 0.2], [mean_value, mean_value], color="#d62728", linewidth=2)
    plt.axhline(0.0, color="#444444", linestyle="--", linewidth=1)
    plt.xticks(
        np.arange(len(ordered_keys)),
        [key.replace("|", "\n") for key in ordered_keys],
        rotation=45,
        ha="right",
    )
    plt.ylabel("Paired improvement")
    plt.title("Per-Seed Paired Improvements")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return {"paired_seed_dot_plot": plot_path}


def _write_seen_vs_unseen_gap_plot(figures_dir: Path, aggregate_rows: list[dict[str, object]]) -> dict[str, Path]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in aggregate_rows:
        bucket = "unseen" if _is_generalization_row(row) else "seen"
        grouped.setdefault(str(row["policy"]), {}).setdefault(bucket, []).append(float(row["throughput_mean"] or 0.0))
    comparable = {
        policy: buckets
        for policy, buckets in grouped.items()
        if buckets.get("seen") and buckets.get("unseen")
    }
    if not comparable:
        return {}
    plot_path = figures_dir / "seen_vs_unseen_gap.png"
    plt = _load_matplotlib_pyplot()

    policies = sorted(comparable)
    seen = [float(np.mean(comparable[policy]["seen"])) for policy in policies]
    unseen = [float(np.mean(comparable[policy]["unseen"])) for policy in policies]
    x_positions = np.arange(len(policies))

    plt.figure(figsize=(10, 5))
    for index, policy in enumerate(policies):
        plt.plot([0, 1], [seen[index], unseen[index]], marker="o", linewidth=2, label=policy)
    plt.xticks([0, 1], ["Seen scenarios", "Unseen scenarios"])
    plt.ylabel("Mean throughput")
    plt.title("Seen vs Unseen Throughput Gap")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return {"seen_vs_unseen_gap_plot": plot_path}


def _write_distinctness_heatmap(figures_dir: Path, distinctness_rows: list[dict[str, object]]) -> dict[str, Path]:
    if not distinctness_rows:
        return {}
    plot_path = figures_dir / "policy_distinctness_heatmap.png"
    plt = _load_matplotlib_pyplot()

    policy_names = sorted(
        {
            str(row["left_policy"]) for row in distinctness_rows
        } | {
            str(row["right_policy"]) for row in distinctness_rows
        }
    )
    if not policy_names:
        return {}
    matrix = np.full((len(policy_names), len(policy_names)), np.nan, dtype=float)
    row_by_pair: dict[tuple[str, str], list[float]] = {}
    for row in distinctness_rows:
        pair = tuple(sorted((str(row["left_policy"]), str(row["right_policy"]))))
        row_by_pair.setdefault(pair, []).append(float(row["identical_decision_rate"]))
    for i, left_policy in enumerate(policy_names):
        matrix[i, i] = 1.0
        for j, right_policy in enumerate(policy_names):
            if i == j:
                continue
            pair = tuple(sorted((left_policy, right_policy)))
            if pair in row_by_pair:
                matrix[i, j] = float(np.mean(row_by_pair[pair]))

    plt.figure(figsize=(8.5, 7))
    image = plt.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(image, label="Identical decision rate")
    plt.xticks(np.arange(len(policy_names)), policy_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(policy_names)), policy_names)
    plt.title("Policy Distinctness Audit")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return {"policy_distinctness_heatmap": plot_path}


def _write_collapse_diagnostics_figure(figures_dir: Path, collapse_rows: list[dict[str, object]]) -> dict[str, Path]:
    if not collapse_rows:
        return {}
    plot_path = figures_dir / "policy_collapse_diagnostics.png"
    plt = _load_matplotlib_pyplot()
    ordered_rows = sorted(
        collapse_rows,
        key=lambda row: (
            str(row["scenario_id"]),
            str(row["left_policy"]),
            str(row["right_policy"]),
        ),
    )[:20]
    labels = [f"{row['scenario_id']}\n{row['left_policy']} vs {row['right_policy']}" for row in ordered_rows]
    values = [float(row.get("investigation_signal", 0.0) or 0.0) for row in ordered_rows]
    plt.figure(figsize=(11, max(5, 0.45 * len(ordered_rows))))
    plt.barh(np.arange(len(ordered_rows)), values, color="#8c564b")
    plt.yticks(np.arange(len(ordered_rows)), labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Collapse investigation signal")
    plt.title("Policy Collapse Diagnostics")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return {"policy_collapse_diagnostics_figure": plot_path}


def _write_mechanism_figure(
    figures_dir: Path,
    rows: list[dict[str, object]],
    claim_rows: list[dict[str, object]],
) -> dict[str, Path]:
    claim = next((row for row in claim_rows if str(row["scenario_id"]) == "integrated_narrow_bottleneck"), None)
    if claim is None:
        return {}
    selected = _select_reference_pair(rows, claim)
    if selected is None:
        return {}
    baseline_row, challenger_row = selected
    baseline_traces = _read_csv_dicts(Path(str(baseline_row.get("robot_trajectories_path", ""))))
    challenger_traces = _read_csv_dicts(Path(str(challenger_row.get("robot_trajectories_path", ""))))
    baseline_exec = _read_csv_dicts(Path(str(baseline_row.get("executions_path", ""))))
    challenger_exec = _read_csv_dicts(Path(str(challenger_row.get("executions_path", ""))))
    if not baseline_traces or not challenger_traces:
        return {}

    plot_path = figures_dir / "integrated_narrow_bottleneck_mechanism.png"
    plt = _load_matplotlib_pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    scenario_layout = _load_layout_metadata(Path(str(challenger_row["scenario_config"])))

    _plot_layout_trajectories(
        axes[0, 0],
        scenario_layout=scenario_layout,
        trajectory_rows=baseline_traces,
        title=f"Baseline routes: {baseline_row['policy']}",
        color="#d62728",
    )
    _plot_layout_trajectories(
        axes[0, 1],
        scenario_layout=scenario_layout,
        trajectory_rows=challenger_traces,
        title=f"Planner routes: {challenger_row['policy']}",
        color="#2ca02c",
    )
    _plot_bottleneck_occupancy(
        axes[1, 0],
        baseline_traces=baseline_traces,
        challenger_traces=challenger_traces,
    )
    _plot_completion_cdf(
        axes[1, 1],
        baseline_exec=baseline_exec,
        challenger_exec=challenger_exec,
        baseline_label=str(baseline_row["policy"]),
        challenger_label=str(challenger_row["policy"]),
    )
    figure.suptitle("Mechanism Figure: Integrated Narrow Bottleneck")
    figure.tight_layout()
    figure.savefig(plot_path)
    plt.close(figure)
    return {"integrated_narrow_bottleneck_mechanism": plot_path}


def _write_congestion_heatmap(figures_dir: Path, rows: list[dict[str, object]]) -> dict[str, Path]:
    integrated_rows = [
        row
        for row in rows
        if str(row.get("scenario_id")) == "integrated_narrow_bottleneck"
        and str(row.get("coordination_mode")) == "integrated"
    ]
    if not integrated_rows:
        return {}
    target_row = min(
        integrated_rows,
        key=lambda row: (
            -float(row.get("throughput") or 0.0),
            float(row.get("p95_task_completion_time") or float("inf")),
        ),
    )
    trajectory_rows = _read_csv_dicts(Path(str(target_row.get("robot_trajectories_path", ""))))
    if not trajectory_rows:
        return {}
    plot_path = figures_dir / "integrated_narrow_bottleneck_congestion_heatmap.png"
    plt = _load_matplotlib_pyplot()
    figure, axis = plt.subplots(figsize=(7, 6))
    scenario_layout = _load_layout_metadata(Path(str(target_row["scenario_config"])))
    _plot_congestion_heatmap(axis, scenario_layout, trajectory_rows)
    axis.set_title(f"Congestion Heatmap: {target_row['policy']}")
    figure.tight_layout()
    figure.savefig(plot_path)
    plt.close(figure)
    return {"integrated_congestion_heatmap": plot_path}


def _write_dispatch_decision_explainer(figures_dir: Path, rows: list[dict[str, object]]) -> dict[str, Path]:
    dispatch_rows = [
        row for row in rows
        if str(row.get("coordination_mode")) == "dispatch"
        and Path(str(row.get("dispatch_traces_path", ""))).exists()
    ]
    target_row = next(
        (
            row for row in dispatch_rows
            if "due_pressure" in str(row.get("scenario_id"))
            and str(row.get("policy")) not in {"fifo", "random"}
        ),
        None,
    )
    if target_row is None and dispatch_rows:
        target_row = max(dispatch_rows, key=lambda row: float(row.get("throughput") or 0.0))
    if target_row is None:
        return {}
    trace_rows = _read_csv_dicts(Path(str(target_row.get("dispatch_traces_path", ""))))
    if not trace_rows:
        return {}

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in trace_rows:
        grouped.setdefault(str(row["dispatch_index"]), []).append(row)
    candidate_groups = [
        group
        for group in grouped.values()
        if len(group) >= 3 and any(_float_or_none(row.get("policy_score")) is not None for row in group)
    ]
    if not candidate_groups:
        return {}
    selected_group = max(candidate_groups, key=lambda group: _dispatch_group_spread(group))

    plot_path = figures_dir / "dispatch_decision_explainer.png"
    plt = _load_matplotlib_pyplot()
    ordered_rows = sorted(
        selected_group,
        key=lambda row: (
            float("-inf") if _float_or_none(row.get("policy_score")) is None else float(row["policy_score"]),
            -int(row.get("policy_rank", 9999) or 9999),
        ),
        reverse=True,
    )[:8]
    labels = [
        f"{row['candidate_robot_id']} -> {row['candidate_task_id'] or row['candidate_charging_node_id']}"
        for row in ordered_rows
    ]
    scores = [float(_float_or_none(row.get("policy_score")) or 0.0) for row in ordered_rows]
    colors = ["#2ca02c" if row["is_selected"] == "True" else "#7f7f7f" for row in ordered_rows]
    score_label = next(
        (str(row.get("policy_score_label")) for row in ordered_rows if row.get("policy_score_label")),
        "policy score",
    )

    plt.figure(figsize=(10, 5.5))
    plt.barh(np.arange(len(ordered_rows)), scores, color=colors)
    plt.yticks(np.arange(len(ordered_rows)), labels)
    plt.gca().invert_yaxis()
    plt.xlabel(score_label.replace("_", " "))
    plt.title(
        f"Dispatch Decision Explainer: {target_row['scenario_id']} ({target_row['policy']})"
    )
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return {"dispatch_decision_explainer": plot_path}


def _scenario_panel_group(row: dict[str, object]) -> str:
    scenario_id = str(row.get("scenario_id", ""))
    if "reserved" in scenario_id:
        return "reserved"
    if "bottleneck" in scenario_id:
        return "bottleneck"
    if "open" in scenario_id:
        return "open"
    if "unseen" in scenario_id:
        return "generalization"
    return "other"


def _decision_trace(coordination_mode: str, row: dict[str, object]) -> list[str]:
    if coordination_mode == "integrated":
        trace_path = Path(str(row.get("macro_decisions_path", "")))
        records = _read_csv_dicts(trace_path)
        return [
            "|".join(
                (
                    record.get("decision_index", ""),
                    record.get("robot_id", ""),
                    record.get("macro_type", ""),
                    record.get("task_id", ""),
                    record.get("charging_node", ""),
                    record.get("route_nodes", ""),
                )
            )
            for record in records
        ]
    trace_path = Path(str(row.get("dispatch_traces_path", "")))
    records = _read_csv_dicts(trace_path)
    selected = [record for record in records if record.get("is_selected") == "True"]
    selected.sort(key=lambda record: int(record.get("dispatch_index", 0)))
    return [
        "|".join(
            (
                record.get("dispatch_index", ""),
                record.get("selected_robot_id", ""),
                record.get("selected_action_type", ""),
                record.get("selected_task_id", ""),
                record.get("selected_charging_node_id", ""),
            )
        )
        for record in selected
    ]


def _select_reference_pair(
    rows: list[dict[str, object]],
    claim: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]] | None:
    scenario_id = str(claim["scenario_id"])
    baseline_policy = str(claim["baseline_policy"])
    challenger_policy = str(claim["challenger_policy"])
    baseline_rows = [
        row for row in rows
        if str(row["scenario_id"]) == scenario_id and str(row["policy"]) == baseline_policy
    ]
    challenger_rows = [
        row for row in rows
        if str(row["scenario_id"]) == scenario_id and str(row["policy"]) == challenger_policy
    ]
    challenger_by_seed = {int(row["seed"]): row for row in challenger_rows}
    shared = [
        (baseline_row, challenger_by_seed[int(baseline_row["seed"])])
        for baseline_row in baseline_rows
        if int(baseline_row["seed"]) in challenger_by_seed
    ]
    if not shared:
        return None
    return max(
        shared,
        key=lambda pair: _signed_improvement(
            float(pair[0].get(str(claim["primary_metric"])) or 0.0),
            float(pair[1].get(str(claim["primary_metric"])) or 0.0),
            str(claim["primary_metric"]),
        ),
    )


def _load_layout_metadata(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    return payload.get("layout", {})


def _dispatch_collapse_diagnostic(left_row: dict[str, object], right_row: dict[str, object]) -> dict[str, object]:
    left_records = _read_csv_dicts(Path(str(left_row.get("dispatch_traces_path", ""))))
    right_records = _read_csv_dicts(Path(str(right_row.get("dispatch_traces_path", ""))))
    left_selected = [record for record in left_records if record.get("is_selected") == "True"]
    right_selected = [record for record in right_records if record.get("is_selected") == "True"]
    left_margin = _mean_selected_policy_margin(left_records)
    right_margin = _mean_selected_policy_margin(right_records)
    congestion_delay = _mean_value(left_selected, "estimated_pickup_congestion_delay") + _mean_value(left_selected, "estimated_dropoff_congestion_delay")
    due_time_remaining = _mean_value(left_selected, "task_due_time_remaining")
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "benchmark_name": left_row["benchmark_name"],
        "scenario_family": left_row["scenario_family"],
        "scenario_id": left_row["scenario_id"],
        "scenario_name": left_row["scenario_name"],
        "seed": left_row["seed"],
        "left_policy": left_row["policy"],
        "right_policy": right_row["policy"],
        "diagnostic_type": "dispatch",
        "mean_selected_policy_margin_left": left_margin,
        "mean_selected_policy_margin_right": right_margin,
        "mean_selected_congestion_delay": congestion_delay,
        "mean_selected_due_time_remaining": due_time_remaining,
        "investigation_signal": abs(left_margin - right_margin) + congestion_delay,
        "hypothesis": (
            "objective_surface_flat"
            if congestion_delay < 1e-6 and abs(left_margin) < 1e-6 and abs(right_margin) < 1e-6
            else "heuristic_alignment"
        ),
    }


def _integrated_collapse_diagnostic(left_row: dict[str, object], right_row: dict[str, object]) -> dict[str, object]:
    left_records = _read_csv_dicts(Path(str(left_row.get("macro_decisions_path", ""))))
    right_records = _read_csv_dicts(Path(str(right_row.get("macro_decisions_path", ""))))
    left_rank_1_rate = _fraction_with_int_value(left_records, "selected_rank_by_estimated_completion", 1)
    right_rank_1_rate = _fraction_with_int_value(right_records, "selected_rank_by_estimated_completion", 1)
    left_gap = _mean_value(left_records, "selected_completion_gap")
    right_gap = _mean_value(right_records, "selected_completion_gap")
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "benchmark_name": left_row["benchmark_name"],
        "scenario_family": left_row["scenario_family"],
        "scenario_id": left_row["scenario_id"],
        "scenario_name": left_row["scenario_name"],
        "seed": left_row["seed"],
        "left_policy": left_row["policy"],
        "right_policy": right_row["policy"],
        "diagnostic_type": "integrated",
        "fraction_rank_1_left": left_rank_1_rate,
        "fraction_rank_1_right": right_rank_1_rate,
        "mean_completion_gap_left": left_gap,
        "mean_completion_gap_right": right_gap,
        "investigation_signal": max(left_rank_1_rate, right_rank_1_rate) - min(left_gap, right_gap),
        "hypothesis": (
            "greedy_candidate_equivalence"
            if left_rank_1_rate > 0.95 and right_rank_1_rate > 0.95 and left_gap < 1e-6 and right_gap < 1e-6
            else "planner_policy_alignment"
        ),
    }


def _plot_layout_trajectories(axis, *, scenario_layout: dict[str, object], trajectory_rows: list[dict[str, str]], title: str, color: str) -> None:
    from matplotlib.patches import Rectangle

    for blocked_cell in scenario_layout.get("blocked_cells", []):
        row, column = blocked_cell
        axis.add_patch(
            Rectangle(
                (float(column) - 0.5, float(row) - 0.5),
                1.0,
                1.0,
                facecolor="#cfcfcf",
                edgecolor="#8a8a8a",
            )
        )
    for record in trajectory_rows:
        start_x = _float_or_none(record.get("start_x"))
        start_y = _float_or_none(record.get("start_y"))
        end_x = _float_or_none(record.get("end_x"))
        end_y = _float_or_none(record.get("end_y"))
        if None in {start_x, start_y, end_x, end_y}:
            start = _node_id_to_xy(str(record.get("source_id", "")))
            end = _node_id_to_xy(str(record.get("target_id", "")))
            if start is None or end is None:
                continue
            start_x, start_y = start
            end_x, end_y = end
        axis.plot([start_x, end_x], [start_y, end_y], color=color, alpha=0.35, linewidth=2)
    axis.set_xlim(-0.6, float(scenario_layout.get("columns", 1)) - 0.4)
    axis.set_ylim(float(scenario_layout.get("rows", 1)) - 0.4, -0.6)
    axis.set_aspect("equal")
    axis.set_title(title)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")


def _plot_bottleneck_occupancy(axis, *, baseline_traces: list[dict[str, str]], challenger_traces: list[dict[str, str]]) -> None:
    hotspot_nodes = _hotspot_nodes([*baseline_traces, *challenger_traces])
    baseline_series = _occupancy_series(baseline_traces, hotspot_nodes=hotspot_nodes)
    challenger_series = _occupancy_series(challenger_traces, hotspot_nodes=hotspot_nodes)
    axis.plot(baseline_series["time"], baseline_series["occupancy"], label="baseline", color="#d62728")
    axis.plot(challenger_series["time"], challenger_series["occupancy"], label="planner", color="#2ca02c")
    axis.set_title("Bottleneck occupancy over time")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Traversals through hotspot corridor")
    axis.legend(loc="best")


def _plot_completion_cdf(axis, *, baseline_exec: list[dict[str, str]], challenger_exec: list[dict[str, str]], baseline_label: str, challenger_label: str) -> None:
    baseline = sorted(
        value
        for value in (_float_or_none(row.get("completion_time")) for row in baseline_exec)
        if value is not None
    )
    challenger = sorted(
        value
        for value in (_float_or_none(row.get("completion_time")) for row in challenger_exec)
        if value is not None
    )
    if baseline:
        axis.plot(baseline, np.linspace(0.0, 1.0, num=len(baseline), endpoint=True), label=baseline_label, color="#d62728")
    if challenger:
        axis.plot(challenger, np.linspace(0.0, 1.0, num=len(challenger), endpoint=True), label=challenger_label, color="#2ca02c")
    axis.set_title("Completion-time CDF")
    axis.set_xlabel("Completion time (s)")
    axis.set_ylabel("CDF")
    axis.legend(loc="best")


def _occupancy_series(trajectory_rows: list[dict[str, str]], *, hotspot_nodes: tuple[str, ...]) -> dict[str, list[float]]:
    if not trajectory_rows:
        return {"time": [], "occupancy": []}
    counts: dict[int, int] = {}
    for row in trajectory_rows:
        start_time = _float_or_none(row.get("start_time"))
        end_time = _float_or_none(row.get("end_time"))
        source_id = str(row.get("source_id", ""))
        target_id = str(row.get("target_id", ""))
        if start_time is None or end_time is None:
            continue
        if hotspot_nodes and source_id not in hotspot_nodes and target_id not in hotspot_nodes:
            continue
        for bucket in range(int(math.floor(start_time)), int(math.ceil(end_time)) + 1):
            counts[bucket] = counts.get(bucket, 0) + 1
    ordered = sorted(counts.items())
    return {
        "time": [float(time_value) for time_value, _ in ordered],
        "occupancy": [float(count) for _, count in ordered],
    }


def _plot_congestion_heatmap(axis, scenario_layout: dict[str, object], trajectory_rows: list[dict[str, str]]) -> None:
    rows = int(scenario_layout.get("rows", 1))
    columns = int(scenario_layout.get("columns", 1))
    node_counts = np.zeros((rows, columns), dtype=float)
    for blocked_cell in scenario_layout.get("blocked_cells", []):
        row, column = blocked_cell
        node_counts[int(row), int(column)] = np.nan
    for row in trajectory_rows:
        for field_name in ("source_id", "target_id"):
            node_id = str(row.get(field_name, ""))
            if not node_id.startswith("r") or "_c" not in node_id:
                continue
            grid_row, grid_column = node_id[1:].split("_c", maxsplit=1)
            node_counts[int(grid_row), int(grid_column)] = np.nan_to_num(node_counts[int(grid_row), int(grid_column)], nan=0.0) + 1.0
    image = axis.imshow(node_counts, cmap="inferno")
    axis.figure.colorbar(image, ax=axis, label="Traversal count")
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")


def _dispatch_group_spread(group: list[dict[str, str]]) -> float:
    scores = []
    for row in group:
        policy_score = _float_or_none(row.get("policy_score"))
        scores.append(policy_score if policy_score is not None else _dispatch_explanation_score(row))
    return max(scores) - min(scores)


def _dispatch_explanation_score(row: Mapping[str, str]) -> float:
    priority = float(row.get("task_priority", 0.0) or 0.0)
    due_time = _float_or_none(row.get("task_due_time_remaining"))
    travel_time = float(row.get("travel_to_pickup_time", 0.0) or 0.0) + float(row.get("pickup_to_dropoff_time", 0.0) or 0.0)
    congestion = float(row.get("estimated_pickup_congestion_delay", 0.0) or 0.0) + float(row.get("estimated_dropoff_congestion_delay", 0.0) or 0.0)
    blocked = float(row.get("estimated_pickup_blocked_segments", 0.0) or 0.0) + float(row.get("estimated_dropoff_blocked_segments", 0.0) or 0.0)
    due_term = 0.0 if due_time is None else max(0.0, 120.0 - due_time) / 10.0
    return priority * 2.0 + due_term - 0.35 * travel_time - 0.8 * congestion - 0.6 * blocked


def _hotspot_nodes(trajectory_rows: list[dict[str, str]]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    coordinates = [
        coordinate
        for coordinate in (_node_id_to_grid(str(row.get(field_name, ""))) for row in trajectory_rows for field_name in ("source_id", "target_id"))
        if coordinate is not None
    ]
    max_row = max((coordinate[0] for coordinate in coordinates), default=0)
    max_column = max((coordinate[1] for coordinate in coordinates), default=0)
    for row in trajectory_rows:
        for field_name in ("source_id", "target_id"):
            node_id = str(row.get(field_name, ""))
            if "_c" not in node_id:
                continue
            coordinate = _node_id_to_grid(node_id)
            if coordinate is None:
                continue
            is_interior = 0 < coordinate[0] < max_row and 0 < coordinate[1] < max_column
            counts[node_id] = counts.get(node_id, 0) + (2 if is_interior else 1)
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return tuple(node_id for node_id, _count in ordered[:3])


def _node_id_to_xy(node_id: str) -> tuple[float, float] | None:
    coordinate = _node_id_to_grid(node_id)
    if coordinate is None:
        return None
    row, column = coordinate
    return float(column), float(row)


def _node_id_to_grid(node_id: str) -> tuple[int, int] | None:
    if not node_id.startswith("r") or "_c" not in node_id:
        return None
    row_text, column_text = node_id[1:].split("_c", maxsplit=1)
    return int(row_text), int(column_text)


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path or not path.exists() or not path.is_file():
        return []
    if not path.read_text(encoding="utf-8").strip():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean_selected_policy_margin(rows: list[dict[str, str]]) -> float:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("dispatch_index", "")), []).append(row)
    margins: list[float] = []
    for group in grouped.values():
        selected = next((row for row in group if row.get("is_selected") == "True"), None)
        if selected is None:
            continue
        selected_score = _float_or_none(selected.get("policy_score"))
        candidate_scores = sorted(
            (
                score for score in (_float_or_none(row.get("policy_score")) for row in group)
                if score is not None
            ),
            reverse=True,
        )
        if selected_score is None or len(candidate_scores) < 2:
            continue
        margins.append(selected_score - candidate_scores[1])
    return float(np.mean(margins)) if margins else 0.0


def _mean_value(rows: list[dict[str, str]], field_name: str) -> float:
    values = [value for value in (_float_or_none(row.get(field_name)) for row in rows) if value is not None]
    return float(np.mean(values)) if values else 0.0


def _fraction_with_int_value(rows: list[dict[str, str]], field_name: str, value: int) -> float:
    if not rows:
        return 0.0
    count = sum(1 for row in rows if int(float(row.get(field_name, 0) or 0)) == value)
    return count / len(rows)


def _mean_std_ci(values: np.ndarray) -> tuple[float, float, float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    ci_low, ci_high = _ci_bounds(values)
    return mean, std, ci_low, ci_high


def _ci_bounds(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, mean
    std = float(np.std(values, ddof=1))
    ci_halfwidth = float(1.96 * std / math.sqrt(len(values)))
    return mean - ci_halfwidth, mean + ci_halfwidth


def _float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _all_fieldnames(rows: list[dict[str, object]]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return tuple(ordered)
