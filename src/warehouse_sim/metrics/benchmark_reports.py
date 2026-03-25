"""Aggregate report writers for policy benchmarks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def write_benchmark_report(
    output_dir: Path,
    benchmark_name: str,
    rows: list[dict[str, object]],
) -> dict[str, Path]:
    """Write aggregate benchmark comparison artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "benchmark_summary.csv"
    summary_json = output_dir / "benchmark_summary.json"
    aggregate_csv = output_dir / "benchmark_policy_aggregates.csv"

    if rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        summary_csv.write_text("", encoding="utf-8")
        aggregate_csv.write_text("", encoding="utf-8")
        summary_json.write_text(
            json.dumps(
                {
                    "benchmark_name": benchmark_name,
                    "runs": [],
                    "aggregates": [],
                    "best_by_scenario": {},
                    "per_seed_breakdown": {},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"summary_csv": summary_csv, "aggregate_csv": aggregate_csv, "summary_json": summary_json}

    aggregate_rows = _aggregate_rows(rows)
    with aggregate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    best_by_scenario: dict[str, dict[str, object]] = {}
    for row in aggregate_rows:
        scenario = str(row["scenario_name"])
        current_best = best_by_scenario.get(scenario)
        if current_best is None or _aggregate_ranking_tuple(row) < _aggregate_ranking_tuple(current_best):
            best_by_scenario[scenario] = row

    per_seed_breakdown: dict[str, dict[str, list[dict[str, object]]]] = {}
    for row in rows:
        per_seed_breakdown.setdefault(str(row["scenario_name"]), {}).setdefault(str(row["policy"]), []).append(row)

    payload = {
        "benchmark_name": benchmark_name,
        "runs": rows,
        "aggregates": aggregate_rows,
        "best_by_scenario": best_by_scenario,
        "per_seed_breakdown": per_seed_breakdown,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"summary_csv": summary_csv, "aggregate_csv": aggregate_csv, "summary_json": summary_json}


def _aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["scenario_name"]),
            str(row["policy"]),
            str(row.get("coordination_mode", "dispatch")),
            str(row["execution_model"]),
            str(row.get("motion_model", "graph_embedded")),
        )
        grouped.setdefault(key, []).append(row)

    numeric_fields = [
        key
        for key, value in rows[0].items()
        if key not in {"seed", "summary_path"}
        and isinstance(value, int | float)
        and not isinstance(value, bool)
    ]
    aggregate_rows: list[dict[str, object]] = []
    for (scenario_name, policy, coordination_mode, execution_model, motion_model), grouped_rows in sorted(grouped.items()):
        aggregate_row: dict[str, object] = {
            "scenario_name": scenario_name,
            "policy": policy,
            "coordination_mode": coordination_mode,
            "execution_model": execution_model,
            "motion_model": motion_model,
            "run_count": len(grouped_rows),
            "seeds": ",".join(str(row["seed"]) for row in grouped_rows),
        }
        for metric in numeric_fields:
            values = np.asarray([float(row[metric]) for row in grouped_rows], dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ci_halfwidth = float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
            aggregate_row[f"{metric}_mean"] = mean
            aggregate_row[f"{metric}_std"] = std
            aggregate_row[f"{metric}_ci95_low"] = mean - ci_halfwidth
            aggregate_row[f"{metric}_ci95_high"] = mean + ci_halfwidth
        aggregate_rows.append(aggregate_row)
    return aggregate_rows


def _aggregate_ranking_tuple(row: dict[str, object]) -> tuple[float, float, float, float]:
    return (
        -float(row["tasks_completed_mean"]),
        float(row["average_waiting_time_mean"]) if row["average_waiting_time_mean"] is not None else float("inf"),
        float(row["average_turnaround_time_mean"])
        if row["average_turnaround_time_mean"] is not None
        else float("inf"),
        float(row["makespan_mean"]),
    )
