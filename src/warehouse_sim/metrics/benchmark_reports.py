"""Aggregate report writers for policy benchmarks."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def write_benchmark_report(
    output_dir: Path,
    benchmark_name: str,
    rows: list[dict[str, object]],
) -> dict[str, Path]:
    """Write aggregate benchmark comparison artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "benchmark_summary.csv"
    summary_json = output_dir / "benchmark_summary.json"

    if rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        summary_csv.write_text("", encoding="utf-8")

    best_by_scenario: dict[str, dict[str, object]] = {}
    for row in rows:
        scenario = str(row["scenario_name"])
        current_best = best_by_scenario.get(scenario)
        ranking = (
            -int(row["tasks_completed"]),
            float(row["average_waiting_time"]) if row["average_waiting_time"] is not None else float("inf"),
            float(row["average_turnaround_time"]) if row["average_turnaround_time"] is not None else float("inf"),
            float(row["makespan"]),
        )
        if current_best is None or ranking < _ranking_tuple(current_best):
            best_by_scenario[scenario] = row

    payload = {
        "benchmark_name": benchmark_name,
        "runs": rows,
        "best_by_scenario": best_by_scenario,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"summary_csv": summary_csv, "summary_json": summary_json}


def _ranking_tuple(row: dict[str, object]) -> tuple[float, float, float, float]:
    return (
        -float(row["tasks_completed"]),
        float(row["average_waiting_time"]) if row["average_waiting_time"] is not None else float("inf"),
        float(row["average_turnaround_time"]) if row["average_turnaround_time"] is not None else float("inf"),
        float(row["makespan"]),
    )

