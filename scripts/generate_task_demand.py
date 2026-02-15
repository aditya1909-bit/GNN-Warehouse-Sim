"""Generate synthetic warehouse task arrivals for input modeling.

The process is a non-homogeneous Poisson process simulated via thinning:
- Base arrival rate from mean interarrival time.
- Optional morning-rush multiplier over a time window.
- Optional lunch-break window with zero arrivals.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic warehouse demand generator")
    parser.add_argument("--output", type=Path, default=Path("data/task_demand.csv"))
    parser.add_argument("--horizon-seconds", type=float, default=28_800.0, help="Shift duration")
    parser.add_argument("--mean-interval", type=float, default=10.0, help="Base mean interarrival time")

    parser.add_argument("--rush-start", type=float, default=1_800.0, help="Morning rush start")
    parser.add_argument("--rush-end", type=float, default=7_200.0, help="Morning rush end")
    parser.add_argument("--rush-multiplier", type=float, default=2.0, help="Rate multiplier during rush")

    parser.add_argument("--lunch-start", type=float, default=14_400.0, help="Lunch break start")
    parser.add_argument("--lunch-end", type=float, default=16_200.0, help="Lunch break end")

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-tasks", type=int, default=200)
    return parser.parse_args()


def make_rate_fn(base_rate: float, args: argparse.Namespace):
    def rate_at(t: float) -> float:
        if args.lunch_start <= t < args.lunch_end:
            return 0.0

        rate = base_rate
        if args.rush_start <= t < args.rush_end:
            rate *= args.rush_multiplier
        return rate

    return rate_at


def regime_at(t: float, args: argparse.Namespace) -> str:
    if args.lunch_start <= t < args.lunch_end:
        return "lunch"
    if args.rush_start <= t < args.rush_end:
        return "morning_rush"
    return "base"


def generate_event_times(args: argparse.Namespace) -> np.ndarray:
    if args.mean_interval <= 0:
        raise ValueError("--mean-interval must be > 0")
    if args.horizon_seconds <= 0:
        raise ValueError("--horizon-seconds must be > 0")
    if args.rush_multiplier <= 0:
        raise ValueError("--rush-multiplier must be > 0")

    base_rate = 1.0 / args.mean_interval
    rate_at = make_rate_fn(base_rate, args)

    lambda_max = base_rate * max(1.0, args.rush_multiplier)
    rng = np.random.default_rng(args.seed)

    t = 0.0
    events: list[float] = []

    while t < args.horizon_seconds:
        t += rng.exponential(1.0 / lambda_max)
        if t >= args.horizon_seconds:
            break

        if rng.random() <= rate_at(t) / lambda_max:
            events.append(t)

    return np.asarray(events, dtype=float)


def build_rows(event_times: np.ndarray, args: argparse.Namespace):
    if event_times.size == 0:
        return [], np.array([], dtype=float)

    interarrivals = np.diff(np.insert(event_times, 0, 0.0))
    rows = []
    for idx, (timestamp, interarrival) in enumerate(zip(event_times, interarrivals), start=1):
        rows.append(
            [
                idx,
                round(float(timestamp), 3),
                round(float(interarrival), 3),
                regime_at(float(timestamp), args),
            ]
        )
    return rows, interarrivals


def main() -> None:
    args = parse_args()
    event_times = generate_event_times(args)

    if event_times.size < args.min_tasks:
        raise SystemExit(
            f"Generated {event_times.size} tasks, below --min-tasks={args.min_tasks}. "
            "Increase --horizon-seconds or decrease --mean-interval."
        )

    rows, interarrivals = build_rows(event_times, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Task_ID", "Timestamp", "Interarrival_Time", "Regime"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} tasks to {args.output}")
    print(f"Shift horizon: {args.horizon_seconds:.0f} sec")
    print(f"Observed mean interarrival: {np.mean(interarrivals):.3f} sec")
    print(f"95th percentile interarrival: {np.quantile(interarrivals, 0.95):.3f} sec")


if __name__ == "__main__":
    main()
