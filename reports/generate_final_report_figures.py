from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "outputs" / "canonical_artifacts" / "macro_ppo"
FIGURE_DIR = REPO_ROOT / "reports" / "figures"


def _scenario_label(name: str) -> str:
    mapping = {
        "integrated_reserved_edges": "Reserved\nEdges",
        "integrated_reserved_nodes": "Reserved\nNodes",
        "integrated_narrow_bottleneck": "Narrow\nBottleneck",
        "integrated_high_fleet_density_heavy": "High Fleet\nDensity Heavy",
        "integrated_dense_merge_heavy": "Dense Merge\nHeavy",
    }
    return mapping.get(name, name.replace("integrated_", "").replace("_", " ").title())


def load_learned_eval() -> list[dict]:
    return json.loads((ARTIFACT_ROOT / "evaluation_rollouts.json").read_text())


def load_baseline_eval() -> dict[str, dict]:
    baseline_root = ARTIFACT_ROOT / "baseline_eval"
    rows: dict[str, dict] = {}
    for scenario_dir in baseline_root.iterdir():
        summary_path = scenario_dir / "seed_19" / "summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text())
        rows[scenario_dir.name] = payload["metrics"]
    return rows


def load_warm_start_metrics() -> list[dict]:
    with (ARTIFACT_ROOT / "warm_start_metrics.csv").open() as handle:
        return list(csv.DictReader(handle))


def load_training_metrics() -> list[dict]:
    with (ARTIFACT_ROOT / "training_metrics.csv").open() as handle:
        return list(csv.DictReader(handle))


def plot_throughput_and_wait(learned_eval: list[dict], baseline_eval: dict[str, dict]) -> None:
    scenarios = [row["scenario_name"] for row in learned_eval]
    labels = [_scenario_label(name) for name in scenarios]
    learned_throughput = [row["throughput_per_hour"] for row in learned_eval]
    baseline_throughput = [baseline_eval[name]["throughput_per_hour"] for name in scenarios]
    learned_wait = [row["planner_wait_time_total"] for row in learned_eval]
    baseline_wait = [baseline_eval[name]["planner_wait_time_total"] for name in scenarios]

    x = np.arange(len(scenarios))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)

    axes[0].bar(x - width / 2, baseline_throughput, width, label="Baseline", color="#9aa5b1")
    axes[0].bar(x + width / 2, learned_throughput, width, label="Learned GNN", color="#1f77b4")
    axes[0].set_title("Throughput by Scenario")
    axes[0].set_ylabel("Throughput per hour")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - width / 2, baseline_wait, width, label="Baseline", color="#c7cdd4")
    axes[1].bar(x + width / 2, learned_wait, width, label="Learned GNN", color="#2ca02c")
    axes[1].set_title("Planner Wait Time by Scenario")
    axes[1].set_ylabel("Planner wait time total")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Canonical Validation Comparison on Seed 19", fontsize=14, y=1.02)
    fig.savefig(FIGURE_DIR / "scenario_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ratios(learned_eval: list[dict], baseline_eval: dict[str, dict]) -> None:
    scenarios = [row["scenario_name"] for row in learned_eval]
    labels = [_scenario_label(name) for name in scenarios]
    throughput_ratio = [
        row["throughput_per_hour"] / baseline_eval[row["scenario_name"]]["throughput_per_hour"]
        for row in learned_eval
    ]
    planner_wait_ratio = [
        row["planner_wait_time_total"] / baseline_eval[row["scenario_name"]]["planner_wait_time_total"]
        if baseline_eval[row["scenario_name"]]["planner_wait_time_total"] > 0
        else 0.0
        for row in learned_eval
    ]

    x = np.arange(len(scenarios))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    ax.bar(x - width / 2, throughput_ratio, width, label="Throughput ratio", color="#1f77b4")
    ax.bar(x + width / 2, planner_wait_ratio, width, label="Planner-wait ratio", color="#ff7f0e")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("Ratio vs. baseline")
    ax.set_title("Learned Policy Relative to Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    for idx, ratio in enumerate(throughput_ratio):
        ax.text(idx - width / 2, ratio + 0.03, f"{ratio:.2f}", ha="center", va="bottom", fontsize=8)
    for idx, ratio in enumerate(planner_wait_ratio):
        ax.text(idx + width / 2, ratio + 0.03, f"{ratio:.2f}", ha="center", va="bottom", fontsize=8)

    fig.savefig(FIGURE_DIR / "scenario_ratios.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_warm_start(warm_start_rows: list[dict]) -> None:
    epochs = [int(row["warm_start_epoch"]) for row in warm_start_rows]
    bc_loss = [float(row["mean_bc_loss"]) for row in warm_start_rows]
    match_rate = [float(row["teacher_action_match_rate"]) for row in warm_start_rows]

    fig, ax1 = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    ax2 = ax1.twinx()

    ax1.plot(epochs, bc_loss, marker="o", color="#1f77b4", linewidth=2.0, label="BC loss")
    ax2.plot(epochs, match_rate, marker="s", color="#d62728", linewidth=2.0, label="Teacher match")

    ax1.set_xlabel("Warm-start epoch")
    ax1.set_ylabel("Mean BC loss", color="#1f77b4")
    ax2.set_ylabel("Teacher action match rate", color="#d62728")
    ax1.set_title("Warm-Start Convergence")
    ax1.grid(alpha=0.25)
    ax1.set_xticks(epochs)
    ax2.set_ylim(0.99, 1.0)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="center right", frameon=False)

    fig.savefig(FIGURE_DIR / "warm_start_convergence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ppo_trace(training_rows: list[dict]) -> None:
    episodes = [int(row["episode"]) for row in training_rows]
    throughput = [float(row["throughput_per_hour"]) for row in training_rows]
    planner_wait = [float(row["planner_wait_time_total"]) for row in training_rows]
    safety = [float(row["safety_violations_total"]) for row in training_rows]

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.6), sharex=True, constrained_layout=True)

    axes[0].plot(episodes, throughput, color="#1f77b4", linewidth=2.0)
    axes[0].set_ylabel("Throughput/hr")
    axes[0].set_title("PPO Training Trace Across Episodes")
    axes[0].grid(alpha=0.25)

    axes[1].plot(episodes, planner_wait, color="#ff7f0e", linewidth=2.0)
    axes[1].set_ylabel("Planner wait")
    axes[1].grid(alpha=0.25)

    axes[2].plot(episodes, safety, color="#d62728", linewidth=2.0)
    axes[2].set_ylabel("Safety violations")
    axes[2].set_xlabel("Training episode")
    axes[2].grid(alpha=0.25)

    fig.savefig(FIGURE_DIR / "ppo_training_trace.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    learned_eval = load_learned_eval()
    baseline_eval = load_baseline_eval()
    warm_start_rows = load_warm_start_metrics()
    training_rows = load_training_metrics()

    plot_throughput_and_wait(learned_eval, baseline_eval)
    plot_ratios(learned_eval, baseline_eval)
    plot_warm_start(warm_start_rows)
    plot_ppo_trace(training_rows)


if __name__ == "__main__":
    main()
