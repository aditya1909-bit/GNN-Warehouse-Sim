"""TOML loader for integrated end-to-end PPO training configs."""

from __future__ import annotations

import tomllib
from pathlib import Path

from warehouse_sim.config.integrated_rl_models import (
    BenchmarkGateConfig,
    IntegratedModelConfig,
    IntegratedPPOConfig,
    IntegratedRLCurriculumConfig,
    IntegratedRewardConfig,
    IntegratedRLTrainingConfig,
    IntegratedRuntimeConfig,
    IntegratedWarmStartConfig,
)


def load_integrated_rl_training_config(path: Path) -> IntegratedRLTrainingConfig:
    """Load an integrated PPO training config from TOML."""

    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    curriculum = raw["curriculum"]
    model = raw.get("model", {})
    runtime = raw.get("runtime", {})
    reward = raw.get("reward", {})
    ppo = raw.get("ppo", {})
    warm_start = raw.get("warm_start", {})
    benchmark_gate = raw.get("benchmark_gate", {})
    output = raw.get("output", {})
    return IntegratedRLTrainingConfig(
        name=str(raw["name"]),
        curriculum=IntegratedRLCurriculumConfig(
            scenario_configs=tuple((path.parent / Path(str(item))).resolve() for item in curriculum["scenario_configs"]),
            train_seeds=tuple(int(seed) for seed in curriculum.get("train_seeds", ())),
            validation_seeds=tuple(int(seed) for seed in curriculum.get("validation_seeds", ())),
            scenario_weights={
                str(scenario_name): float(weight)
                for scenario_name, weight in curriculum.get("scenario_weights", {}).items()
            },
        ),
        model=IntegratedModelConfig(
            hidden_dim=int(model.get("hidden_dim", 64)),
            warehouse_message_passing_layers=int(model.get("warehouse_message_passing_layers", 1)),
            conflict_message_passing_layers=int(model.get("conflict_message_passing_layers", 2)),
            dropout=float(model.get("dropout", 0.0)),
            top_k_conflicting_robots=int(model.get("top_k_conflicting_robots", 4)),
        ),
        runtime=IntegratedRuntimeConfig(
            device=str(runtime.get("device", "mps")),
            rollout_workers=int(runtime.get("rollout_workers", 4)),
            episodes_per_sync=int(runtime.get("episodes_per_sync", 4)),
            inference_batch_size=int(runtime.get("inference_batch_size", 8)),
        ),
        reward=IntegratedRewardConfig(
            task_completion=float(reward.get("task_completion", 1.0)),
            waiting_time=float(reward.get("waiting_time", -0.01)),
            congestion_delay=float(reward.get("congestion_delay", -0.02)),
            safety_violation=float(reward.get("safety_violation", -1.0)),
            path_conflict=float(reward.get("path_conflict", -0.05)),
            planner_wait_time=float(reward.get("planner_wait_time", -0.02)),
            wait_insertion_time=float(reward.get("wait_insertion_time", -0.02)),
        ),
        ppo=IntegratedPPOConfig(
            learning_rate=float(ppo.get("learning_rate", 3e-4)),
            clip_epsilon=float(ppo.get("clip_epsilon", 0.2)),
            gamma=float(ppo.get("gamma", 0.99)),
            gae_lambda=float(ppo.get("gae_lambda", 0.95)),
            ppo_epochs=int(ppo.get("ppo_epochs", 3)),
            total_episodes=int(ppo.get("total_episodes", 12)),
            learner_minibatch_size=int(ppo.get("learner_minibatch_size", 32)),
        ),
        warm_start=IntegratedWarmStartConfig(
            epochs=int(warm_start.get("epochs", 0)),
            learning_rate=float(warm_start.get("learning_rate", 1e-3)),
            teacher_policy=str(warm_start.get("teacher_policy", "prioritized_sipp_coordinator")),
            teacher_mixture={
                str(policy_name): float(weight)
                for policy_name, weight in warm_start.get("teacher_mixture", {}).items()
            },
        ),
        benchmark_gate=BenchmarkGateConfig(
            max_safety_violations=int(benchmark_gate.get("max_safety_violations", 0)),
            min_task_completion_rate=float(benchmark_gate.get("min_task_completion_rate", 0.98)),
            min_throughput_ratio_vs_baseline=float(
                benchmark_gate.get("min_throughput_ratio_vs_baseline", 0.9)
            ),
            min_policy_distinctness_vs_teacher=float(
                benchmark_gate.get("min_policy_distinctness_vs_teacher", 0.0)
            ),
        ),
        output_dir=(path.parent / Path(str(output.get("output_dir", "outputs/integrated_rl")))).resolve(),
    )
