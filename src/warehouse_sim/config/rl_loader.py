"""TOML loader for RL fine-tuning configs."""

from __future__ import annotations

import tomllib
from pathlib import Path

from warehouse_sim.config.rl_models import PPOConfig, RLCurriculumConfig, RLFineTuningConfig, RewardConfig


def load_rl_fine_tuning_config(path: Path) -> RLFineTuningConfig:
    """Load an RL fine-tuning config from TOML."""

    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    curriculum = raw["curriculum"]
    reward = raw.get("reward", {})
    ppo = raw.get("ppo", {})
    output = raw.get("output", {})
    return RLFineTuningConfig(
        name=str(raw["name"]),
        pretrained_artifact_path=(path.parent / Path(str(raw["pretrained_artifact_path"]))).resolve(),
        curriculum=RLCurriculumConfig(
            scenario_configs=tuple(
                (path.parent / Path(str(item))).resolve()
                for item in curriculum["scenario_configs"]
            ),
            train_seeds=tuple(int(seed) for seed in curriculum.get("train_seeds", ())),
            validation_seeds=tuple(int(seed) for seed in curriculum.get("validation_seeds", ())),
        ),
        reward=RewardConfig(
            task_completion=float(reward.get("task_completion", 1.0)),
            waiting_time=float(reward.get("waiting_time", -0.01)),
            congestion_delay=float(reward.get("congestion_delay", -0.02)),
            blocked_events=float(reward.get("blocked_events", -0.05)),
        ),
        ppo=PPOConfig(
            learning_rate=float(ppo.get("learning_rate", 1e-4)),
            clip_epsilon=float(ppo.get("clip_epsilon", 0.2)),
            gamma=float(ppo.get("gamma", 0.99)),
            gae_lambda=float(ppo.get("gae_lambda", 0.95)),
            ppo_epochs=int(ppo.get("ppo_epochs", 3)),
            rollout_horizon=int(ppo.get("rollout_horizon", 4)),
            total_episodes=int(ppo.get("total_episodes", 12)),
        ),
        output_dir=(path.parent / Path(str(output.get("output_dir", "outputs/rl_fine_tuning")))).resolve(),
    )
