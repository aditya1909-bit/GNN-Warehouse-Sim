"""Configuration models for RL fine-tuning of graph dispatch policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from warehouse_sim.config.models import ConfigValidationError


@dataclass(frozen=True)
class RLCurriculumConfig:
    """Scenario curriculum and seed splits for RL training."""

    scenario_configs: tuple[Path, ...]
    train_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.scenario_configs:
            raise ConfigValidationError("rl curriculum must include at least one scenario config.")
        if not self.train_seeds:
            raise ConfigValidationError("rl curriculum must include at least one train seed.")


@dataclass(frozen=True)
class RewardConfig:
    """Shaped reward coefficients for dispatch-event RL."""

    task_completion: float = 1.0
    waiting_time: float = -0.01
    congestion_delay: float = -0.02
    blocked_events: float = -0.05


@dataclass(frozen=True)
class PPOConfig:
    """PPO hyperparameters for graph dispatch fine-tuning."""

    learning_rate: float = 1e-4
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 3
    rollout_horizon: int = 4
    total_episodes: int = 12


@dataclass(frozen=True)
class RLFineTuningConfig:
    """Top-level RL fine-tuning configuration."""

    name: str
    pretrained_artifact_path: Path
    curriculum: RLCurriculumConfig
    reward: RewardConfig
    ppo: PPOConfig
    output_dir: Path

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigValidationError("rl fine-tuning name must be non-empty.")
