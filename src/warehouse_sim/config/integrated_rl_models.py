"""Configuration models for integrated end-to-end PPO training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from warehouse_sim.config.models import ConfigValidationError


@dataclass(frozen=True)
class IntegratedRLCurriculumConfig:
    """Scenario curriculum for integrated PPO training."""

    scenario_configs: tuple[Path, ...]
    train_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.scenario_configs:
            raise ConfigValidationError("integrated RL curriculum must include at least one scenario config.")
        if not self.train_seeds:
            raise ConfigValidationError("integrated RL curriculum must include at least one train seed.")


@dataclass(frozen=True)
class IntegratedRewardConfig:
    """Reward coefficients for integrated end-to-end PPO."""

    task_completion: float = 1.0
    waiting_time: float = -0.01
    congestion_delay: float = -0.02
    safety_violation: float = -1.0


@dataclass(frozen=True)
class IntegratedPPOConfig:
    """PPO hyperparameters for integrated training."""

    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 3
    total_episodes: int = 12


@dataclass(frozen=True)
class IntegratedWarmStartConfig:
    """Behavior-cloning warm start before PPO fine-tuning."""

    epochs: int = 0
    learning_rate: float = 1e-3
    teacher_policy: str = "prioritized_sipp_coordinator"

    def __post_init__(self) -> None:
        if self.epochs < 0:
            raise ConfigValidationError("integrated warm_start.epochs must be >= 0.")
        if self.learning_rate <= 0:
            raise ConfigValidationError("integrated warm_start.learning_rate must be > 0.")
        if self.teacher_policy != "prioritized_sipp_coordinator":
            raise ConfigValidationError(
                "integrated warm_start.teacher_policy must currently be prioritized_sipp_coordinator."
            )


@dataclass(frozen=True)
class BenchmarkGateConfig:
    """Thresholds for stronger learned-coordination claims."""

    max_safety_violations: int = 0
    min_task_completion_rate: float = 0.98
    min_throughput_ratio_vs_baseline: float = 0.9


@dataclass(frozen=True)
class IntegratedRLTrainingConfig:
    """Top-level config for integrated end-to-end PPO training."""

    name: str
    curriculum: IntegratedRLCurriculumConfig
    reward: IntegratedRewardConfig
    ppo: IntegratedPPOConfig
    warm_start: IntegratedWarmStartConfig
    benchmark_gate: BenchmarkGateConfig
    output_dir: Path

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigValidationError("integrated RL training name must be non-empty.")
