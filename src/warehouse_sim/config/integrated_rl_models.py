"""Configuration models for integrated dense-traffic macro PPO training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from warehouse_sim.config.models import ConfigValidationError


@dataclass(frozen=True)
class IntegratedRLCurriculumConfig:
    """Scenario curriculum for integrated PPO training."""

    scenario_configs: tuple[Path, ...]
    train_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    scenario_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scenario_configs:
            raise ConfigValidationError("integrated RL curriculum must include at least one scenario config.")
        if not self.train_seeds:
            raise ConfigValidationError("integrated RL curriculum must include at least one train seed.")
        for scenario_name, weight in self.scenario_weights.items():
            if not scenario_name:
                raise ConfigValidationError("integrated RL curriculum scenario_weights keys must be non-empty.")
            if weight <= 0:
                raise ConfigValidationError("integrated RL curriculum scenario_weights values must be > 0.")


@dataclass(frozen=True)
class IntegratedRewardConfig:
    """Reward coefficients for integrated end-to-end PPO."""

    task_completion: float = 1.0
    waiting_time: float = -0.01
    congestion_delay: float = -0.02
    safety_violation: float = -1.0
    path_conflict: float = -0.05
    planner_wait_time: float = -0.02
    wait_insertion_time: float = -0.02


@dataclass(frozen=True)
class IntegratedModelConfig:
    """Architecture parameters for conflict-aware integrated PPO."""

    hidden_dim: int = 64
    warehouse_message_passing_layers: int = 1
    conflict_message_passing_layers: int = 2
    dropout: float = 0.0
    top_k_conflicting_robots: int = 4

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ConfigValidationError("integrated model.hidden_dim must be > 0.")
        if self.warehouse_message_passing_layers <= 0:
            raise ConfigValidationError("integrated model.warehouse_message_passing_layers must be > 0.")
        if self.conflict_message_passing_layers <= 0:
            raise ConfigValidationError("integrated model.conflict_message_passing_layers must be > 0.")
        if self.dropout < 0:
            raise ConfigValidationError("integrated model.dropout must be >= 0.")
        if self.top_k_conflicting_robots <= 0:
            raise ConfigValidationError("integrated model.top_k_conflicting_robots must be > 0.")


@dataclass(frozen=True)
class IntegratedPPOConfig:
    """PPO hyperparameters for integrated training."""

    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 3
    total_episodes: int = 12
    learner_minibatch_size: int = 32

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ConfigValidationError("integrated ppo.learning_rate must be > 0.")
        if self.ppo_epochs <= 0:
            raise ConfigValidationError("integrated ppo.ppo_epochs must be > 0.")
        if self.total_episodes <= 0:
            raise ConfigValidationError("integrated ppo.total_episodes must be > 0.")
        if self.learner_minibatch_size <= 0:
            raise ConfigValidationError("integrated ppo.learner_minibatch_size must be > 0.")


@dataclass(frozen=True)
class IntegratedRuntimeConfig:
    """Runtime parallelism for Apple Silicon training and evaluation."""

    device: str = "mps"
    rollout_workers: int = 4
    episodes_per_sync: int = 4
    inference_batch_size: int = 8

    def __post_init__(self) -> None:
        if self.device not in {"mps", "cpu"}:
            raise ConfigValidationError("integrated runtime.device must be mps or cpu.")
        if self.rollout_workers <= 0:
            raise ConfigValidationError("integrated runtime.rollout_workers must be > 0.")
        if self.episodes_per_sync <= 0:
            raise ConfigValidationError("integrated runtime.episodes_per_sync must be > 0.")
        if self.inference_batch_size <= 0:
            raise ConfigValidationError("integrated runtime.inference_batch_size must be > 0.")


@dataclass(frozen=True)
class IntegratedWarmStartConfig:
    """Behavior-cloning warm start before PPO fine-tuning."""

    epochs: int = 0
    learning_rate: float = 1e-3
    teacher_policy: str = "prioritized_sipp_coordinator"
    teacher_mixture: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        supported_teacher_policies = {
            "prioritized_sipp_coordinator",
            "optimal_mapf_coordinator",
        }
        if self.epochs < 0:
            raise ConfigValidationError("integrated warm_start.epochs must be >= 0.")
        if self.learning_rate <= 0:
            raise ConfigValidationError("integrated warm_start.learning_rate must be > 0.")
        if self.teacher_policy not in supported_teacher_policies:
            raise ConfigValidationError(
                "integrated warm_start.teacher_policy must be prioritized_sipp_coordinator or optimal_mapf_coordinator."
            )
        for teacher_name, weight in self.teacher_mixture.items():
            if teacher_name not in supported_teacher_policies:
                raise ConfigValidationError(
                    "integrated warm_start.teacher_mixture contains an unsupported policy."
                )
            if weight <= 0:
                raise ConfigValidationError("integrated warm_start.teacher_mixture values must be > 0.")


@dataclass(frozen=True)
class BenchmarkGateConfig:
    """Thresholds for stronger learned-coordination claims."""

    max_safety_violations: int = 0
    min_task_completion_rate: float = 0.98
    min_throughput_ratio_vs_baseline: float = 0.9
    min_policy_distinctness_vs_teacher: float = 0.0

    def __post_init__(self) -> None:
        if self.max_safety_violations < 0:
            raise ConfigValidationError("integrated benchmark_gate.max_safety_violations must be >= 0.")
        if not 0.0 <= self.min_task_completion_rate <= 1.0:
            raise ConfigValidationError(
                "integrated benchmark_gate.min_task_completion_rate must be between 0 and 1."
            )
        if self.min_throughput_ratio_vs_baseline < 0.0:
            raise ConfigValidationError(
                "integrated benchmark_gate.min_throughput_ratio_vs_baseline must be >= 0."
            )
        if not 0.0 <= self.min_policy_distinctness_vs_teacher <= 1.0:
            raise ConfigValidationError(
                "integrated benchmark_gate.min_policy_distinctness_vs_teacher must be between 0 and 1."
            )


@dataclass(frozen=True)
class IntegratedRLTrainingConfig:
    """Top-level config for integrated end-to-end PPO training."""

    name: str
    curriculum: IntegratedRLCurriculumConfig
    model: IntegratedModelConfig
    runtime: IntegratedRuntimeConfig
    reward: IntegratedRewardConfig
    ppo: IntegratedPPOConfig
    warm_start: IntegratedWarmStartConfig
    benchmark_gate: BenchmarkGateConfig
    output_dir: Path

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigValidationError("integrated RL training name must be non-empty.")
