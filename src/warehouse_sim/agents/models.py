"""Robot abstractions for the first simulation baseline."""

from __future__ import annotations

from dataclasses import dataclass, field


class RobotValidationError(ValueError):
    """Raised when a robot definition is invalid."""


@dataclass(frozen=True)
class RobotSpec:
    """Immutable robot definition used to initialize simulations."""

    robot_id: str
    initial_node: str
    speed_multiplier: float = 1.0
    available_from: float = 0.0

    def __post_init__(self) -> None:
        if not self.robot_id:
            raise RobotValidationError("robot_id must be non-empty.")
        if not self.initial_node:
            raise RobotValidationError("initial_node must be non-empty.")
        if self.speed_multiplier <= 0:
            raise RobotValidationError("speed_multiplier must be > 0.")
        if self.available_from < 0:
            raise RobotValidationError("available_from must be >= 0.")


@dataclass
class RobotState:
    """Mutable robot state tracked during a simulation run."""

    spec: RobotSpec
    current_node: str
    available_time: float
    total_busy_time: float = 0.0
    total_idle_time: float = 0.0
    total_travel_time: float = 0.0
    total_travel_distance: float = 0.0
    completed_task_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_spec(cls, spec: RobotSpec) -> "RobotState":
        """Create the initial mutable state for a robot spec."""

        return cls(
            spec=spec,
            current_node=spec.initial_node,
            available_time=spec.available_from,
        )

