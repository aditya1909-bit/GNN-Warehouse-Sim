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
    battery_capacity: float = 0.0
    initial_charge_fraction: float = 1.0

    def __post_init__(self) -> None:
        if not self.robot_id:
            raise RobotValidationError("robot_id must be non-empty.")
        if not self.initial_node:
            raise RobotValidationError("initial_node must be non-empty.")
        if self.speed_multiplier <= 0:
            raise RobotValidationError("speed_multiplier must be > 0.")
        if self.available_from < 0:
            raise RobotValidationError("available_from must be >= 0.")
        if self.battery_capacity < 0:
            raise RobotValidationError("battery_capacity must be >= 0.")
        if not 0.0 <= self.initial_charge_fraction <= 1.0:
            raise RobotValidationError("initial_charge_fraction must be between 0 and 1.")


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
    total_congestion_delay: float = 0.0
    blocked_traversal_events: int = 0
    battery_level: float = 0.0
    total_energy_consumed: float = 0.0
    total_energy_charged: float = 0.0
    total_charging_time: float = 0.0
    charging_events: int = 0
    battery_depletion_events: int = 0
    completed_task_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_spec(cls, spec: RobotSpec) -> "RobotState":
        """Create the initial mutable state for a robot spec."""

        return cls(
            spec=spec,
            current_node=spec.initial_node,
            available_time=spec.available_from,
            battery_level=spec.battery_capacity * spec.initial_charge_fraction,
        )
