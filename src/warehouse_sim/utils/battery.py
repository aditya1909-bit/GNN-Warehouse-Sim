"""Battery and charging helpers shared across simulation stacks."""

from __future__ import annotations

from dataclasses import dataclass

from warehouse_sim.environment import WarehouseEnvironment


@dataclass(frozen=True)
class ChargingOption:
    """Nearest charging option from a node."""

    charging_node_id: str
    distance: float
    travel_time: float
    energy_required: float


@dataclass(frozen=True)
class BatteryActionEstimate:
    """Energy feasibility summary for one action candidate."""

    action_type: str
    estimated_action_energy: float
    battery_after_action: float
    post_action_battery_fraction: float
    charger_reachable_after_action: bool
    nearest_charging_node_id: str | None
    nearest_charging_distance: float | None
    nearest_charging_travel_time: float | None
    charge_duration_to_full: float | None


def battery_enabled(battery_config) -> bool:
    return bool(battery_config is not None and getattr(battery_config, "enabled", False))


def battery_fraction(*, battery_level: float, battery_config) -> float:
    if not battery_enabled(battery_config):
        return 1.0
    return max(min(battery_level / max(float(battery_config.capacity), 1e-9), 1.0), 0.0)


def reserve_energy(battery_config) -> float:
    return float(battery_config.capacity) * float(battery_config.minimum_reserve_fraction)


def travel_energy(distance: float, battery_config) -> float:
    return max(distance, 0.0) * float(battery_config.travel_energy_per_distance)


def nearest_charging_option(
    environment: WarehouseEnvironment,
    *,
    source_node: str,
    speed_multiplier: float,
    battery_config,
) -> ChargingOption | None:
    charging_nodes = environment.charging_nodes()
    if not charging_nodes:
        return None
    best: tuple[float, str, float] | None = None
    for node in charging_nodes:
        distance = environment.distance(source_node, node.node_id)
        travel_time = environment.travel_time(source_node, node.node_id) / max(speed_multiplier, 1e-9)
        ranking = (travel_time, node.node_id, distance)
        if best is None or ranking < best:
            best = ranking
    if best is None:
        return None
    travel_time, node_id, distance = best
    return ChargingOption(
        charging_node_id=node_id,
        distance=distance,
        travel_time=travel_time,
        energy_required=travel_energy(distance, battery_config),
    )


def estimate_task_action(
    environment: WarehouseEnvironment,
    *,
    robot_node: str,
    pickup_node: str,
    dropoff_node: str,
    battery_level: float,
    speed_multiplier: float,
    battery_config,
) -> BatteryActionEstimate | None:
    if not battery_enabled(battery_config):
        return BatteryActionEstimate(
            action_type="task",
            estimated_action_energy=0.0,
            battery_after_action=battery_level,
            post_action_battery_fraction=1.0,
            charger_reachable_after_action=True,
            nearest_charging_node_id=None,
            nearest_charging_distance=None,
            nearest_charging_travel_time=None,
            charge_duration_to_full=None,
        )

    travel_distance = (
        environment.distance(robot_node, pickup_node)
        + environment.distance(pickup_node, dropoff_node)
    )
    action_energy = travel_energy(travel_distance, battery_config) + float(battery_config.service_energy)
    battery_after_action = battery_level - action_energy
    nearest_charge = nearest_charging_option(
        environment,
        source_node=dropoff_node,
        speed_multiplier=speed_multiplier,
        battery_config=battery_config,
    )
    reachable_after_action = False
    if nearest_charge is not None:
        reachable_after_action = battery_after_action >= nearest_charge.energy_required + reserve_energy(battery_config)
    return BatteryActionEstimate(
        action_type="task",
        estimated_action_energy=action_energy,
        battery_after_action=battery_after_action,
        post_action_battery_fraction=battery_fraction(
            battery_level=battery_after_action,
            battery_config=battery_config,
        ),
        charger_reachable_after_action=reachable_after_action,
        nearest_charging_node_id=None if nearest_charge is None else nearest_charge.charging_node_id,
        nearest_charging_distance=None if nearest_charge is None else nearest_charge.distance,
        nearest_charging_travel_time=None if nearest_charge is None else nearest_charge.travel_time,
        charge_duration_to_full=None,
    )


def estimate_charge_action(
    environment: WarehouseEnvironment,
    *,
    robot_node: str,
    charging_node_id: str,
    battery_level: float,
    speed_multiplier: float,
    battery_config,
) -> BatteryActionEstimate:
    distance = environment.distance(robot_node, charging_node_id)
    action_energy = travel_energy(distance, battery_config)
    battery_after_travel = battery_level - action_energy
    charge_duration = max(float(battery_config.capacity) - battery_after_travel, 0.0) / max(
        float(battery_config.charge_rate),
        1e-9,
    )
    return BatteryActionEstimate(
        action_type="charge",
        estimated_action_energy=action_energy,
        battery_after_action=float(battery_config.capacity),
        post_action_battery_fraction=1.0,
        charger_reachable_after_action=battery_after_travel >= reserve_energy(battery_config),
        nearest_charging_node_id=charging_node_id,
        nearest_charging_distance=distance,
        nearest_charging_travel_time=environment.travel_time(robot_node, charging_node_id) / max(speed_multiplier, 1e-9),
        charge_duration_to_full=charge_duration,
    )
