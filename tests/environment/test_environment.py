"""Tests for the warehouse environment abstraction."""

from __future__ import annotations

import pytest

from warehouse_sim.environment import WarehouseEnvironment, Zone
from warehouse_sim.graph import SyntheticGridLayoutConfig, build_synthetic_grid_layout


def test_environment_derives_zones_from_node_metadata() -> None:
    graph = build_synthetic_grid_layout(
        SyntheticGridLayoutConfig(
            rows=2,
            columns=2,
            zone_labels={(0, 0): "storage_zone", (1, 1): "dropoff_zone"},
        )
    )
    environment = WarehouseEnvironment(graph=graph)

    assert {zone.zone_id for zone in environment.zones()} == {"dropoff_zone", "storage_zone"}
    assert environment.default_node_for_zone("storage_zone").node_id == "r0_c0"
    assert environment.zone_for_node("r1_c0") is None


def test_environment_rejects_duplicate_zone_membership() -> None:
    graph = build_synthetic_grid_layout(SyntheticGridLayoutConfig(rows=1, columns=2))

    with pytest.raises(ValueError):
        WarehouseEnvironment(
            graph=graph,
            zones=(
                Zone(zone_id="zone_a", node_ids=("r0_c0",)),
                Zone(zone_id="zone_b", node_ids=("r0_c0",)),
            ),
        )

