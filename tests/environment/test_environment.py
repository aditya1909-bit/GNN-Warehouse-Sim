"""Tests for the warehouse environment abstraction."""

from __future__ import annotations

import pytest

from warehouse_sim.environment import (
    ObstaclePolygon,
    WarehouseEnvironment,
    Zone,
    obstacle_rectangles_from_blocked_cells,
)
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


def test_environment_retains_obstacle_geometry_from_blocked_cells() -> None:
    graph = build_synthetic_grid_layout(
        SyntheticGridLayoutConfig(
            rows=3,
            columns=3,
            edge_length=2.0,
            blocked_cells=((1, 1),),
        )
    )
    environment = WarehouseEnvironment(
        graph=graph,
        obstacles=obstacle_rectangles_from_blocked_cells(((1, 1),), edge_length=2.0),
    )

    obstacles = environment.obstacles()
    assert len(obstacles) == 1
    assert obstacles[0].obstacle_id == "blocked_r1_c1"
    assert obstacles[0].min_x == 1.0
    assert obstacles[0].min_y == 1.0
    assert obstacles[0].max_x == 3.0
    assert obstacles[0].max_y == 3.0


def test_environment_retains_explicit_polygon_obstacle_geometry() -> None:
    graph = build_synthetic_grid_layout(SyntheticGridLayoutConfig(rows=3, columns=3))
    polygon = ObstaclePolygon(
        obstacle_id="polygon_1",
        vertices=((0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)),
    )

    environment = WarehouseEnvironment(graph=graph, obstacles=(polygon,))

    obstacles = environment.obstacles()
    assert len(obstacles) == 1
    assert obstacles[0].obstacle_id == "polygon_1"
    assert obstacles[0].vertices == polygon.vertices
