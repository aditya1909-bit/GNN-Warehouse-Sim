"""Tests for obstacle-aware continuous geometry helpers."""

from __future__ import annotations

from warehouse_sim.environment import ObstaclePolygon, obstacle_rectangles_from_blocked_cells
from warehouse_sim.integrated.geometry import (
    inflate_obstacles,
    polyline_distance,
    segment_has_line_of_sight,
    visibility_graph_k_shortest_paths,
    visibility_graph_shortest_path,
)


def test_inflate_obstacles_expands_blocked_cell_geometry() -> None:
    obstacle = obstacle_rectangles_from_blocked_cells(((1, 1),), edge_length=1.0)[0]

    inflated = inflate_obstacles((obstacle,), margin=0.25)[0]

    assert inflated.min_x == 0.25
    assert inflated.min_y == 0.25
    assert inflated.max_x == 1.75
    assert inflated.max_y == 1.75


def test_segment_has_line_of_sight_rejects_crossing_obstacle() -> None:
    obstacle = obstacle_rectangles_from_blocked_cells(((1, 1),), edge_length=1.0)[0]

    assert not segment_has_line_of_sight((0.0, 0.0), (2.0, 2.0), obstacles=(obstacle,))
    assert segment_has_line_of_sight((0.0, 0.0), (0.0, 2.0), obstacles=(obstacle,))


def test_visibility_graph_shortest_path_routes_around_inflated_obstacle() -> None:
    obstacle = obstacle_rectangles_from_blocked_cells(((1, 1),), edge_length=1.0)[0]
    inflated = inflate_obstacles((obstacle,), margin=0.25)

    path = visibility_graph_shortest_path((0.0, 0.0), (2.0, 2.0), obstacles=inflated)

    assert path is not None
    assert len(path) > 2
    assert polyline_distance(path) > 2.8284
    assert all(
        segment_has_line_of_sight(start, end, obstacles=inflated)
        for start, end in zip(path, path[1:])
    )


def test_visibility_graph_k_shortest_paths_returns_multiple_polygon_routes() -> None:
    obstacle = ObstaclePolygon(
        obstacle_id="center_block",
        vertices=((0.75, -0.25), (0.75, 0.25), (1.25, 0.25), (1.25, -0.25)),
    )

    paths = visibility_graph_k_shortest_paths(
        (0.0, 0.0),
        (2.0, 0.0),
        obstacles=inflate_obstacles((obstacle,), margin=0.05),
        k=2,
    )

    assert len(paths) == 2
    assert paths[0] != paths[1]
    assert all(len(path) > 2 for path in paths)
