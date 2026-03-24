"""Tests for synthetic warehouse graph layouts and pathfinding."""

from __future__ import annotations

import pytest

from warehouse_sim.graph import (
    GraphValidationError,
    NodeType,
    PathNotFoundError,
    SyntheticGridLayoutConfig,
    build_synthetic_grid_layout,
)


def test_grid_layout_builds_expected_nodes_and_edges() -> None:
    graph = build_synthetic_grid_layout(
        SyntheticGridLayoutConfig(
            rows=2,
            columns=3,
            edge_length=2.0,
            travel_speed=1.0,
            special_node_types={(0, 0): NodeType.STORAGE, (1, 2): NodeType.PICK_STATION},
            zone_labels={(0, 0): "storage_zone", (1, 2): "pick_zone"},
        )
    )

    assert len(graph.nodes()) == 6
    assert len(graph.edges()) == 7
    assert graph.node("r0_c0").node_type == NodeType.STORAGE
    assert graph.node("r1_c2").zone == "pick_zone"
    assert graph.distance("r0_c0", "r1_c2") == pytest.approx(6.0)


def test_shortest_path_respects_one_way_edges() -> None:
    graph = build_synthetic_grid_layout(
        SyntheticGridLayoutConfig(
            rows=1,
            columns=3,
            directed_edges=(((0, 0), (0, 1)),),
        )
    )

    assert graph.shortest_path("r0_c0", "r0_c2") == ("r0_c0", "r0_c1", "r0_c2")
    with pytest.raises(PathNotFoundError):
        graph.shortest_path("r0_c1", "r0_c0")


def test_blocked_and_obstacle_cells_break_paths() -> None:
    graph = build_synthetic_grid_layout(
        SyntheticGridLayoutConfig(
            rows=2,
            columns=2,
            blocked_cells=((0, 1),),
            special_node_types={(1, 0): NodeType.OBSTACLE},
        )
    )

    assert len(graph.nodes()) == 3
    with pytest.raises(PathNotFoundError):
        graph.shortest_path("r0_c0", "r1_c1")


def test_invalid_directed_edge_definition_raises() -> None:
    with pytest.raises(GraphValidationError):
        SyntheticGridLayoutConfig(
            rows=2,
            columns=2,
            directed_edges=(((0, 0), (1, 1)),),
        )

