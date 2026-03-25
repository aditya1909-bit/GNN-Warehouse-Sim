"""Tests for graph featurization helpers."""

from __future__ import annotations

from warehouse_sim.graph import (
    SyntheticGridLayoutConfig,
    WarehouseEdge,
    WarehouseGraph,
    WarehouseNode,
    build_graph_features,
    build_synthetic_grid_layout,
)


def test_build_graph_features_expands_undirected_edges_to_directed_arcs() -> None:
    graph = WarehouseGraph()
    graph.add_node(WarehouseNode(node_id="a", x=0, y=0, zone="zone_a"))
    graph.add_node(WarehouseNode(node_id="b", x=1, y=0, zone="zone_b"))
    graph.add_edge(WarehouseEdge(source="a", target="b", distance=1.0, travel_time=2.0, directed=False))

    features = build_graph_features(graph)

    assert [arc.source_id for arc in features.arcs] == ["a", "b"]
    assert [arc.target_id for arc in features.arcs] == ["b", "a"]
    assert features.nodes[0].outbound_degree == 1
    assert features.nodes[0].inbound_degree == 1
    assert features.nodes[0].shortest_path_transit_count == 0
    assert features.arcs[0].shortest_path_traversal_count == 1
    assert features.arcs[1].shortest_path_traversal_count == 1


def test_build_graph_features_uses_zone_lookup_when_provided() -> None:
    graph = build_synthetic_grid_layout(SyntheticGridLayoutConfig(rows=1, columns=2))

    features = build_graph_features(
        graph,
        zone_lookup=lambda node_id: "custom_zone" if node_id == "r0_c0" else None,
    )

    zone_by_node = {node.node_id: node.zone_id for node in features.nodes}
    assert zone_by_node["r0_c0"] == "custom_zone"
