"""Synthetic layout builders for warehouse topology experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

from warehouse_sim.graph.models import (
    GraphValidationError,
    NodeType,
    WarehouseEdge,
    WarehouseGraph,
    WarehouseNode,
)

GridCoordinate = tuple[int, int]


@dataclass(frozen=True)
class SyntheticGridLayoutConfig:
    """Configuration for a rectangular warehouse grid layout."""

    rows: int
    columns: int
    edge_length: float = 1.0
    travel_speed: float = 1.0
    blocked_cells: tuple[GridCoordinate, ...] = ()
    special_node_types: dict[GridCoordinate, NodeType] = field(default_factory=dict)
    zone_labels: dict[GridCoordinate, str] = field(default_factory=dict)
    directed_edges: tuple[tuple[GridCoordinate, GridCoordinate], ...] = ()

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise GraphValidationError("rows must be > 0.")
        if self.columns <= 0:
            raise GraphValidationError("columns must be > 0.")
        if self.edge_length <= 0:
            raise GraphValidationError("edge_length must be > 0.")
        if self.travel_speed <= 0:
            raise GraphValidationError("travel_speed must be > 0.")

        blocked = set(self.blocked_cells)
        for cell in blocked:
            _validate_cell(cell=cell, config=self)

        for cell in self.special_node_types:
            _validate_cell(cell=cell, config=self)
        for cell in self.zone_labels:
            _validate_cell(cell=cell, config=self)
        for source, target in self.directed_edges:
            _validate_cell(cell=source, config=self)
            _validate_cell(cell=target, config=self)
            if source in blocked or target in blocked:
                raise GraphValidationError("directed_edges cannot reference blocked cells.")
            if _manhattan_distance(source, target) != 1:
                raise GraphValidationError("directed_edges must connect orthogonally adjacent cells.")


def build_synthetic_grid_layout(config: SyntheticGridLayoutConfig) -> WarehouseGraph:
    """Build a grid-based warehouse graph with weighted travel edges."""

    graph = WarehouseGraph()
    blocked = set(config.blocked_cells)
    one_way_pairs = set(config.directed_edges)
    travel_time = config.edge_length / config.travel_speed

    for row in range(config.rows):
        for column in range(config.columns):
            cell = (row, column)
            if cell in blocked:
                continue

            node_type = config.special_node_types.get(cell, NodeType.TRANSIT)
            graph.add_node(
                WarehouseNode(
                    node_id=grid_node_id(row=row, column=column),
                    x=column,
                    y=row,
                    node_type=node_type,
                    zone=config.zone_labels.get(cell),
                )
            )

    for row in range(config.rows):
        for column in range(config.columns):
            cell = (row, column)
            if cell in blocked:
                continue
            if config.special_node_types.get(cell) == NodeType.OBSTACLE:
                continue

            for neighbor in ((row + 1, column), (row, column + 1)):
                if not _is_active_cell(neighbor, config, blocked):
                    continue
                if config.special_node_types.get(neighbor) == NodeType.OBSTACLE:
                    continue

                source_id = grid_node_id(row=row, column=column)
                target_id = grid_node_id(row=neighbor[0], column=neighbor[1])
                if (cell, neighbor) in one_way_pairs:
                    graph.add_edge(
                        WarehouseEdge(
                            source=source_id,
                            target=target_id,
                            distance=config.edge_length,
                            travel_time=travel_time,
                            directed=True,
                        )
                    )
                elif (neighbor, cell) in one_way_pairs:
                    graph.add_edge(
                        WarehouseEdge(
                            source=target_id,
                            target=source_id,
                            distance=config.edge_length,
                            travel_time=travel_time,
                            directed=True,
                        )
                    )
                else:
                    graph.add_edge(
                        WarehouseEdge(
                            source=source_id,
                            target=target_id,
                            distance=config.edge_length,
                            travel_time=travel_time,
                            directed=False,
                        )
                    )

    return graph


def grid_node_id(row: int, column: int) -> str:
    """Return the canonical node id for a grid coordinate."""

    return f"r{row}_c{column}"


def _validate_cell(cell: GridCoordinate, config: SyntheticGridLayoutConfig) -> None:
    row, column = cell
    if row < 0 or row >= config.rows:
        raise GraphValidationError(f"row index out of bounds for cell {cell}.")
    if column < 0 or column >= config.columns:
        raise GraphValidationError(f"column index out of bounds for cell {cell}.")


def _is_active_cell(
    cell: GridCoordinate,
    config: SyntheticGridLayoutConfig,
    blocked: set[GridCoordinate],
) -> bool:
    row, column = cell
    if row < 0 or row >= config.rows or column < 0 or column >= config.columns:
        return False
    return cell not in blocked


def _manhattan_distance(first: GridCoordinate, second: GridCoordinate) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])

