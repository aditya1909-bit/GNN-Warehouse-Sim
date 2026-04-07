"""Continuous geometry helpers for obstacle-aware integrated motion."""

from __future__ import annotations

from heapq import heappop, heappush
from math import hypot

from warehouse_sim.environment import ObstacleRectangle

Point2D = tuple[float, float]

_GEOMETRY_EPSILON = 1e-6


def inflate_obstacles(
    obstacles: tuple[ObstacleRectangle, ...],
    *,
    margin: float,
) -> tuple[ObstacleRectangle, ...]:
    """Return obstacle rectangles expanded by the requested margin."""

    return tuple(obstacle.inflate(margin) for obstacle in obstacles)


def segment_has_line_of_sight(
    start: Point2D,
    end: Point2D,
    *,
    obstacles: tuple[ObstacleRectangle, ...],
) -> bool:
    """Return whether a straight segment avoids all obstacle interiors."""

    return not any(segment_intersects_rectangle(start, end, obstacle) for obstacle in obstacles)


def segment_intersects_rectangle(
    start: Point2D,
    end: Point2D,
    obstacle: ObstacleRectangle,
) -> bool:
    """Return whether a segment intersects or touches an obstacle rectangle."""

    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    entering = 0.0
    exiting = 1.0

    for p_value, q_value in (
        (-dx, x0 - obstacle.min_x),
        (dx, obstacle.max_x - x0),
        (-dy, y0 - obstacle.min_y),
        (dy, obstacle.max_y - y0),
    ):
        if abs(p_value) <= _GEOMETRY_EPSILON:
            if q_value < 0.0:
                return False
            continue
        ratio = q_value / p_value
        if p_value < 0.0:
            entering = max(entering, ratio)
        else:
            exiting = min(exiting, ratio)
        if entering - exiting > _GEOMETRY_EPSILON:
            return False
    return True


def point_in_obstacle(point: Point2D, obstacle: ObstacleRectangle) -> bool:
    """Return whether a point lies inside or on the boundary of an obstacle."""

    x, y = point
    return (
        obstacle.min_x - _GEOMETRY_EPSILON <= x <= obstacle.max_x + _GEOMETRY_EPSILON
        and obstacle.min_y - _GEOMETRY_EPSILON <= y <= obstacle.max_y + _GEOMETRY_EPSILON
    )


def visibility_graph_shortest_path(
    start: Point2D,
    goal: Point2D,
    *,
    obstacles: tuple[ObstacleRectangle, ...],
) -> tuple[Point2D, ...] | None:
    """Find the shortest obstacle-avoiding polyline over rectangle-corner waypoints."""

    if start == goal:
        return (start,)
    if any(point_in_obstacle(start, obstacle) or point_in_obstacle(goal, obstacle) for obstacle in obstacles):
        return None

    vertices = _deduplicate_points((start, goal, *tuple(_visibility_vertices(obstacles))))
    adjacency: dict[int, list[tuple[int, float]]] = {index: [] for index in range(len(vertices))}
    for left_index, left_point in enumerate(vertices):
        for right_index in range(left_index + 1, len(vertices)):
            right_point = vertices[right_index]
            if not segment_has_line_of_sight(left_point, right_point, obstacles=obstacles):
                continue
            distance = hypot(right_point[0] - left_point[0], right_point[1] - left_point[1])
            adjacency[left_index].append((right_index, distance))
            adjacency[right_index].append((left_index, distance))

    goal_index = 1
    queue: list[tuple[float, int]] = [(0.0, 0)]
    distances = {0: 0.0}
    parents: dict[int, int | None] = {0: None}

    while queue:
        distance_so_far, vertex_index = heappop(queue)
        if distance_so_far > distances.get(vertex_index, float("inf")) + _GEOMETRY_EPSILON:
            continue
        if vertex_index == goal_index:
            return _reconstruct_path(vertices, parents, goal_index)
        for neighbor_index, edge_cost in adjacency[vertex_index]:
            next_distance = distance_so_far + edge_cost
            if next_distance + _GEOMETRY_EPSILON >= distances.get(neighbor_index, float("inf")):
                continue
            distances[neighbor_index] = next_distance
            parents[neighbor_index] = vertex_index
            heappush(queue, (next_distance, neighbor_index))
    return None


def polyline_distance(points: tuple[Point2D, ...]) -> float:
    """Return the Euclidean length of a piecewise-linear path."""

    return sum(hypot(end[0] - start[0], end[1] - start[1]) for start, end in zip(points, points[1:]))


def _visibility_vertices(obstacles: tuple[ObstacleRectangle, ...]) -> tuple[Point2D, ...]:
    vertices: list[Point2D] = []
    for obstacle in obstacles:
        vertices.extend(
            (
                (obstacle.min_x - _GEOMETRY_EPSILON, obstacle.min_y - _GEOMETRY_EPSILON),
                (obstacle.min_x - _GEOMETRY_EPSILON, obstacle.max_y + _GEOMETRY_EPSILON),
                (obstacle.max_x + _GEOMETRY_EPSILON, obstacle.min_y - _GEOMETRY_EPSILON),
                (obstacle.max_x + _GEOMETRY_EPSILON, obstacle.max_y + _GEOMETRY_EPSILON),
            )
        )
    return tuple(vertices)


def _deduplicate_points(points: tuple[Point2D, ...]) -> tuple[Point2D, ...]:
    seen: set[tuple[int, int]] = set()
    deduplicated: list[Point2D] = []
    for x, y in points:
        key = (round(x / _GEOMETRY_EPSILON), round(y / _GEOMETRY_EPSILON))
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append((x, y))
    return tuple(deduplicated)


def _reconstruct_path(
    vertices: tuple[Point2D, ...],
    parents: dict[int, int | None],
    goal_index: int,
) -> tuple[Point2D, ...]:
    path: list[Point2D] = []
    cursor: int | None = goal_index
    while cursor is not None:
        path.append(vertices[cursor])
        cursor = parents.get(cursor)
    path.reverse()
    return tuple(path)
