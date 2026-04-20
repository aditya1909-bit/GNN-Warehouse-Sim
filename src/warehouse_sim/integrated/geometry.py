"""Continuous geometry helpers for obstacle-aware integrated motion."""

from __future__ import annotations

from heapq import heappop, heappush
from math import hypot

from warehouse_sim.environment import ObstaclePolygon, ObstacleRectangle

Point2D = tuple[float, float]
ObstacleGeometry = ObstacleRectangle | ObstaclePolygon

_GEOMETRY_EPSILON = 1e-6
_VISIBILITY_NUDGE = 1e-4


def inflate_obstacles(
    obstacles: tuple[ObstacleGeometry, ...],
    *,
    margin: float,
) -> tuple[ObstacleGeometry, ...]:
    """Return obstacle geometry expanded by the requested margin."""

    return tuple(obstacle.inflate(margin) for obstacle in obstacles)


def segment_has_line_of_sight(
    start: Point2D,
    end: Point2D,
    *,
    obstacles: tuple[ObstacleGeometry, ...],
) -> bool:
    """Return whether a straight segment avoids all obstacle interiors."""

    return not any(segment_intersects_obstacle(start, end, obstacle) for obstacle in obstacles)


def segment_intersects_obstacle(
    start: Point2D,
    end: Point2D,
    obstacle: ObstacleGeometry,
) -> bool:
    """Return whether a segment intersects or touches obstacle geometry."""

    vertices = obstacle_vertices(obstacle)
    if point_in_obstacle(start, obstacle) or point_in_obstacle(end, obstacle):
        return True
    for edge_start, edge_end in zip(vertices, (*vertices[1:], vertices[0])):
        if _segments_intersect(start, end, edge_start, edge_end):
            return True
    midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    return point_in_obstacle(midpoint, obstacle)


def obstacle_vertices(obstacle: ObstacleGeometry) -> tuple[Point2D, ...]:
    """Return the polygon vertex sequence for obstacle geometry."""

    return obstacle.vertices


def point_in_obstacle(point: Point2D, obstacle: ObstacleGeometry) -> bool:
    """Return whether a point lies inside or on the boundary of an obstacle."""

    vertices = obstacle_vertices(obstacle)
    for edge_start, edge_end in zip(vertices, (*vertices[1:], vertices[0])):
        if _point_on_segment(point, edge_start, edge_end):
            return True

    x_value, y_value = point
    inside = False
    previous_x, previous_y = vertices[-1]
    for current_x, current_y in vertices:
        intersects = ((current_y > y_value) != (previous_y > y_value)) and (
            x_value < ((previous_x - current_x) * (y_value - current_y) / (previous_y - current_y)) + current_x
        )
        if intersects:
            inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def visibility_graph_shortest_path(
    start: Point2D,
    goal: Point2D,
    *,
    obstacles: tuple[ObstacleGeometry, ...],
) -> tuple[Point2D, ...] | None:
    """Find the shortest obstacle-avoiding polyline over rectangle-corner waypoints."""

    paths = visibility_graph_k_shortest_paths(start, goal, obstacles=obstacles, k=1)
    return None if not paths else paths[0]


def visibility_graph_k_shortest_paths(
    start: Point2D,
    goal: Point2D,
    *,
    obstacles: tuple[ObstacleGeometry, ...],
    k: int,
) -> tuple[tuple[Point2D, ...], ...]:
    """Enumerate up to k simple visibility-graph paths in ascending distance order."""

    if k <= 0:
        return ()
    if start == goal:
        return ((start,),)
    if any(point_in_obstacle(start, obstacle) or point_in_obstacle(goal, obstacle) for obstacle in obstacles):
        return ()

    vertices, adjacency = _visibility_graph(start, goal, obstacles=obstacles)
    goal_index = 1
    queue: list[tuple[float, tuple[int, ...]]] = [(0.0, (0,))]
    results: list[tuple[Point2D, ...]] = []
    seen_paths: set[tuple[tuple[int, int], ...]] = set()

    while queue and len(results) < k:
        distance_so_far, path = heappop(queue)
        vertex_index = path[-1]
        if vertex_index == goal_index:
            points = tuple(vertices[index] for index in path)
            key = _point_path_key(points)
            if key not in seen_paths:
                seen_paths.add(key)
                results.append(points)
            continue
        for neighbor_index, edge_cost in adjacency[vertex_index]:
            if neighbor_index in path:
                continue
            heappush(queue, (distance_so_far + edge_cost, (*path, neighbor_index)))
    return tuple(results)


def polyline_distance(points: tuple[Point2D, ...]) -> float:
    """Return the Euclidean length of a piecewise-linear path."""

    return sum(hypot(end[0] - start[0], end[1] - start[1]) for start, end in zip(points, points[1:]))


def _visibility_vertices(obstacles: tuple[ObstacleGeometry, ...]) -> tuple[Point2D, ...]:
    vertices: list[Point2D] = []
    for obstacle in obstacles:
        vertices.extend(_nudged_polygon_vertices(obstacle_vertices(obstacle)))
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


def _visibility_graph(
    start: Point2D,
    goal: Point2D,
    *,
    obstacles: tuple[ObstacleGeometry, ...],
) -> tuple[tuple[Point2D, ...], dict[int, list[tuple[int, float]]]]:
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
    return vertices, adjacency


def _nudged_polygon_vertices(vertices: tuple[Point2D, ...]) -> tuple[Point2D, ...]:
    center_x = sum(point[0] for point in vertices) / len(vertices)
    center_y = sum(point[1] for point in vertices) / len(vertices)
    nudged: list[Point2D] = []
    for x_value, y_value in vertices:
        dx = x_value - center_x
        dy = y_value - center_y
        norm = hypot(dx, dy)
        if norm <= _GEOMETRY_EPSILON:
            nudged.append((x_value, y_value))
            continue
        nudged.append(
            (
                x_value + _VISIBILITY_NUDGE * dx / norm,
                y_value + _VISIBILITY_NUDGE * dy / norm,
            )
        )
    return tuple(nudged)


def _point_path_key(points: tuple[Point2D, ...]) -> tuple[tuple[int, int], ...]:
    return tuple((round(x_value / _GEOMETRY_EPSILON), round(y_value / _GEOMETRY_EPSILON)) for x_value, y_value in points)


def _point_on_segment(point: Point2D, start: Point2D, end: Point2D) -> bool:
    cross = _cross(start, end, point)
    if abs(cross) > _GEOMETRY_EPSILON:
        return False
    return (
        min(start[0], end[0]) - _GEOMETRY_EPSILON <= point[0] <= max(start[0], end[0]) + _GEOMETRY_EPSILON
        and min(start[1], end[1]) - _GEOMETRY_EPSILON <= point[1] <= max(start[1], end[1]) + _GEOMETRY_EPSILON
    )


def _segments_intersect(left_start: Point2D, left_end: Point2D, right_start: Point2D, right_end: Point2D) -> bool:
    left_1 = _cross(left_start, left_end, right_start)
    left_2 = _cross(left_start, left_end, right_end)
    right_1 = _cross(right_start, right_end, left_start)
    right_2 = _cross(right_start, right_end, left_end)

    if _point_on_segment(left_start, right_start, right_end):
        return True
    if _point_on_segment(left_end, right_start, right_end):
        return True
    if _point_on_segment(right_start, left_start, left_end):
        return True
    if _point_on_segment(right_end, left_start, left_end):
        return True

    return (
        (left_1 > _GEOMETRY_EPSILON and left_2 < -_GEOMETRY_EPSILON)
        or (left_1 < -_GEOMETRY_EPSILON and left_2 > _GEOMETRY_EPSILON)
    ) and (
        (right_1 > _GEOMETRY_EPSILON and right_2 < -_GEOMETRY_EPSILON)
        or (right_1 < -_GEOMETRY_EPSILON and right_2 > _GEOMETRY_EPSILON)
    )


def _cross(start: Point2D, end: Point2D, point: Point2D) -> float:
    return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])
