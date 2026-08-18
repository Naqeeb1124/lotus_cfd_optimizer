"""Export optimized 2D baffle topology for CAD reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def extract_contours(binary: np.ndarray) -> list[np.ndarray]:
    """Extract contours using scikit-image when available, with a cell-edge fallback."""
    binary = np.asarray(binary, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("binary topology must be a 2D array.")
    try:
        from skimage.measure import find_contours
    except ImportError:
        return _extract_cell_edge_loops(binary)
    return [np.asarray(contour, dtype=float) for contour in find_contours(binary.astype(float), 0.5)]


def _extract_cell_edge_loops(binary: np.ndarray) -> list[np.ndarray]:
    """Return boundary loops from exposed cell edges when scikit-image is absent."""
    ny, nx = binary.shape
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for row, col in zip(*np.nonzero(binary)):
        neighbors = (
            (row - 1, col, ((row, col), (row, col + 1))),
            (row, col + 1, ((row, col + 1), (row + 1, col + 1))),
            (row + 1, col, ((row + 1, col + 1), (row + 1, col))),
            (row, col - 1, ((row + 1, col), (row, col))),
        )
        for nr, nc, edge in neighbors:
            if nr < 0 or nr >= ny or nc < 0 or nc >= nx or not binary[nr, nc]:
                a, b = edge
                edges.add((a, b) if a <= b else (b, a))

    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b in edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    loops: list[np.ndarray] = []
    while edges:
        first_edge = next(iter(edges))
        edges.remove(first_edge)
        start, current = first_edge
        points = [start, current]
        while current != start:
            candidates = [candidate for candidate in adjacency[current] if
                          ((min(current, candidate), max(current, candidate)) in edges)]
            if not candidates:
                break
            nxt = candidates[0]
            edge = (min(current, nxt), max(current, nxt))
            edges.remove(edge)
            current = nxt
            points.append(current)
        loops.append(np.asarray(points, dtype=float))
    return loops


def save_topology_npz(
    path: str | Path,
    density: np.ndarray,
    binary: np.ndarray,
    *,
    width: float,
    height: float,
    velocity_u: np.ndarray | None = None,
    velocity_v: np.ndarray | None = None,
    pressure: np.ndarray | None = None,
    tray_averages: np.ndarray | None = None,
    objective: float | None = None,
    fan_flow: float | None = None,
    fan_pressure: float | None = None,
    target_velocity: float | None = None,
    left_opening_flow: float | None = None,
    right_opening_flow: float | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, object] = {
        "density": np.asarray(density, dtype=float),
        "binary": np.asarray(binary, dtype=bool),
        "width": float(width),
        "height": float(height),
    }
    if velocity_u is not None:
        arrays["velocity_u"] = np.asarray(velocity_u, dtype=float)
    if velocity_v is not None:
        arrays["velocity_v"] = np.asarray(velocity_v, dtype=float)
    if pressure is not None:
        arrays["pressure"] = np.asarray(pressure, dtype=float)
    if tray_averages is not None:
        arrays["tray_averages"] = np.asarray(tray_averages, dtype=float)
    if objective is not None:
        arrays["objective"] = float(objective)
    if fan_flow is not None:
        arrays["fan_flow"] = float(fan_flow)
    if fan_pressure is not None:
        arrays["fan_pressure"] = float(fan_pressure)
    if target_velocity is not None:
        arrays["target_velocity"] = float(target_velocity)
    if left_opening_flow is not None:
        arrays["left_opening_flow"] = float(left_opening_flow)
    if right_opening_flow is not None:
        arrays["right_opening_flow"] = float(right_opening_flow)
    np.savez_compressed(path, **arrays)
    return path


def save_topology_svg(
    path: str | Path,
    binary: np.ndarray,
    *,
    width: float,
    height: float,
    contours: Iterable[np.ndarray] | None = None,
) -> Path:
    """Write a simple millimetre-scaled SVG of the solid topology."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = np.asarray(binary, dtype=bool)
    ny, nx = binary.shape
    contours = list(extract_contours(binary) if contours is None else contours)
    paths: list[str] = []
    for contour in contours:
        if contour.size == 0:
            continue
        # find_contours returns (row, column), with row zero at the top.
        points = [
            (point[1] * width / nx, (ny - point[0]) * height / ny)
            for point in contour
        ]
        commands = " ".join(
            ("M" if index == 0 else "L") + f" {x:.6f},{y:.6f}"
            for index, (x, y) in enumerate(points)
        )
        paths.append(f'<path d="{commands} Z" fill="#263238" stroke="none"/>')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.6f} {height:.6f}" '
        f'width="{width * 1000:.2f}mm" height="{height * 1000:.2f}mm">\n'
        + "\n".join(paths)
        + "\n</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")
    return path
