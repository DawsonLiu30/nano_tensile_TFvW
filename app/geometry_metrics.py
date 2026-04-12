from __future__ import annotations

import math

import numpy as np


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def convex_hull_2d(points_xy) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected Nx2 points, got shape={pts.shape}")
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=float)

    pts = np.unique(pts, axis=0)
    if len(pts) <= 2:
        return pts

    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    return hull


def polygon_area_2d(points_xy) -> float:
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected Nx2 points, got shape={pts.shape}")
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def projected_xy_geometry(atoms, mask=None) -> dict[str, float]:
    pos = np.asarray(atoms.get_positions(), dtype=float)
    if mask is not None:
        pos = pos[np.asarray(mask, dtype=bool)]
    if len(pos) == 0:
        raise ValueError("No atoms available to estimate projected geometry.")

    xy = pos[:, :2]
    span_x = float(xy[:, 0].max() - xy[:, 0].min())
    span_y = float(xy[:, 1].max() - xy[:, 1].min())
    bbox_area = float(max(span_x * span_y, 0.0))

    hull = convex_hull_2d(xy)
    hull_area = float(max(polygon_area_2d(hull), 0.0))
    equiv_diameter_nm = 0.0
    if hull_area > 0.0:
        equiv_diameter_nm = (2.0 * math.sqrt(hull_area / math.pi)) / 10.0

    return {
        "span_x_A": span_x,
        "span_y_A": span_y,
        "area_bbox_xy_A2": bbox_area,
        "area_hull_xy_A2": hull_area,
        "equiv_diameter_hull_nm": float(equiv_diameter_nm),
        "n_hull_vertices": int(len(hull)),
    }
