"""Deterministic, geometry-only FCC [111] hexagonal nanocolumns.

The triangular-layer Wigner-Seitz (2D Voronoi) cells are analytic, not fitted
from a finite point cloud. No relaxation, energy calculation, or surface
quality claim is made. Existing circular builders are not used or changed.
"""
from __future__ import annotations

import math
import numbers

import numpy as np
from ase import Atoms

SELECTION_MODES = (
    "center_inside", "positive_area_intersection", "full_containment",
)
_NORMALS = np.column_stack((np.cos(np.arange(6) * np.pi / 3),
                            np.sin(np.arange(6) * np.pi / 3)))
_VERTEX_ANGLES = np.pi / 6 + np.arange(6) * np.pi / 3
_UNIT_VERTICES = np.column_stack((np.cos(_VERTEX_ANGLES), np.sin(_VERTEX_ANGLES)))


def _finite(value: float, name: str, *, positive: bool = True) -> float:
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}.")
    return value


def _polygon_area(vertices: np.ndarray) -> float:
    if len(vertices) < 3:
        return 0.0
    # Translate first to reduce cancellation for small intersections far away.
    p = np.asarray(vertices, dtype=float) - vertices[0]
    return abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1))
                     - np.dot(p[:, 1], np.roll(p[:, 0], -1)))) / 2


def _clip_hex(vertices: np.ndarray, apothem: float) -> np.ndarray:
    """Clip a convex polygon by the six closed target half-planes."""
    polygon = np.asarray(vertices, dtype=float)
    for normal in _NORMALS:
        if not len(polygon):
            break
        clipped = []
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            da, db = float(a @ normal - apothem), float(b @ normal - apothem)
            inside_a, inside_b = da <= 0, db <= 0
            if inside_a:
                clipped.append(a)
            if inside_a != inside_b:
                clipped.append(a + (b - a) * da / (da - db))
        polygon = np.asarray(clipped, dtype=float).reshape(-1, 2)
    return polygon


def _site_id(i: int, j: int, k: int) -> int:
    """Injective signed-lattice-key encoding, independent of selection/radius."""
    def zigzag(n):
        return 2 * n if n >= 0 else -2 * n - 1

    def pair(a, b):
        return (a + b) * (a + b + 1) // 2 + b

    result = pair(pair(zigzag(i), zigzag(j)), k)
    if result > np.iinfo(np.int64).max:
        raise ValueError("Lattice key exceeds the int64 generated-site-ID range.")
    return result


def build_hex_111(
    *,
    a0: float,
    radius_A: float,
    selection_mode: str = "center_inside",
    abc_repeats: int = 2,
    vacuum: float = 5.0,
    length_tolerance_A: float = 1e-9,
    area_tolerance_A2: float = 1e-10,
) -> Atoms:
    """Build an unrelaxed Al column with fixed [111] ABC phase and xy origin.

    ``radius_A`` is the target hexagon's circumradius, not an equal-area
    circular radius. ``abc_repeats`` complete ABC periods give
    Lz = 3 * abc_repeats * a0/sqrt(3). The target, lattice, and cell center
    remain fixed; atoms are never recentered by their selected centroid.

    Positions relative to the target center are
    (-d/2, 0) + i*(d, 0) + j*(d/2, sqrt(3)*d/2)
    + (k % 3)*(0, a0/sqrt(6)), z = k*a0/sqrt(3), d=a0/sqrt(2).
    This basis fixes the WS/target vertices at 30 + 60*m degrees.

    ``positive_area_intersection`` requires clipped area > area_tolerance_A2;
    point/edge tangency is excluded. Other rules use closed half-planes with
    length_tolerance_A. IDs are hex111_v1 IDs, NOT historical parent_site_id.
    """
    a0 = _finite(a0, "a0")
    radius = _finite(radius_A, "radius_A")
    vacuum = _finite(vacuum, "vacuum")
    length_tol = _finite(length_tolerance_A, "length_tolerance_A", positive=False)
    area_tol = _finite(area_tolerance_A2, "area_tolerance_A2", positive=False)
    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"selection_mode must be one of {SELECTION_MODES}.")
    if isinstance(abc_repeats, bool) or not isinstance(abc_repeats, numbers.Integral) or abc_repeats < 1:
        raise ValueError("abc_repeats must be a positive integer.")
    abc_repeats = int(abc_repeats)
    d, h = a0 / np.sqrt(2), a0 / np.sqrt(3)
    u, v = np.array([d, 0.0]), np.array([d / 2, np.sqrt(3) * d / 2])
    origin, shift = np.array([-d / 2, 0.0]), np.array([0.0, a0 / np.sqrt(6)])
    ws_radius, apothem = d / np.sqrt(3), radius * np.sqrt(3) / 2
    ws_vertices = _UNIT_VERTICES * ws_radius
    # Conservative index bound from the inverse triangular basis; includes
    # every cell that can reach the target and the full tolerance envelope.
    bound = radius + ws_radius + 2 * length_tol + np.linalg.norm(origin) + 2 * np.linalg.norm(shift)
    extent = int(np.ceil(2 * bound / d)) + 2
    phases = []
    for phase in range(3):
        selected = []
        for i in range(-extent, extent + 1):
            for j in range(-extent, extent + 1):
                xy = origin + i * u + j * v + phase * shift
                if np.linalg.norm(xy) > radius + ws_radius + 2 * length_tol:
                    continue
                polygon = xy + ws_vertices
                overlap = _polygon_area(_clip_hex(polygon, apothem))
                if selection_mode == "center_inside":
                    keep = np.all(xy @ _NORMALS.T <= apothem + length_tol)
                elif selection_mode == "full_containment":
                    keep = np.all(polygon @ _NORMALS.T <= apothem + length_tol)
                else:
                    keep = overlap > area_tol
                if keep:
                    selected.append((i, j, xy, overlap))
        phases.append(selected)
    keys, positions, overlap_areas = [], [], []
    for k in range(3 * abc_repeats):
        for i, j, xy, overlap in phases[k % 3]:
            keys.append((i, j, k))
            positions.append([xy[0] + radius + vacuum, xy[1] + radius + vacuum, k * h])
            overlap_areas.append(overlap)
    if not keys:
        raise ValueError("Selection contains no atoms; increase radius_A or choose another rule.")
    positions = np.asarray(positions)
    side, lz = 2 * (radius + vacuum), 3 * abc_repeats * h
    if np.any(positions[:, :2] <= 0) or np.any(positions[:, :2] >= side):
        raise ValueError("vacuum is too small for the selected atomic centers; increase it.")
    atoms = Atoms("Al" * len(keys), positions=positions, cell=[side, side, lz], pbc=True)
    keys = np.asarray(keys, dtype=np.int64)
    for index, name in enumerate(("lattice_i", "lattice_j", "layer_k")):
        atoms.new_array(name, keys[:, index])
    atoms.new_array("generated_site_id", np.array([_site_id(*map(int, key)) for key in keys], dtype=np.int64))
    atoms.new_array("hex_overlap_area_A2", np.asarray(overlap_areas))
    atoms.info.update(
        geometry_only=True, generator="hex_111", generator_version=1,
        selection_mode=selection_mode, site_id_namespace="hex111_v1",
        site_id_definition="Cantor(Cantor(zigzag(i),zigzag(j)),k); not legacy parent IDs",
        lattice_constant_A=a0, orientation="111", abc_repeats=abc_repeats,
        layer_spacing_A=float(h), axial_periodic=True,
        cross_section_shape="hex_111", cross_section_radius_A=radius,
        cross_section_area_A2=float(3 * np.sqrt(3) * radius**2 / 2),
        hex_vertex_angle_deg=30.0, target_center_xy_A=[radius + vacuum] * 2,
        lattice_origin_xy_A=origin.tolist(), lattice_u_A=u.tolist(),
        lattice_v_A=v.tolist(), abc_phase_shift_xy_A=shift.tolist(),
        ws_method="analytic_infinite_triangular_lattice_2d",
        vacuum_parameter_A=vacuum, xy_box_rule="Lx=Ly=2*(radius_A+vacuum)",
        length_tolerance_A=length_tol, area_tolerance_A2=area_tol,
        minimum_transverse_image_gap_A=(side - np.ptp(positions[:, :2], axis=0)).tolist(),
    )
    return atoms
