from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import write
from ase.lattice.cubic import FaceCenteredCubic

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.aluminum_defaults import AL_FCC_A0_TFVW_ANG


DEFAULT_CONFIG = {
    "builder": "periodic_prism",
    "a0": AL_FCC_A0_TFVW_ANG,
    "diameter_nm": 2.0,
    "radius_A": None,
    "length_z": 20.0,
    "vacuum": 10.0,
    "orientation": "111",
    "cross_section_shape": "circle",
    "shape_rotation_deg": 0.0,
    "size": 3.5,
    "gamma100": 1.00,
    "gamma110": 1.06,
}


ORIENTATION_DIRECTIONS = {
    "111": [[1, -1, 0], [1, 1, -2], [1, 1, 1]],
    "100": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "110": [[-1, 1, 0], [0, 0, 1], [1, 1, 0]],
}


CROSS_SECTION_SIDES = {
    "hexagon": 6,
    "triangle": 3,
}


def _normalize(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return arr / norm


def _resolve_radius_A(diameter_nm: float | None, radius_A: float | None) -> float:
    if radius_A is None:
        if diameter_nm is None:
            diameter_nm = float(DEFAULT_CONFIG["diameter_nm"])
        radius_A = 0.5 * float(diameter_nm) * 10.0
    elif diameter_nm is not None:
        expected = 0.5 * float(diameter_nm) * 10.0
        if not math.isclose(float(radius_A), expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"radius_A ({float(radius_A):.6f}) and diameter_nm ({float(diameter_nm):.6f}) "
                f"are inconsistent; expected radius_A={expected:.6f}."
            )

    radius_A = float(radius_A)
    if radius_A <= 0.0:
        raise ValueError(f"radius_A must be positive, got {radius_A}")
    return radius_A


def _set_xy_vacuum_and_center(atoms, vacuum: float) -> None:
    pos = atoms.get_positions()
    xmin, ymin = np.min(pos[:, :2], axis=0)
    xmax, ymax = np.max(pos[:, :2], axis=0)

    span_x = float(xmax - xmin)
    span_y = float(ymax - ymin)
    cell = atoms.get_cell().copy()
    cell[0, :] = [span_x + 2.0 * float(vacuum), 0.0, 0.0]
    cell[1, :] = [0.0, span_y + 2.0 * float(vacuum), 0.0]
    atoms.set_cell(cell, scale_atoms=False)

    pos = atoms.get_positions()
    pos[:, 0] += (cell[0, 0] / 2.0) - float(pos[:, 0].mean())
    pos[:, 1] += (cell[1, 1] / 2.0) - float(pos[:, 1].mean())
    atoms.set_positions(pos)


def _regular_polygon_vertices(n_sides: int, radius: float, rotation_deg: float) -> np.ndarray:
    if int(n_sides) < 3:
        raise ValueError(f"n_sides must be >= 3, got {n_sides}")
    angles = np.deg2rad(float(rotation_deg)) + 2.0 * np.pi * np.arange(int(n_sides)) / int(n_sides)
    return np.column_stack([float(radius) * np.cos(angles), float(radius) * np.sin(angles)])


def _inside_convex_polygon(points_xy: np.ndarray, vertices_xy: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    points = np.asarray(points_xy, dtype=float)
    vertices = np.asarray(vertices_xy, dtype=float)
    keep = np.ones(points.shape[0], dtype=bool)
    for i in range(len(vertices)):
        a = vertices[i]
        b = vertices[(i + 1) % len(vertices)]
        edge = b - a
        rel = points - a
        cross_z = edge[0] * rel[:, 1] - edge[1] * rel[:, 0]
        keep &= cross_z >= -float(tol)
    return keep


def _cross_section_mask(
    positions: np.ndarray,
    *,
    center_xy: tuple[float, float],
    radius_A: float,
    shape: str,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    rel_xy = np.asarray(positions[:, :2], dtype=float).copy()
    rel_xy[:, 0] -= float(center_xy[0])
    rel_xy[:, 1] -= float(center_xy[1])

    shape_key = str(shape).strip().lower()
    if shape_key in {"circle", "circular", "nanocolumn"}:
        radial2 = rel_xy[:, 0] ** 2 + rel_xy[:, 1] ** 2
        return radial2 <= float(radius_A) ** 2
    if shape_key in CROSS_SECTION_SIDES:
        vertices = _regular_polygon_vertices(
            CROSS_SECTION_SIDES[shape_key],
            radius=float(radius_A),
            rotation_deg=float(rotation_deg),
        )
        return _inside_convex_polygon(rel_xy, vertices)
    raise ValueError(
        f"Unsupported cross-section shape '{shape}'. "
        f"Use circle, hexagon, or triangle."
    )


def cross_section_area_A2(shape: str, radius_A: float) -> float:
    shape_key = str(shape).strip().lower()
    radius = float(radius_A)
    if shape_key in {"circle", "circular", "nanocolumn"}:
        return float(np.pi * radius**2)
    if shape_key in CROSS_SECTION_SIDES:
        n_sides = CROSS_SECTION_SIDES[shape_key]
        return float(0.5 * n_sides * radius**2 * np.sin(2.0 * np.pi / n_sides))
    raise ValueError(f"Unsupported cross-section shape '{shape}'.")


def build_periodic_prism(
    a0: float = DEFAULT_CONFIG["a0"],
    diameter_nm: float | None = DEFAULT_CONFIG["diameter_nm"],
    radius_A: float | None = DEFAULT_CONFIG["radius_A"],
    length_z: float = DEFAULT_CONFIG["length_z"],
    vacuum: float = DEFAULT_CONFIG["vacuum"],
    orientation: str = DEFAULT_CONFIG["orientation"],
    cross_section_shape: str = DEFAULT_CONFIG["cross_section_shape"],
    shape_rotation_deg: float = DEFAULT_CONFIG["shape_rotation_deg"],
):
    """Build an axially periodic Al prism with a selected xy cross-section.

    The model is infinite along the axial direction through PBC. ``length_z``
    controls the axial repeat length of the simulation cell, not a finite
    physical column length.
    """
    radius = _resolve_radius_A(diameter_nm=diameter_nm, radius_A=radius_A)

    if orientation not in ORIENTATION_DIRECTIONS:
        raise ValueError(f"Unsupported orientation '{orientation}'.")

    base_atoms = FaceCenteredCubic(
        directions=ORIENTATION_DIRECTIONS[orientation],
        size=(1, 1, 1),
        symbol="Al",
        pbc=True,
        latticeconstant=float(a0),
    )

    lx, ly, lz = base_atoms.cell.lengths()
    nx = max(1, int(np.ceil((2.0 * radius + 5.0) / float(lx))))
    ny = max(1, int(np.ceil((2.0 * radius + 5.0) / float(ly))))
    nz = max(1, int(np.ceil(float(length_z) / float(lz))))

    supercell = base_atoms.repeat((nx, ny, nz))
    pos = supercell.get_positions()
    cell = supercell.get_cell()

    cx = 0.5 * float(cell[0, 0])
    cy = 0.5 * float(cell[1, 1])
    mask = _cross_section_mask(
        pos,
        center_xy=(cx, cy),
        radius_A=radius,
        shape=str(cross_section_shape),
        rotation_deg=float(shape_rotation_deg),
    )

    prism = supercell[mask]
    if len(prism) == 0:
        raise RuntimeError(
            "Cross-section mask removed all atoms. Increase the diameter/radius "
            "or choose a different shape rotation."
        )
    pos = prism.get_positions()
    pos[:, 0] += (radius + float(vacuum)) - float(pos[:, 0].mean())
    pos[:, 1] += (radius + float(vacuum)) - float(pos[:, 1].mean())
    prism.set_positions(pos)
    prism.set_cell(
        [
            2.0 * radius + 2.0 * float(vacuum),
            2.0 * radius + 2.0 * float(vacuum),
            float(supercell.cell[2, 2]),
        ]
    )
    prism.pbc = [True, True, True]
    prism.info["cross_section_shape"] = str(cross_section_shape).strip().lower()
    prism.info["cross_section_radius_A"] = radius
    prism.info["cross_section_area_A2"] = cross_section_area_A2(str(cross_section_shape), radius)
    prism.info["axial_periodic"] = True

    return prism


def build_circular_nanowire(
    a0: float = DEFAULT_CONFIG["a0"],
    diameter_nm: float | None = DEFAULT_CONFIG["diameter_nm"],
    radius_A: float | None = DEFAULT_CONFIG["radius_A"],
    length_z: float = DEFAULT_CONFIG["length_z"],
    vacuum: float = DEFAULT_CONFIG["vacuum"],
    orientation: str = DEFAULT_CONFIG["orientation"],
):
    return build_periodic_prism(
        a0=float(a0),
        diameter_nm=diameter_nm,
        radius_A=radius_A,
        length_z=float(length_z),
        vacuum=float(vacuum),
        orientation=str(orientation),
        cross_section_shape="circle",
        shape_rotation_deg=0.0,
    )


def rotate_to_make_a3_along_z(atoms):
    cell = atoms.get_cell().array
    a1, a2, a3 = cell[0], cell[1], cell[2]

    ez = _normalize(a3)
    a1_perp = a1 - np.dot(a1, ez) * ez
    if np.linalg.norm(a1_perp) < 1e-8:
        a2_perp = a2 - np.dot(a2, ez) * ez
        ex = _normalize(a2_perp)
    else:
        ex = _normalize(a1_perp)

    ey = np.cross(ez, ex)
    basis = np.array([ex, ey, ez])
    atoms.set_cell(np.dot(cell, basis.T), scale_atoms=True)
    atoms.rotate(a3, "z", rotate_cell=True)
    return atoms


def build_wulff_nanocrystal(
    a0: float = DEFAULT_CONFIG["a0"],
    size: float = DEFAULT_CONFIG["size"],
    length_z: float = DEFAULT_CONFIG["length_z"],
    vacuum: float = DEFAULT_CONFIG["vacuum"],
    gamma100: float = DEFAULT_CONFIG["gamma100"],
    gamma110: float = DEFAULT_CONFIG["gamma110"],
):
    base = bulk("Al", "fcc", a=float(a0), cubic=True)
    base = rotate_to_make_a3_along_z(base)

    Lx0, Ly0, Lz0, *_ = base.cell.cellpar()
    min_xy = 2.0 * float(size) + 6.0

    nx = max(1, int(np.ceil(min_xy / float(Lx0))))
    ny = max(1, int(np.ceil(min_xy / float(Ly0))))
    nz = max(1, int(np.ceil(float(length_z) / float(Lz0))))

    blk = base.repeat((nx, ny, nz))
    origin = 0.5 * blk.get_cell().array.sum(axis=0)
    rel = blk.get_positions() - origin
    xy = rel[:, :2]

    n100 = np.array([[1.0, 0.0], [0.0, 1.0]])
    n110 = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)

    d100 = float(size) * float(gamma100)
    d110 = float(size) * float(gamma110)

    keep = (
        (np.abs(xy @ n100[0]) <= d100)
        & (np.abs(xy @ n100[1]) <= d100)
        & (np.abs(xy @ n110[0]) <= d110)
        & (np.abs(xy @ n110[1]) <= d110)
    )

    atoms = blk[keep]
    _set_xy_vacuum_and_center(atoms, vacuum=float(vacuum))
    atoms.pbc = [True, True, True]
    return atoms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--builder",
        default=DEFAULT_CONFIG["builder"],
        choices=["periodic_prism", "paper_circular", "wulff_prism"],
        help="Geometry generator to use.",
    )
    parser.add_argument("--a0", type=float, default=DEFAULT_CONFIG["a0"])
    parser.add_argument("--diameter", type=float, default=DEFAULT_CONFIG["diameter_nm"])
    parser.add_argument("--radius-A", dest="radius_A", type=float, default=DEFAULT_CONFIG["radius_A"])
    parser.add_argument("--orientation", type=str, default=DEFAULT_CONFIG["orientation"], choices=list(ORIENTATION_DIRECTIONS))
    parser.add_argument(
        "--cross-section-shape",
        default=DEFAULT_CONFIG["cross_section_shape"],
        choices=["circle", "hexagon", "triangle"],
        help="xy cross-section shape for an axially periodic prism.",
    )
    parser.add_argument(
        "--shape-rotation-deg",
        type=float,
        default=DEFAULT_CONFIG["shape_rotation_deg"],
        help="Rotation of the polygonal cross-section in the xy plane.",
    )
    parser.add_argument(
        "--length-z",
        type=float,
        default=DEFAULT_CONFIG["length_z"],
        help="Target builder length along z in Angstrom. This is not in nm.",
    )
    parser.add_argument("--vacuum", type=float, default=DEFAULT_CONFIG["vacuum"])
    parser.add_argument("--size", type=float, default=DEFAULT_CONFIG["size"])
    parser.add_argument("--gamma100", type=float, default=DEFAULT_CONFIG["gamma100"])
    parser.add_argument("--gamma110", type=float, default=DEFAULT_CONFIG["gamma110"])
    parser.add_argument("--output", default="", help="Output VASP filename.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.builder == "periodic_prism":
        atoms = build_periodic_prism(
            a0=float(args.a0),
            diameter_nm=float(args.diameter) if args.radius_A is None else None,
            radius_A=None if args.radius_A is None else float(args.radius_A),
            length_z=float(args.length_z),
            vacuum=float(args.vacuum),
            orientation=str(args.orientation),
            cross_section_shape=str(args.cross_section_shape),
            shape_rotation_deg=float(args.shape_rotation_deg),
        )
        shape = str(args.cross_section_shape)
        output = args.output or f"init_{args.orientation}_Al_{shape}_{0.5 * atoms.cell[0,0] - args.vacuum:.2f}A_radius.vasp"
        print(
            f"[builder] periodic_prism orientation=[{args.orientation}] "
            f"shape={shape} radius_A={atoms.info.get('cross_section_radius_A', float('nan')):.4f} "
            f"area_A2={atoms.info.get('cross_section_area_A2', float('nan')):.4f}"
        )
    elif args.builder == "paper_circular":
        atoms = build_circular_nanowire(
            a0=float(args.a0),
            diameter_nm=float(args.diameter) if args.radius_A is None else None,
            radius_A=None if args.radius_A is None else float(args.radius_A),
            length_z=float(args.length_z),
            vacuum=float(args.vacuum),
            orientation=str(args.orientation),
        )
        output = args.output or f"init_{args.orientation}_Al_{0.5 * atoms.cell[0,0] - args.vacuum:.2f}A_radius.vasp"
        print(
            f"[builder] paper_circular orientation=[{args.orientation}] "
            f"radius_A={0.5 * (atoms.cell[0,0] - 2.0 * args.vacuum):.4f}"
        )
    else:
        atoms = build_wulff_nanocrystal(
            a0=float(args.a0),
            size=float(args.size),
            length_z=float(args.length_z),
            vacuum=float(args.vacuum),
            gamma100=float(args.gamma100),
            gamma110=float(args.gamma110),
        )
        output = args.output or "test_structure.vasp"
        print(
            f"[builder] wulff_prism size={float(args.size):.4f} "
            f"gamma100={float(args.gamma100):.4f} gamma110={float(args.gamma110):.4f}"
        )

    print(f"Atoms: {len(atoms)}")
    print(f"Cell (A): {atoms.get_cell().lengths()}")
    print(f"PBC: {atoms.get_pbc()}")

    write(output, atoms, vasp5=True, direct=True)
    print(f"Written: {output}")
