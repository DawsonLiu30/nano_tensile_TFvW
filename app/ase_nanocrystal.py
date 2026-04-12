from __future__ import annotations

import argparse
import itertools

import numpy as np
from ase.io import write
from ase.lattice.cubic import FaceCenteredCubic


DEFAULT_CONFIG = {
    "a0": 4.05,
    "diameter_nm": 2.0,
    "length_z": 20.0,
    "vacuum": 5.0,
    "axial_vacuum": None,
    "orientation": "111",
}


def _orientation_directions(orientation: str) -> list[list[int]]:
    if orientation == "111":
        return [[1, -1, 0], [1, 1, -2], [1, 1, 1]]
    if orientation == "100":
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    if orientation == "110":
        return [[-1, 1, 0], [0, 0, 1], [1, 1, 0]]
    raise ValueError(f"Unsupported orientation: {orientation}. Use 111, 100, or 110.")


def _normalized(v) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError(f"Cannot normalize zero vector: {v}")
    return arr / norm


def _orientation_basis(orientation: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dirs = _orientation_directions(orientation)
    ex = _normalized(dirs[0])
    ey = _normalized(dirs[1])
    ez = _normalized(dirs[2])
    return ex, ey, ez


def _canonical_2d_unit(v2: np.ndarray, tol: float = 1e-10) -> tuple[float, float] | None:
    vec = np.asarray(v2, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= tol:
        return None
    vec = vec / norm
    if abs(vec[0]) > tol:
        sign = 1.0 if vec[0] > 0.0 else -1.0
    elif abs(vec[1]) > tol:
        sign = 1.0 if vec[1] > 0.0 else -1.0
    else:
        return None
    vec *= sign
    return (round(float(vec[0]), 10), round(float(vec[1]), 10))


def _family_equivalents(family: str) -> list[np.ndarray]:
    family = family.strip("{}")
    if family == "100":
        base = (1, 0, 0)
    elif family == "110":
        base = (1, 1, 0)
    elif family == "111":
        base = (1, 1, 1)
    else:
        raise ValueError(f"Unsupported family: {family}")

    out: set[tuple[int, int, int]] = set()
    for perm in set(itertools.permutations(base)):
        nz = [i for i, x in enumerate(perm) if x != 0]
        for signs in itertools.product([-1, 1], repeat=len(nz)):
            vec = list(perm)
            for idx, sgn in zip(nz, signs):
                vec[idx] *= sgn
            out.add(tuple(vec))
    return [np.asarray(v, dtype=float) for v in sorted(out)]


def _project_miller_to_lab(miller: np.ndarray, orientation: str) -> np.ndarray:
    ex, ey, ez = _orientation_basis(orientation)
    n = np.asarray(miller, dtype=float)
    lab = np.array([np.dot(n, ex), np.dot(n, ey), np.dot(n, ez)], dtype=float)
    return _normalized(lab)


def _projected_wulff_halfspaces(
    orientation: str,
    size: float,
    surface_energies: dict[str, float],
) -> list[tuple[np.ndarray, float, str]]:
    gamma_ref = min(float(v) for v in surface_energies.values())
    halfspaces: list[tuple[np.ndarray, float, str]] = []
    seen: set[tuple[str, tuple[float, float]]] = set()

    for family, gamma in surface_energies.items():
        gamma = float(gamma)
        if gamma <= 0.0:
            raise ValueError(f"Surface energy must be positive for {family}, got {gamma}")
        distance = float(size) * gamma_ref / gamma
        for miller in _family_equivalents(family):
            n_lab = _project_miller_to_lab(miller, orientation)
            key = _canonical_2d_unit(n_lab[:2])
            if key is None:
                continue
            seen_key = (family, key)
            if seen_key in seen:
                continue
            seen.add(seen_key)
            halfspaces.append((np.asarray(key, dtype=float), distance, family))

    if not halfspaces:
        raise RuntimeError("No projected facet halfspaces were generated.")
    return halfspaces


def _repeat_counts_for_shape(
    base_atoms,
    radial_extent: float,
    length_z: float,
    margin: float,
) -> tuple[int, int, int]:
    lx, ly, lz = base_atoms.cell.lengths()
    cover_xy = 2.0 * (float(radial_extent) + float(margin))
    cover_z = float(length_z) + 2.0 * float(margin)
    nx = max(1, int(np.ceil(cover_xy / lx)))
    ny = max(1, int(np.ceil(cover_xy / ly)))
    nz = max(1, int(np.ceil(cover_z / lz)))
    return nx, ny, nz


def _set_centered_finite_cell(atoms, cell_x: float, cell_y: float, vacuum_z: float):
    pos = atoms.get_positions().copy()
    span = np.ptp(pos, axis=0)
    if span[2] <= 0.0:
        raise ValueError("Structure has zero z-span; cannot build a finite-length wire.")

    cell_z = float(span[2] + 2.0 * float(vacuum_z))

    shift = np.array(
        [
            0.5 * float(cell_x) - 0.5 * span[0] - float(pos[:, 0].min()),
            0.5 * float(cell_y) - 0.5 * span[1] - float(pos[:, 1].min()),
            float(vacuum_z) - float(pos[:, 2].min()),
        ],
        dtype=float,
    )
    pos += shift

    atoms.set_positions(pos)
    atoms.set_cell([float(cell_x), float(cell_y), cell_z])
    atoms.pbc = [False, False, False]
    atoms.info["axial_vacuum_A"] = float(vacuum_z)
    atoms.info["finite_cell"] = True
    return atoms


def build_circular_nanowire(
    a0=DEFAULT_CONFIG["a0"],
    diameter_nm=DEFAULT_CONFIG["diameter_nm"],
    length_z=DEFAULT_CONFIG["length_z"],
    vacuum=DEFAULT_CONFIG["vacuum"],
    orientation=DEFAULT_CONFIG["orientation"],
    axial_vacuum=DEFAULT_CONFIG["axial_vacuum"],
):
    radius = (float(diameter_nm) * 10.0) / 2.0
    vacuum = float(vacuum)
    vacuum_z = float(vacuum if axial_vacuum is None else axial_vacuum)

    base_atoms = FaceCenteredCubic(
        directions=_orientation_directions(str(orientation)),
        size=(1, 1, 1),
        symbol="Al",
        pbc=True,
        latticeconstant=float(a0),
    )

    lx, ly, lz = base_atoms.cell.lengths()
    nx = int(np.ceil((2.0 * radius + 5.0) / lx))
    ny = int(np.ceil((2.0 * radius + 5.0) / ly))
    nz = int(np.ceil(float(length_z) / lz))
    nx, ny, nz = max(1, nx), max(1, ny), max(1, nz)

    supercell = base_atoms.repeat((nx, ny, nz))
    cx = supercell.cell[0, 0] / 2.0
    cy = supercell.cell[1, 1] / 2.0
    pos = supercell.get_positions()

    dist2 = (pos[:, 0] - cx) ** 2 + (pos[:, 1] - cy) ** 2
    mask = dist2 <= radius**2
    nanowire = supercell[mask]

    cell_x = 2.0 * radius + 2.0 * vacuum
    cell_y = 2.0 * radius + 2.0 * vacuum
    nanowire = _set_centered_finite_cell(nanowire, cell_x=cell_x, cell_y=cell_y, vacuum_z=vacuum_z)
    nanowire.info["builder"] = "circular_nanowire"
    nanowire.info["radius_A"] = float(radius)
    nanowire.info["orientation"] = str(orientation)
    return nanowire


def build_wulff_nanocrystal(
    a0=4.05,
    size=5.0,
    length_z=40.0,
    vacuum=5.0,
    gamma100=1.0,
    gamma110=1.06,
    orientation="111",
    axial_vacuum=None,
):
    """
    Build a finite-length faceted nanocrystal for grip-based tensile tests.

    `size` is the characteristic in-plane facet distance in Angstrom.
    Side facets are generated from projected Wulff halfspaces of the provided
    surface-energy families, while the tensile axis is clipped to `length_z`.
    """

    orientation = str(orientation)
    base_atoms = FaceCenteredCubic(
        directions=_orientation_directions(orientation),
        size=(1, 1, 1),
        symbol="Al",
        pbc=True,
        latticeconstant=float(a0),
    )

    margin = 2.0 * float(a0)
    radial_extent = float(size)
    nx, ny, nz = _repeat_counts_for_shape(
        base_atoms,
        radial_extent=radial_extent,
        length_z=float(length_z),
        margin=margin,
    )
    supercell = base_atoms.repeat((nx, ny, nz))

    pos = supercell.get_positions().copy()
    cell_lengths = supercell.cell.lengths()
    center = np.array(
        [0.5 * cell_lengths[0], 0.5 * cell_lengths[1], 0.5 * cell_lengths[2]],
        dtype=float,
    )
    rel = pos - center

    side_halfspaces = _projected_wulff_halfspaces(
        orientation=orientation,
        size=float(size),
        surface_energies={"100": float(gamma100), "110": float(gamma110)},
    )

    mask = np.ones(len(supercell), dtype=bool)
    xy = rel[:, :2]
    for normal_xy, distance, _family in side_halfspaces:
        mask &= np.abs(xy @ normal_xy) <= (float(distance) + 1e-8)

    half_length = 0.5 * float(length_z)
    mask &= np.abs(rel[:, 2]) <= (half_length + 1e-8)

    atoms = supercell[mask]
    if len(atoms) == 0:
        raise RuntimeError(
            "Faceted builder produced an empty structure. "
            "Try increasing --size or --length-z."
        )

    span = np.ptp(atoms.get_positions(), axis=0)
    cell_x = float(span[0] + 2.0 * float(vacuum))
    cell_y = float(span[1] + 2.0 * float(vacuum))
    vacuum_z = float(vacuum if axial_vacuum is None else axial_vacuum)
    atoms = _set_centered_finite_cell(atoms, cell_x=cell_x, cell_y=cell_y, vacuum_z=vacuum_z)

    atoms.info["builder"] = "wulff_nanocrystal"
    atoms.info["size_A"] = float(size)
    atoms.info["gamma100"] = float(gamma100)
    atoms.info["gamma110"] = float(gamma110)
    atoms.info["orientation"] = orientation
    atoms.info["facet_families"] = "100,110"
    atoms.info["n_side_halfspaces"] = int(len(side_halfspaces))
    return atoms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diameter", type=float, default=DEFAULT_CONFIG["diameter_nm"])
    parser.add_argument(
        "--orientation",
        type=str,
        default=DEFAULT_CONFIG["orientation"],
        choices=["111", "100", "110"],
    )
    parser.add_argument("--length-z", type=float, default=DEFAULT_CONFIG["length_z"])
    parser.add_argument("--vacuum", type=float, default=DEFAULT_CONFIG["vacuum"])
    parser.add_argument(
        "--axial-vacuum",
        type=float,
        default=None,
        help="Optional vacuum to add above/below the finite wire along z.",
    )
    args = parser.parse_args()

    print(f"Build [{args.orientation}] Al nanowire | diameter={args.diameter:.3f} nm")
    atoms = build_circular_nanowire(
        diameter_nm=float(args.diameter),
        orientation=str(args.orientation),
        length_z=float(args.length_z),
        vacuum=float(args.vacuum),
        axial_vacuum=args.axial_vacuum,
    )

    print(f"Atoms: {len(atoms)}")
    print(f"Cell (A): {atoms.cell.lengths()}")
    print(f"PBC: {atoms.pbc}")
    vasp_filename = f"init_{args.orientation}_Al_{args.diameter}nm.vasp"
    write(vasp_filename, atoms, vasp5=True, direct=True)
    print(f"Wrote {vasp_filename}")
