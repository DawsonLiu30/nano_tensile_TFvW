import numpy as np
from ase.build import bulk
from ase.io import write

DEFAULT_CONFIG = {
    "a0": 4.05,
    "size": 3.5,          
    "length_z": 40.0,     
    "vacuum": 5.0,        
    "gamma100": 1.00,
    "gamma110": 1.06,
}


def normalize(v):
    v = np.array(v, float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero vector in normalize()")
    return v / n


def rotate_to_make_a3_along_z(atoms):
    """
    Rotate atoms so that cell[2] aligns with Cartesian z.
    Also make ex in plane perpendicular to ez.
    """
    cell = atoms.get_cell().array
    a1, a2, a3 = cell[0], cell[1], cell[2]

    ez = normalize(a3)
    a1_perp = a1 - np.dot(a1, ez) * ez

    if np.linalg.norm(a1_perp) < 1e-8:
        a2_perp = a2 - np.dot(a2, ez) * ez
        ex = normalize(a2_perp)
    else:
        ex = normalize(a1_perp)

    ey = np.cross(ez, ex)
    new_cell_basis = np.array([ex, ey, ez])

    atoms.set_cell(np.dot(cell, new_cell_basis.T), scale_atoms=True)
    atoms.rotate(a3, "z", rotate_cell=True)
    return atoms


def _set_cell_vacuum_xy_only(atoms, vacuum_xy):
    pos = atoms.get_positions()
    xmin, ymin, zmin = pos.min(axis=0)
    xmax, ymax, zmax = pos.max(axis=0)

    Lx = (xmax - xmin) + 2.0 * vacuum_xy
    Ly = (ymax - ymin) + 2.0 * vacuum_xy
    Lz = (zmax - zmin)  

    # 設成正交盒（方便看、也符合 nanowire 設定）
    atoms.set_cell([Lx, Ly, Lz], scale_atoms=False)

    # 只在 x/y 置中；z 不動
    pos = atoms.get_positions()
    pos[:, 0] += (Lx / 2.0) - pos[:, 0].mean()
    pos[:, 1] += (Ly / 2.0) - pos[:, 1].mean()

    pos[:, 2] -= pos[:, 2].min()

    atoms.set_positions(pos)
    return atoms


def build_wulff_nanocrystal(
    a0=DEFAULT_CONFIG["a0"],
    size=DEFAULT_CONFIG["size"],
    length_z=DEFAULT_CONFIG["length_z"],
    vacuum=DEFAULT_CONFIG["vacuum"],
    gamma100=DEFAULT_CONFIG["gamma100"],
    gamma110=DEFAULT_CONFIG["gamma110"],
):

    base = bulk("Al", "fcc", a=a0, cubic=True)
    base = rotate_to_make_a3_along_z(base)

    Lx0, Ly0, Lz0, *_ = base.cell.cellpar()

    min_xy = 2 * size + 6.0
    nx = int(np.ceil(min_xy / Lx0))
    ny = int(np.ceil(min_xy / Ly0))
    nz = int(np.ceil(length_z / Lz0))

    blk = base.repeat((nx, ny, nz))

    # 以幾何中心做裁切
    cell = blk.get_cell().array
    origin = 0.5 * cell.sum(axis=0)
    r = blk.get_positions() - origin
    xy = r[:, :2]

    # Wulff prism side walls: {100}+{110}
    n100 = np.array([[1.0, 0.0], [0.0, 1.0]])
    n110 = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)

    d100 = size * gamma100
    d110 = size * gamma110

    keep = (
        (np.abs(xy @ n100[0]) <= d100) &
        (np.abs(xy @ n100[1]) <= d100) &
        (np.abs(xy @ n110[0]) <= d110) &
        (np.abs(xy @ n110[1]) <= d110)
    )

    atoms = blk[keep]

    atoms = _set_cell_vacuum_xy_only(atoms, vacuum_xy=vacuum)
 
    atoms.pbc = (False, False, True)

    return atoms


if __name__ == "__main__":
    print(f"Building with default config: {DEFAULT_CONFIG}")
    atoms = build_wulff_nanocrystal()

    print(f"Atoms: {len(atoms)}")
    print(f"Cell (Å): {atoms.get_cell().lengths()}")
    print(f"PBC: {atoms.get_pbc()}")

    write("test_structure.xyz", atoms)
    write("test_structure.vasp", atoms, vasp5=True, direct=True)
    print("Written: test_structure.xyz, test_structure.vasp")
    print("Done.")
