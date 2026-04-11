import numpy as np
from ase.io import write
from ase.lattice.cubic import FaceCenteredCubic
import argparse

DEFAULT_CONFIG = {
    "a0": 4.05,
    "diameter_nm": 2.0,
    "length_z": 20.0,
    "vacuum": 5.0,
    "orientation": "111",
}

def build_circular_nanowire(
    a0=DEFAULT_CONFIG["a0"],
    diameter_nm=DEFAULT_CONFIG["diameter_nm"],
    length_z=DEFAULT_CONFIG["length_z"],
    vacuum=DEFAULT_CONFIG["vacuum"],
    orientation=DEFAULT_CONFIG["orientation"]
):
    radius = (diameter_nm * 10.0) / 2.0
    
    if orientation == "111":
        directions = [[1, -1, 0], [1, 1, -2], [1, 1, 1]]
    elif orientation == "100":
        directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    elif orientation == "110":
        directions = [[-1, 1, 0], [0, 0, 1], [1, 1, 0]]
    else:
        raise ValueError("不支援的晶向！請選擇 111, 100, 或 110")
    
    base_atoms = FaceCenteredCubic(
        directions=directions,
        size=(1, 1, 1), 
        symbol='Al', 
        pbc=True, 
        latticeconstant=a0
    )
    
    lx, ly, lz = base_atoms.cell.lengths()
    nx = int(np.ceil((2.0 * radius + 5.0) / lx))
    ny = int(np.ceil((2.0 * radius + 5.0) / ly))
    nz = int(np.ceil(length_z / lz))
    nx, ny, nz = max(1, nx), max(1, ny), max(1, nz)
    
    supercell = base_atoms.repeat((nx, ny, nz))
    cx = supercell.cell[0, 0] / 2.0
    cy = supercell.cell[1, 1] / 2.0
    pos = supercell.get_positions()
    
    dist2 = (pos[:, 0] - cx)**2 + (pos[:, 1] - cy)**2
    mask = dist2 <= radius**2
    nanowire = supercell[mask]
    
    new_lx = 2.0 * radius + 2.0 * vacuum
    new_ly = 2.0 * radius + 2.0 * vacuum
    new_lz = supercell.cell[2, 2]
    
    wire_pos = nanowire.get_positions()
    cur_cx = wire_pos[:, 0].mean()
    cur_cy = wire_pos[:, 1].mean()
    
    wire_pos[:, 0] += (new_lx / 2.0) - cur_cx
    wire_pos[:, 1] += (new_ly / 2.0) - cur_cy
    
    nanowire.set_positions(wire_pos)
    nanowire.set_cell([new_lx, new_ly, new_lz])
    nanowire.pbc = [True, True, True] 
    
    return nanowire

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diameter", type=float, default=DEFAULT_CONFIG["diameter_nm"])
    parser.add_argument("--orientation", type=str, default=DEFAULT_CONFIG["orientation"], choices=["111", "100", "110"])
    args = parser.parse_args()

    print(f"🔨 建構 [{args.orientation}] 完美圓形奈米線 | 直徑: {args.diameter} nm")
    atoms = build_circular_nanowire(diameter_nm=args.diameter, orientation=args.orientation)

    print(f"總原子數: {len(atoms)}")
    vasp_filename = f"init_{args.orientation}_Al_{args.diameter}nm.vasp"
    write(vasp_filename, atoms, vasp5=True, direct=True)
    print(f"✅ 檔案已成功生成: {vasp_filename}")