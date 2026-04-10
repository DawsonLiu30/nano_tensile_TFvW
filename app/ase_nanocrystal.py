import numpy as np
from ase.io import write
from ase.lattice.cubic import FaceCenteredCubic

# 根據論文設定的預設參數
DEFAULT_CONFIG = {
    "a0": 4.05,           # Al 的晶格常數
    "diameter_nm": 2.0,   # 奈米線直徑 (論文使用 1, 2, 4, 8 nm)
    "length_z": 20.0,     # 奈米線長度 (約 20 nm 或更短)
    "vacuum": 5.0,        # 真空層大小，確保週期影像間至少有 10 Å 真空 (兩側各 5 Å)
}

def build_circular_nanowire_111(
    a0=DEFAULT_CONFIG["a0"],
    diameter_nm=DEFAULT_CONFIG["diameter_nm"],
    length_z=DEFAULT_CONFIG["length_z"],
    vacuum=DEFAULT_CONFIG["vacuum"],
):
    """
    根據論文要求，建立 [111] 晶向、圓形截面的鋁奈米線。
    """
    # 將直徑 (nm) 轉換為半徑 (Angstrom)
    radius = (diameter_nm * 10.0) / 2.0
    
    # 定義 [111] 方向的正交基礎晶胞
    # 這樣設定能確保 Z 軸完美對齊 [111] 晶向
    direction_x = [1, -1, 0]
    direction_y = [1, 1, -2]
    direction_z = [1, 1, 1]
    
    # 建立基礎正交晶胞
    base_atoms = FaceCenteredCubic(
        directions=[direction_x, direction_y, direction_z],
        size=(1, 1, 1), 
        symbol='Al', 
        pbc=True, 
        latticeconstant=a0
    )
    
    # 取得基礎晶胞的尺寸
    lx, ly, lz = base_atoms.cell.lengths()
    
    # 計算需要複製幾次才能涵蓋我們目標的圓柱體大小
    nx = int(np.ceil((2.0 * radius + 5.0) / lx))
    ny = int(np.ceil((2.0 * radius + 5.0) / ly))
    nz = int(np.ceil(length_z / lz))
    
    # 確保至少複製 1 次
    nx, ny, nz = max(1, nx), max(1, ny), max(1, nz)
    
    # 擴充成巨大的超晶胞
    supercell = base_atoms.repeat((nx, ny, nz))
    
    # 找出 XY 平面的中心點
    cx = supercell.cell[0, 0] / 2.0
    cy = supercell.cell[1, 1] / 2.0
    
    # 取得所有原子的座標
    pos = supercell.get_positions()
    
    # 計算每個原子到中心軸的距離平方 (x^2 + y^2)
    dx = pos[:, 0] - cx
    dy = pos[:, 1] - cy
    dist2 = dx**2 + dy**2
    
    # 建立圓形遮罩：保留在半徑範圍內的原子
    mask = dist2 <= radius**2
    nanowire = supercell[mask]
    
    # 重新設定晶胞大小以加入真空層 (符合論文提到的至少 10 A 真空)
    new_lx = 2.0 * radius + 2.0 * vacuum
    new_ly = 2.0 * radius + 2.0 * vacuum
    new_lz = supercell.cell[2, 2]  # Z 軸保持原本的週期長度
    
    # 將切出來的奈米線置中於新的晶胞內
    wire_pos = nanowire.get_positions()
    cur_cx = wire_pos[:, 0].mean()
    cur_cy = wire_pos[:, 1].mean()
    
    wire_pos[:, 0] += (new_lx / 2.0) - cur_cx
    wire_pos[:, 1] += (new_ly / 2.0) - cur_cy
    
    nanowire.set_positions(wire_pos)
    nanowire.set_cell([new_lx, new_ly, new_lz])
    
    # 論文中提到「所有方向皆施加週期性邊界條件」
    nanowire.pbc = [True, True, True] 
    
    return nanowire

if __name__ == "__main__":
    print(f"Building [111] Circular Nanowire with config: {DEFAULT_CONFIG}")
    atoms = build_circular_nanowire_111()

    print(f"Total Atoms: {len(atoms)}")
    print(f"Cell dimensions (Å): {atoms.get_cell().lengths()}")
    print(f"PBC: {atoms.get_pbc()}")

    write("init_111_nanowire.xyz", atoms)
    write("init_111_nanowire.vasp", atoms, vasp5=True, direct=True)
    print("Files successfully generated: init_111_nanowire.xyz, init_111_nanowire.vasp")
    print("You're all set to reproduce the paper's geometry!")
