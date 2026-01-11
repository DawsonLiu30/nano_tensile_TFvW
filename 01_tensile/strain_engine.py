import numpy as np

def apply_strain(atoms, strain_rate, bottom_idx, top_idx, axis=2, debug=False):

    pos = atoms.get_positions()

    bottom_idx = np.asarray(bottom_idx, dtype=int).ravel()
    top_idx    = np.asarray(top_idx, dtype=int).ravel()

    z = pos[:, axis]
    z_bot = float(z[bottom_idx].max())
    z_top = float(z[top_idx].min())
    L = z_top - z_bot
    if L <= 0:
        raise ValueError(f"Bad free length: z_bot={z_bot}, z_top={z_top}, L={L}")

    dL = L * float(strain_rate)

    n = len(pos)
    fixed = np.zeros(n, dtype=bool)
    fixed[bottom_idx] = True
    fixed[top_idx] = True
    free = ~fixed

    # free: 線性 mapping（以 bottom 上緣為錨點）
    t = (z[free] - z_bot) / L
    pos[free, axis] = z_bot + t * (L + dL)

    # top: 整塊平移
    pos[top_idx, axis] += dL

    # 寫回原子座標
    atoms.set_positions(pos)

    # 更新 cell（不縮放原子，因為我們已手動移動）
    cell = atoms.get_cell().copy()
    cell[axis, axis] += dL
    atoms.set_cell(cell, scale_atoms=False)

    if debug:
        z2 = atoms.get_positions()[:, axis]
        print(f"[apply_strain] z_bot={z_bot:.6f} z_top={z_top:.6f} L={L:.6f} dL={dL:.6f}")
        print(f"[apply_strain] top z(min/mean/max)={z2[top_idx].min():.6f}/{z2[top_idx].mean():.6f}/{z2[top_idx].max():.6f}")
        print(f"[apply_strain] bot z(min/mean/max)={z2[bottom_idx].min():.6f}/{z2[bottom_idx].mean():.6f}/{z2[bottom_idx].max():.6f}")

    return atoms
