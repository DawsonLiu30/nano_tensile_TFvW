import numpy as np


def _as_1d_int_array(idx, name="idx"):
    """Convert any index-like input to a flat int numpy array."""
    arr = np.asarray(idx, dtype=int).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    return arr


def compute_gauge(atoms, bottom_idx, top_idx, axis=2):
    """
    Compute gauge geometry based on current positions and grip indices.

    Returns
    -------
    info : dict with keys
        z_bot, z_top, L (all float)
    """
    pos = atoms.get_positions()
    bottom_idx = _as_1d_int_array(bottom_idx, "bottom_idx")
    top_idx = _as_1d_int_array(top_idx, "top_idx")

    z = pos[:, axis]
    z_bot = float(z[bottom_idx].max())
    z_top = float(z[top_idx].min())
    L = z_top - z_bot
    if L <= 0:
        raise ValueError(f"Bad free length: z_bot={z_bot}, z_top={z_top}, L={L}")

    return {"z_bot": z_bot, "z_top": z_top, "L": L}


def apply_strain(
    atoms,
    strain_rate,
    bottom_idx,
    top_idx,
    axis=2,
    debug=False,
    gauge_idx=None,
    shift_top=True,
    keep_cell_consistent=True,
    return_info=False,
):
    """
    Apply an axial tensile strain to atoms by:
      - Keeping bottom & top grip atoms fixed (rigid bodies),
      - Affinely stretching the gauge region,
      - Optionally shifting the top grip by dL,
      - Optionally extending the simulation cell by dL.

    Parameters
    ----------
    atoms : ASE Atoms
    strain_rate : float
        Engineering strain increment per cycle, e.g. 0.01 for +1%.
    bottom_idx, top_idx : array-like
        Indices of grip atoms (fixed). Usually loaded from bottom_idx.npy / top_idx.npy.
    axis : int
        Axis index (default 2 for z).
    debug : bool
        Print detailed diagnostics.
    gauge_idx : array-like or None
        If provided, ONLY these atoms (excluding fixed grips) are affinely stretched.
        This lets you define a clean gauge region independent of bottom/top selection.
        If None (default), all non-grip atoms are stretched (backward compatible).
    shift_top : bool
        If True (default), translate the entire top grip by +dL.
    keep_cell_consistent : bool
        If True (default), increase cell[axis,axis] by dL (scale_atoms=False).
    return_info : bool
        If True, return (atoms, info_dict). Else return atoms only.

    Notes
    -----
    - This function does NOT decide grip thickness. That comes from how bottom/top idx are generated.
    - For sanity tests, using return_info=True gives you L and dL for reporting.
    """
    pos = atoms.get_positions()

    bottom_idx = _as_1d_int_array(bottom_idx, "bottom_idx")
    top_idx = _as_1d_int_array(top_idx, "top_idx")

    # Basic fixed mask
    n = len(pos)
    fixed = np.zeros(n, dtype=bool)
    fixed[bottom_idx] = True
    fixed[top_idx] = True

    # Gauge geometry based on current positions
    g = compute_gauge(atoms, bottom_idx, top_idx, axis=axis)
    z_bot, z_top, L = g["z_bot"], g["z_top"], g["L"]

    # Strain increment in length
    dL = L * float(strain_rate)

    z = pos[:, axis]

    # Decide which atoms to stretch (gauge atoms)
    if gauge_idx is None:
        stretch_mask = ~fixed
    else:
        gauge_idx = _as_1d_int_array(gauge_idx, "gauge_idx")
        stretch_mask = np.zeros(n, dtype=bool)
        stretch_mask[gauge_idx] = True
        # Never stretch fixed grip atoms
        stretch_mask &= ~fixed

    # Affine stretch only for selected atoms
    if np.any(stretch_mask):
        t = (z[stretch_mask] - z_bot) / L
        pos[stretch_mask, axis] = z_bot + t * (L + dL)

    # Rigid shift top grip
    if shift_top:
        pos[top_idx, axis] += dL

    atoms.set_positions(pos)

    # Extend cell in tensile direction (do not scale atoms)
    if keep_cell_consistent:
        cell = atoms.get_cell().copy()
        cell[axis, axis] += dL
        atoms.set_cell(cell, scale_atoms=False)

    # Diagnostics
    info = {
        "axis": int(axis),
        "strain_rate": float(strain_rate),
        "z_bot": float(z_bot),
        "z_top": float(z_top),
        "L": float(L),
        "dL": float(dL),
        "n_atoms": int(n),
        "n_fixed": int(fixed.sum()),
        "n_stretched": int(stretch_mask.sum()),
        "shift_top": bool(shift_top),
        "keep_cell_consistent": bool(keep_cell_consistent),
    }

    if debug:
        z2 = atoms.get_positions()[:, axis]
        print(
            f"[apply_strain] axis={axis} strain_rate={float(strain_rate):.6f} "
            f"z_bot={z_bot:.6f} z_top={z_top:.6f} L={L:.6f} dL={dL:.6f}"
        )
        print(
            f"[apply_strain] fixed atoms: bottom={len(bottom_idx)} top={len(top_idx)} "
            f"total_fixed={fixed.sum()} stretched={stretch_mask.sum()}"
        )
        print(
            f"[apply_strain] top z(min/mean/max)={z2[top_idx].min():.6f}/"
            f"{z2[top_idx].mean():.6f}/{z2[top_idx].max():.6f}"
        )
        print(
            f"[apply_strain] bot z(min/mean/max)={z2[bottom_idx].min():.6f}/"
            f"{z2[bottom_idx].mean():.6f}/{z2[bottom_idx].max():.6f}"
        )
        if np.any(stretch_mask):
            print(
                f"[apply_strain] stretched z(min/max)={z2[stretch_mask].min():.6f}/"
                f"{z2[stretch_mask].max():.6f}"
            )

    if return_info:
        return atoms, info
    return atoms

