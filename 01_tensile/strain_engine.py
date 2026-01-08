#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
from ase import Atoms


def stretch_free_region_z_by_indices(
    atoms: Atoms,
    bottom_idx: np.ndarray,
    top_idx: np.ndarray,
    free_idx: np.ndarray,
    stretch: float,
) -> Atoms:
    """
    真・拉伸（固定 index，不重切、不猜 top/bottom）：

    - bottom grip：固定不動
    - free region：以 a'（bottom grip 上緣）為錨點做 z 向縮放
      z_new = a' + (z_old - a') * stretch
    - top grip：整塊向上平移 elongation
      elongation = (a'' - a') * (stretch - 1)
    - cell 的 z 長度同步增加 elongation（scale_atoms=False）

    這樣 L_grip = mean(z_top) - mean(z_bottom) 會真正變大。
    """

    if stretch <= 0:
        raise ValueError(f"stretch must be > 0, got {stretch}")

    pos = atoms.get_positions().copy()
    z = pos[:, 2]

    a_prime = float(z[bottom_idx].max())       # bottom grip 上緣
    a_dprime = float(z[top_idx].min())         # top grip 下緣

    L_free_old = a_dprime - a_prime
    if L_free_old <= 0:
        raise RuntimeError(
            f"Invalid free length (a''-a'={L_free_old:.6f} Å). "
            "Your grips overlap or free region is empty. Try smaller grip_thickness."
        )

    elongation = L_free_old * (stretch - 1.0)

    # bottom grip: do nothing
    # free region: scale about a'
    pos[free_idx, 2] = a_prime + (z[free_idx] - a_prime) * stretch

    # top grip: translate upward
    pos[top_idx, 2] = z[top_idx] + elongation

    atoms.set_positions(pos)

    # update cell length along z
    cell = atoms.get_cell().copy()
    cell[2, 2] += elongation
    atoms.set_cell(cell, scale_atoms=False)

    return atoms

