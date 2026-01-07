#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
from ase import Atoms


def stretch_free_region_z(atoms: Atoms, fixed_mask: np.ndarray, stretch: float) -> Atoms:
    """
    依照手繪圖邏輯的「連續」拉伸（重要）：
    - bottom grip：不動
    - free region：以 a' 為錨點縮放  z' = a' + (z-a')*stretch
    - top grip：整體平移 elongation = (stretch-1)*L_free，確保在 a'' 連續

    同時更新 cell 的 z 長度（+ elongation）
    """

    if stretch <= 0:
        raise ValueError("stretch 必須 > 0，例如 1.005 或 1.01")

    atoms2 = atoms.copy()
    pos = atoms2.get_positions().copy()
    z = pos[:, 2]

    fixed_mask = fixed_mask.astype(bool)
    if fixed_mask.shape[0] != len(atoms2):
        raise ValueError("fixed_mask 長度與 atoms 不一致")

    free_mask = ~fixed_mask
    if free_mask.sum() == 0:
        raise ValueError("free region 為 0（全部都 fixed 了），請檢查 grip_thickness 或 fixed_mask")

    # 將 fixed 分成 bottom/top：用「fixed 區的中位數」當切分點，比用 box 中點更穩
    z_fixed = z[fixed_mask]
    z_split = float(np.median(z_fixed))

    bottom_mask = fixed_mask & (z <= z_split)
    top_mask = fixed_mask & (z > z_split)

    # fallback：如果某一邊為空，用 fixed 的 min/max 抓最底/最頂那層
    eps = 1e-8
    if bottom_mask.sum() == 0:
        zfmin = float(z_fixed.min())
        bottom_mask = fixed_mask & (z <= zfmin + eps)
    if top_mask.sum() == 0:
        zfmax = float(z_fixed.max())
        top_mask = fixed_mask & (z >= zfmax - eps)

    if bottom_mask.sum() == 0 or top_mask.sum() == 0:
        raise ValueError("無法判定 bottom/top grips，請檢查 fixed_mask 是否真的含上下兩端夾爪")

    # a' 與 a''：交界面
    a_prime = float(z[bottom_mask].max())         # bottom grip 上緣
    a_double_prime = float(z[top_mask].min())     # top grip 下緣

    L_free_old = a_double_prime - a_prime
    if L_free_old <= 0:
        raise ValueError(
            f"L_free_old <= 0 (a'={a_prime}, a''={a_double_prime})，表示 mask 分割或幾何有問題"
        )

    elongation = L_free_old * (stretch - 1.0)

    # 1) bottom grip：不動（pos[bottom_mask,2] 不改）
    # 2) free：以 a' 錨點縮放
    pos[free_mask, 2] = a_prime + (z[free_mask] - a_prime) * stretch
    # 3) top grip：整體平移，確保在 a'' 連續
    pos[top_mask, 2] = z[top_mask] + elongation

    atoms2.set_positions(pos)

    # 更新 cell：z 長度 + elongation（避免 top 往上跑出 cell）
    cell = atoms2.get_cell().copy()
    cell[2, 2] = cell[2, 2] + elongation
    atoms2.set_cell(cell, scale_atoms=False)

    return atoms2
