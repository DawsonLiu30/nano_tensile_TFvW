#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import ase.io
import numpy as np

from strain_engine import stretch_free_region_z
from dft_engine import run_dft_relax


def _z_mid_from_grips(atoms, fixed_mask: np.ndarray) -> float:
    """用固定夾具（bottom/top）的平均 z 估算中央位置 z_mid。"""
    z = atoms.get_positions()[:, 2]
    z_fixed = z[fixed_mask]
    if z_fixed.size < 2:
        # fallback：用整盒中點
        return float(0.5 * (z.min() + z.max()))
    # 用 fixed 的中位數來切 bottom/top
    z_split = float(np.median(z_fixed))
    bottom = fixed_mask & (z <= z_split)
    top = fixed_mask & (z > z_split)

    if bottom.sum() == 0 or top.sum() == 0:
        return float(0.5 * (z.min() + z.max()))

    return float(0.5 * (z[bottom].mean() + z[top].mean()))


def gap_mid_window(atoms, fixed_mask: np.ndarray, window: float = 2.0) -> float:
    """
    只看「中央區域」的最大 z-gap：
    - 取自由區原子
    - 只保留 |z - z_mid| <= window 的原子
    - 排序後取最大相鄰差
    """
    z = atoms.get_positions()[:, 2]
    free = ~fixed_mask
    z_mid = _z_mid_from_grips(atoms, fixed_mask)

    sel = free & (np.abs(z - z_mid) <= window)
    zz = np.sort(z[sel])
    if zz.size < 3:
        return 0.0
    return float(np.diff(zz).max())


def max_nn_free(atoms, fixed_mask: np.ndarray) -> float:
    """
    自由區 max nearest-neighbor distance（3D）：
    - 對每個自由區原子 i，找最近的另一個自由區原子距離 d_i
    - 回傳 max(d_i)
    """
    pos = atoms.get_positions()
    idx = np.where(~fixed_mask)[0]
    if idx.size < 2:
        return 0.0

    P = pos[idx]  # (M, 3)
    M = P.shape[0]

    # 直接 O(M^2) 距離矩陣（M~幾百可接受）
    # dist2[i,i] = +inf 避免選到自己
    diff = P[:, None, :] - P[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, np.inf)

    nn = np.sqrt(np.min(dist2, axis=1))  # 每個點最近鄰
    return float(np.max(nn))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--stretch", type=float, default=1.005)
    parser.add_argument("--grip_thickness", type=float, default=3.0)

    # DFTpy
    parser.add_argument("--pppath", required=True)
    parser.add_argument("--ppfile", required=True)
    parser.add_argument("--element", default="Al")
    parser.add_argument("--kedf", default="TFvW")
    parser.add_argument("--spacing", type=float, default=0.35)

    parser.add_argument("--optimizer", default="FIRE", choices=["FIRE", "BFGS"])
    parser.add_argument("--fmax", type=float, default=0.08)
    parser.add_argument("--relax_steps", type=int, default=80)
    parser.add_argument("--max_cycles", type=int, default=200)
    parser.add_argument("--opt_log", default=None)

    # Fracture criteria
    parser.add_argument("--fracture_mode", default="gap_mid", choices=["gap_mid", "max_nn", "both"],
                        help="gap_mid: central-window z-gap; max_nn: max nearest-neighbor; both: require both exceed thresholds")
    parser.add_argument("--gap_window", type=float, default=2.0, help="Å, only used for gap_mid")
    parser.add_argument("--gap_break", type=float, default=3.0, help="Å, fracture if gap_mid > this (when used)")
    parser.add_argument("--maxnn_break", type=float, default=3.0 * 2.86, help="Å, fracture if max_nn > this (when used)")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    atoms = ase.io.read(args.input)

    # 固定夾具 mask：只算一次（固定同一群原子）
    z = atoms.get_positions()[:, 2]
    fixed_mask = (z <= z.min() + args.grip_thickness) | (z >= z.max() - args.grip_thickness)

    print(f"=== Starting Tensile Loop: {args.input} ===")
    print(f"fixed atoms: {int(fixed_mask.sum())} / total {len(atoms)}")
    print(f"fracture_mode = {args.fracture_mode}")
    print(f"gap_window={args.gap_window} Å, gap_break={args.gap_break} Å")
    print(f"maxnn_break={args.maxnn_break} Å")

    for cycle in range(args.max_cycles):
        cyc = f"{cycle:03d}"

        # 1) Relax
        print(f"\nCycle {cyc}: Relaxing...")
        if args.opt_log is None:
            args.opt_log = str(outdir / "opt.log")

        atoms, energy = run_dft_relax(
            atoms, fixed_mask, args, workdir=(outdir / f"relax_{cyc}")
        )

        relaxed_path = outdir / f"cycle_{cyc}_relaxed.xyz"
        ase.io.write(str(relaxed_path), atoms)

        # 2) Fracture check on RELAXED structure
        gap = gap_mid_window(atoms, fixed_mask, window=args.gap_window)
        maxnn = max_nn_free(atoms, fixed_mask)

        # 判斷規則
        gap_fail = (gap >= args.gap_break)
        maxnn_fail = (maxnn >= args.maxnn_break)

        if args.fracture_mode == "gap_mid":
            fractured = gap_fail
        elif args.fracture_mode == "max_nn":
            fractured = maxnn_fail
        else:  # both
            fractured = (gap_fail and maxnn_fail)

        print(f"  Energy: {energy:.6f} (units follow calculator)")
        print(f"  gap_mid(window={args.gap_window}Å): {gap:.6f} Å (break>{args.gap_break}) -> {'FAIL' if gap_fail else 'OK'}")
        print(f"  max_nn(free): {maxnn:.6f} Å (break>{args.maxnn_break}) -> {'FAIL' if maxnn_fail else 'OK'}")

        if fractured:
            print(f"!!! FRACTURE DETECTED at cycle {cyc} !!!")
            break

        # 3) Stretch
        print(f"Cycle {cyc}: Stretching (x{args.stretch})...")
        atoms = stretch_free_region_z(atoms, fixed_mask, args.stretch)

        stretched_path = outdir / f"cycle_{cyc}_stretched.xyz"
        ase.io.write(str(stretched_path), atoms)

    print("\n=== Finished ===")


if __name__ == "__main__":
    main()
