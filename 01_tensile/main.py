#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import ase.io

from dft_engine import run_dft_relax
from strain_engine import stretch_free_region_z_by_indices


AL_NN = 2.86  # 你先用這個近似（FCC Al 最近鄰距離量級）


def L_grip(atoms, bottom_idx: np.ndarray, top_idx: np.ndarray) -> float:
    z = atoms.get_positions()[:, 2]
    return float(z[top_idx].mean() - z[bottom_idx].mean())


def gap_mid_window(atoms, bottom_idx: np.ndarray, top_idx: np.ndarray, free_idx: np.ndarray, window: float) -> float:
    z = atoms.get_positions()[:, 2]
    z_mid = 0.5 * (float(z[bottom_idx].mean()) + float(z[top_idx].mean()))
    sel = free_idx[np.abs(z[free_idx] - z_mid) <= window]
    if sel.size < 3:
        return 0.0
    zz = np.sort(z[sel])
    return float(np.diff(zz).max())


def max_nn_free(atoms, free_idx: np.ndarray) -> float:
    if free_idx.size < 2:
        return 0.0
    pos = atoms.get_positions()[free_idx]  # (M,3)
    diff = pos[:, None, :] - pos[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, np.inf)
    nn = np.sqrt(np.min(dist2, axis=1))
    return float(np.max(nn))


def main():
    parser = argparse.ArgumentParser()

    # IO
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", default="results")

    # Tensile
    parser.add_argument("--stretch", type=float, default=1.01)
    parser.add_argument("--grip_thickness", type=float, default=1.0)

    # DFTpy
    parser.add_argument("--pppath", required=True)
    parser.add_argument("--ppfile", required=True)
    parser.add_argument("--element", default="Al")
    parser.add_argument("--kedf", default="TFvW")
    parser.add_argument("--spacing", type=float, default=0.35)
    parser.add_argument("--optimizer", default="BFGS", choices=["FIRE", "BFGS"])
    parser.add_argument("--fmax", type=float, default=0.08)
    parser.add_argument("--relax_steps", type=int, default=80)

    # Loop
    parser.add_argument("--max_cycles", type=int, default=200)

    # Fracture (新制)
    parser.add_argument("--fracture_mode", default="gap_mid", choices=["gap_mid", "max_nn", "both"])
    parser.add_argument("--gap_window", type=float, default=2.0)
    parser.add_argument("--gap_break", type=float, default=None)      # 若 None → 由 fracture_factor 決定
    parser.add_argument("--maxnn_break", type=float, default=None)    # 若 None → fracture_factor * 2.86

    # Fracture (舊制相容：你之前 sbatch 用 fracture_factor=3.0)
    parser.add_argument("--fracture_factor", type=float, default=3.0)

    # 舊 sbatch 可能有 calctype（不使用但吃下來避免 argparse 爆炸）
    parser.add_argument("--calctype", default=None)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    atoms = ase.io.read(args.input)

    # ----- grips selection ONCE -----
    z0 = atoms.get_positions()[:, 2]
    zmin0, zmax0 = float(z0.min()), float(z0.max())

    bottom_idx = np.where(z0 <= zmin0 + args.grip_thickness)[0]
    top_idx = np.where(z0 >= zmax0 - args.grip_thickness)[0]

    if bottom_idx.size == 0 or top_idx.size == 0:
        raise RuntimeError(
            f"Grip selection failed: bottom={bottom_idx.size}, top={top_idx.size}. "
            f"Try adjusting --grip_thickness (current {args.grip_thickness} Å)."
        )

    fixed_mask = np.zeros(len(atoms), dtype=bool)
    fixed_mask[bottom_idx] = True
    fixed_mask[top_idx] = True
    free_idx = np.where(~fixed_mask)[0]

    # Save indices for analysis (這超重要，不然你會用漂移的尺量)
    np.save(outdir / "bottom_idx.npy", bottom_idx)
    np.save(outdir / "top_idx.npy", top_idx)
    np.save(outdir / "free_idx.npy", free_idx)
    np.save(outdir / "fixed_mask.npy", fixed_mask)

    # thresholds
    gap_break = args.gap_break if args.gap_break is not None else float(args.fracture_factor)
    maxnn_break = args.maxnn_break if args.maxnn_break is not None else float(args.fracture_factor) * AL_NN

    print(f"=== Starting Tensile Loop: {args.input} ===")
    print(f"total atoms: {len(atoms)}")
    print(f"fixed atoms: {int(fixed_mask.sum())} (bottom={len(bottom_idx)}, top={len(top_idx)}), free={len(free_idx)}")
    print(f"stretch per cycle = x{args.stretch}")
    print(f"fracture_mode = {args.fracture_mode}")
    print(f"gap_window={args.gap_window} Å, gap_break={gap_break} Å")
    print(f"maxnn_break={maxnn_break:.2f} Å (fracture_factor={args.fracture_factor} * {AL_NN})")

    L0 = L_grip(atoms, bottom_idx, top_idx)
    print(f"L_grip baseline (L0) = {L0:.6f} Å")

    for cycle in range(args.max_cycles):
        cyc = f"{cycle:03d}"

        # 1) Relax
        print(f"\nCycle {cyc}: Relaxing...")
        atoms, energy = run_dft_relax(atoms, fixed_mask, args)

        ase.io.write(str(outdir / f"cycle_{cyc}_relaxed.xyz"), atoms)

        Lr = L_grip(atoms, bottom_idx, top_idx)
        strain = (Lr - L0) / L0 if L0 != 0 else 0.0

        gap = gap_mid_window(atoms, bottom_idx, top_idx, free_idx, window=float(args.gap_window))
        maxnn = max_nn_free(atoms, free_idx)

        gap_fail = gap >= gap_break
        maxnn_fail = maxnn >= maxnn_break

        if args.fracture_mode == "gap_mid":
            fractured = gap_fail
        elif args.fracture_mode == "max_nn":
            fractured = maxnn_fail
        else:
            fractured = gap_fail and maxnn_fail

        print(f"  Energy: {energy:.6f} (units follow calculator)")
        print(f"  L_grip(relaxed) = {Lr:.6f} Å  strain={strain:.6f}")
        print(f"  gap_mid(window={args.gap_window}Å) = {gap:.6f} Å  -> {'FAIL' if gap_fail else 'OK'}")
        print(f"  max_nn(free) = {maxnn:.6f} Å  -> {'FAIL' if maxnn_fail else 'OK'}")

        if fractured:
            print(f"!!! FRACTURE DETECTED at cycle {cyc} !!!")
            break

        # 2) Stretch (關鍵：top grip 會真的往上推)
        print(f"Cycle {cyc}: Stretching (x{args.stretch})...")
        atoms = stretch_free_region_z_by_indices(atoms, bottom_idx, top_idx, free_idx, float(args.stretch))
        ase.io.write(str(outdir / f"cycle_{cyc}_stretched.xyz"), atoms)

        Ls = L_grip(atoms, bottom_idx, top_idx)
        print(f"  L_grip(stretched) = {Ls:.6f} Å  ratio(stretched/relaxed)={Ls/Lr:.8f}")

    print("\n=== Finished ===")


if __name__ == "__main__":
    main()

