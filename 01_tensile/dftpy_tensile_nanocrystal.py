#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple, Dict

import numpy as np

import ase.io
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io.trajectory import Trajectory

from dftpy.config import DefaultOption, OptionFormat
from dftpy.api.api4ase import DFTpyCalculator


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--pppath", required=True)
    p.add_argument("--ppfile", required=True)
    p.add_argument("--element", default="Al")

    # OF-DFT
    p.add_argument("--kedf", default="TFvW")  # 你要 TFvW
    p.add_argument("--calctype", default="Energy Force")
    p.add_argument("--optimizer", default="BFGS", choices=["BFGS", "FIRE"])
    p.add_argument("--fmax", type=float, default=0.08)
    p.add_argument("--relax_steps", type=int, default=80)

    # tensile protocol
    p.add_argument("--grip_thickness", type=float, default=3.0)  # top+bottom fixed thickness (Å)
    p.add_argument("--stretch", type=float, default=1.01)        # 1% each cycle
    p.add_argument("--max_cycles", type=int, default=5000)
    p.add_argument("--fracture_factor", type=float, default=3.0) # thr = 3 * nn0

    # grid
    p.add_argument("--spacing", type=float, default=0.35)

    # fracture detection (focus on middle of b segment)
    p.add_argument(
        "--gap_window",
        type=float,
        default=2.0,
        help="Use atoms in free region within z_mid±gap_window (Å) to compute gap_mid_A.",
    )

    # performance hints (optional; mostly controlled by sbatch env)
    p.add_argument("--print_threads", action="store_true", help="Print OMP/MKL/OPENBLAS thread envs.")

    return p.parse_args()


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def nearest_neighbor_distance_global(atoms: Atoms) -> float:
    """Return global nearest-neighbor distance (min over all atom pairs). O(N^2) but N~300 ok."""
    pos = atoms.get_positions()
    n = len(pos)
    dmin = 1e30
    for i in range(n - 1):
        rij = pos[i + 1 :] - pos[i]
        dij = np.sqrt((rij**2).sum(axis=1))
        dmin = min(dmin, float(dij.min()))
    return float(dmin)


def build_fixed_mask_grips(atoms: Atoms, grip_thickness: float) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Define fixed grips by absolute z (based on initial structure):
      fixed if z <= zmin+thickness OR z >= zmax-thickness
    This keeps the same mask forever (important!).
    """
    z = atoms.get_positions()[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    z_low = zmin + grip_thickness
    z_high = zmax - grip_thickness
    fixed = (z <= z_low) | (z >= z_high)

    info = {
        "zmin": zmin,
        "zmax": zmax,
        "z_low": z_low,
        "z_high": z_high,
        "z_mid": 0.5 * (z_low + z_high),
        "L0_free": z_high - z_low,  # b length in your notation
    }
    return fixed, info


def free_region_length_measured(atoms: Atoms, fixed_mask: np.ndarray) -> float:
    """
    Measure current free-region length by z-range of FREE atoms only.
    This can shrink/expand after relax (that's physics).
    """
    z = atoms.get_positions()[:, 2]
    free = ~fixed_mask
    zf = z[free]
    if zf.size < 2:
        return 0.0
    return float(zf.max() - zf.min())


def compute_gap_mid_max_dz(atoms: Atoms, fixed_mask: np.ndarray, z_mid: float, gap_window: float) -> float:
    """
    gap_mid_A (NEW, better for fracture):
      - select free atoms within z_mid±gap_window
      - sort their z
      - return max consecutive dz (largest void along z in the mid section)
    If too few atoms in window -> fallback to all free atoms.
    """
    z = atoms.get_positions()[:, 2]
    free = ~fixed_mask
    mid = (z >= (z_mid - gap_window)) & (z <= (z_mid + gap_window))
    sel = np.where(free & mid)[0]

    if sel.size < 3:
        sel = np.where(free)[0]
        if sel.size < 3:
            return 0.0

    zsel = np.sort(z[sel])
    dz = np.diff(zsel)
    if dz.size == 0:
        return 0.0
    return float(dz.max())


def stretch_free_region_z(atoms: Atoms, fixed_mask: np.ndarray, z_mid: float, stretch: float) -> Atoms:
    """
    Scale ONLY free region along z about z_mid; grips remain unchanged.
    """
    pos = atoms.get_positions().copy()
    free = ~fixed_mask
    pos[free, 2] = z_mid + stretch * (pos[free, 2] - z_mid)
    atoms.set_positions(pos)
    return atoms


# -----------------------------
# DFTpy
# -----------------------------
def make_dftpy_config(args: argparse.Namespace):
    """
    Conservative config to avoid DFTpy OptionFormat KeyError across versions.
    """
    conf = DefaultOption()

    conf["PATH"]["pppath"] = str(Path(args.pppath))
    conf["PP"][args.element] = str(args.ppfile)

    conf["JOB"]["calctype"] = args.calctype
    conf["KEDF"]["kedf"] = args.kedf
    conf["GRID"]["spacing"] = float(args.spacing)

    return OptionFormat(conf)


def relax_with_dftpy(atoms: Atoms, fixed_mask: np.ndarray, args: argparse.Namespace) -> Tuple[Atoms, float]:
    atoms = atoms.copy()
    atoms.set_constraint(FixAtoms(mask=fixed_mask))

    conf = make_dftpy_config(args)
    atoms.calc = DFTpyCalculator(config=conf)

    if args.optimizer.upper() == "FIRE":
        from ase.optimize import FIRE as ASEOpt
    else:
        from ase.optimize import BFGS as ASEOpt

    opt = ASEOpt(atoms, logfile=None)
    opt.run(fmax=args.fmax, steps=args.relax_steps)

    E = float(atoms.get_potential_energy())  # eV
    return atoms, E


def single_point_energy(atoms: Atoms, fixed_mask: np.ndarray, args: argparse.Namespace) -> float:
    try:
        sp = atoms.copy()
        sp.set_constraint(FixAtoms(mask=fixed_mask))
        conf = make_dftpy_config(args)
        sp.calc = DFTpyCalculator(config=conf)
        return float(sp.get_potential_energy())
    except Exception:
        return float("nan")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    if args.print_threads:
        keys = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
        print("[INFO] Thread envs:")
        for k in keys:
            print(f"  {k}={os.environ.get(k, '')}")

    traj_path = outdir / "tensile.traj"
    log_path = outdir / "tensile_log.csv"
    err_path = outdir / "error_last.txt"

    print(f"[INFO] Reading structure: {args.input}")
    atoms = ase.io.read(args.input)

    fixed_mask, zinfo = build_fixed_mask_grips(atoms, args.grip_thickness)

    nn0 = nearest_neighbor_distance_global(atoms)
    thr = args.fracture_factor * nn0

    L0_free = float(zinfo["L0_free"])
    z_mid = float(zinfo["z_mid"])

    print(f"[INFO] nn0_A={nn0:.6f}  fracture_thr_A={thr:.6f}  (factor={args.fracture_factor})")
    print(
        f"[INFO] grip_thickness={args.grip_thickness:.3f} A  "
        f"z_low={zinfo['z_low']:.3f}  z_high={zinfo['z_high']:.3f}  z_mid={z_mid:.3f}  "
        f"L0_free(b)={L0_free:.6f}  gap_window={args.gap_window:.3f}"
    )

    traj = Trajectory(str(traj_path), "w", atoms)

    if not log_path.exists():
        with open(log_path, "w") as f:
            f.write(
                "cycle,stage,"
                "L0_free_A,L_free_meas_A,"
                "applied_strain_percent,measured_strain_percent,"
                "energy_eV,gap_mid_A,nn0_A,fracture_thr_A,broken\n"
            )

    stretch_acc = 1.0  # cumulative applied stretch (monotonic)

    def write_log(cycle: int, stage: str, E: float, gap_mid: float, broken: int, atoms_now: Atoms):
        L_meas = free_region_length_measured(atoms_now, fixed_mask)
        applied_strain = (stretch_acc - 1.0) * 100.0
        measured_strain = (L_meas / L0_free - 1.0) * 100.0 if L0_free > 0 else 0.0

        with open(log_path, "a") as f:
            f.write(
                f"{cycle},{stage},"
                f"{L0_free:.15g},{L_meas:.15g},"
                f"{applied_strain:.15g},{measured_strain:.15g},"
                f"{E:.15g},{gap_mid:.15g},{nn0:.15g},{thr:.15g},{broken}\n"
            )

    try:
        ase.io.write(outdir / "cycle_000_start.xyz", atoms)

        for cycle in range(args.max_cycles + 1):
            # ---- relax
            print(f"[INFO] Cycle {cycle}: relax (free region only)")
            atoms_relaxed, E_relax = relax_with_dftpy(atoms, fixed_mask, args)
            traj.write(atoms_relaxed)
            ase.io.write(outdir / f"cycle_{cycle:03d}_relaxed.xyz", atoms_relaxed)

            gap_mid = compute_gap_mid_max_dz(atoms_relaxed, fixed_mask, z_mid, args.gap_window)
            broken = 1 if gap_mid >= thr else 0
            write_log(cycle, "relax", E_relax, gap_mid, broken, atoms_relaxed)

            if broken:
                print(f"[STOP] Fractured after relax: gap_mid={gap_mid:.4f} >= thr={thr:.4f}")
                break

            if cycle == args.max_cycles:
                print("[WARN] Reached max_cycles without fracture.")
                break

            # ---- stretch (apply to free region only)
            atoms_stretched = atoms_relaxed.copy()
            atoms_stretched = stretch_free_region_z(atoms_stretched, fixed_mask, z_mid, args.stretch)
            stretch_acc *= args.stretch  # applied strain accumulates here (monotonic)

            ase.io.write(outdir / f"cycle_{cycle+1:03d}_stretched.xyz", atoms_stretched)

            E_sp = single_point_energy(atoms_stretched, fixed_mask, args)
            gap_mid_s = compute_gap_mid_max_dz(atoms_stretched, fixed_mask, z_mid, args.gap_window)
            broken_s = 1 if gap_mid_s >= thr else 0
            write_log(cycle + 1, "stretch", E_sp, gap_mid_s, broken_s, atoms_stretched)

            if broken_s:
                print(f"[STOP] Fractured right after stretch: gap_mid={gap_mid_s:.4f} >= thr={thr:.4f}")
                break

            atoms = atoms_stretched

    except Exception as e:
        with open(err_path, "w") as f:
            f.write("CRASHED:\n")
            f.write(repr(e) + "\n")
        raise
    finally:
        traj.close()


if __name__ == "__main__":
    main()

