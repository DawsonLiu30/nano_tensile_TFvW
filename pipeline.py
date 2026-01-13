#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full pipeline: build Al supercell with vacuum + vacancy, prepare grips, and run tensile.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import write


def build_supercell(element: str = "Al", a: float = 4.05, reps: tuple[int, int, int] = (5, 5, 3)):
    uc = bulk(element, "fcc", a=a, cubic=True)
    atoms = uc.repeat(reps)
    return atoms


def embed_in_vacuum(atoms, vacuum: float = 5.0):
    pos = atoms.get_positions()
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)

    lengths = (maxs - mins) + 2.0 * vacuum
    cell = np.diag(lengths)

    pos_shifted = pos - mins + vacuum
    atoms.set_cell(cell)
    atoms.set_positions(pos_shifted)
    atoms.set_pbc(True)
    return atoms


def remove_center_atom(atoms):
    pos = atoms.get_positions()
    center = pos.mean(axis=0)
    d2 = ((pos - center) ** 2).sum(axis=1)
    idx = int(np.argmin(d2))
    removed_pos = pos[idx].copy()
    del atoms[idx]
    return idx, removed_pos


def select_grip_indices(atoms, axis: int = 2, grip_width: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    pos = atoms.get_positions()
    z = pos[:, axis]
    z_min = float(z.min())
    z_max = float(z.max())
    bottom = np.where(z <= z_min + grip_width)[0]
    top = np.where(z >= z_max - grip_width)[0]
    if bottom.size == 0 or top.size == 0:
        raise RuntimeError("Failed to select grip atoms; increase grip_width.")
    overlap = np.intersect1d(bottom, top)
    if overlap.size > 0:
        raise RuntimeError("Grip selections overlap; reduce grip_width.")
    return bottom.astype(int), top.astype(int)


def prepare_tensile_inputs(workdir: Path, atoms, bottom_idx: np.ndarray, top_idx: np.ndarray) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    init_xyz = workdir / "init.xyz"
    write(str(init_xyz), atoms)
    np.save(str(workdir / "bottom_idx.npy"), bottom_idx)
    np.save(str(workdir / "top_idx.npy"), top_idx)
    return init_xyz


def run_pipeline(
    *,
    workdir: str | Path,
    pp: str | Path,
    spacing: float,
    step: float = 0.005,
    cycles: int = 200,
    fmax: float = 0.08,
    relax_steps: int = 80,
    debug_strain: bool = False,
    element: str = "Al",
    a: float = 4.05,
    reps: tuple[int, int, int] = (5, 5, 3),
    vacuum: float = 5.0,
    grip_width: float = 2.0,
) -> None:
    workdir = Path(workdir).resolve()

    atoms = build_supercell(element=element, a=a, reps=reps)
    atoms = embed_in_vacuum(atoms, vacuum=vacuum)
    removed_idx, removed_pos = remove_center_atom(atoms)

    bottom_idx, top_idx = select_grip_indices(atoms, axis=2, grip_width=grip_width)
    init_xyz = prepare_tensile_inputs(workdir, atoms, bottom_idx, top_idx)

    manifest = {
        "structure": {
            "element": element,
            "a": a,
            "reps": list(reps),
            "vacuum": vacuum,
            "vacancy_removed_index": removed_idx,
            "vacancy_removed_position": removed_pos.tolist(),
            "init_xyz": str(init_xyz),
            "bottom_idx": str(workdir / "bottom_idx.npy"),
            "top_idx": str(workdir / "top_idx.npy"),
        },
        "tensile": {
            "pp": str(Path(pp).expanduser().resolve()),
            "spacing": spacing,
            "step": step,
            "cycles": cycles,
            "fmax": fmax,
            "relax_steps": relax_steps,
            "debug_strain": debug_strain,
        },
    }
    (workdir / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2))

    repo_root = Path(__file__).resolve().parent
    tensile_dir = repo_root / "01_tensile"
    if str(tensile_dir) not in sys.path:
        sys.path.insert(0, str(tensile_dir))
    from main import run_tensile

    run_tensile(
        workdir=workdir,
        init=init_xyz,
        pp=pp,
        spacing=spacing,
        step=step,
        cycles=cycles,
        fmax=fmax,
        relax_steps=relax_steps,
        debug_strain=debug_strain,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--pp", required=True)
    ap.add_argument("--spacing", type=float, required=True)
    ap.add_argument("--step", type=float, default=0.005)
    ap.add_argument("--cycles", type=int, default=200)
    ap.add_argument("--fmax", type=float, default=0.08)
    ap.add_argument("--relax-steps", type=int, default=80)
    ap.add_argument("--debug-strain", action="store_true")
    ap.add_argument("--element", default="Al")
    ap.add_argument("--a", type=float, default=4.05)
    ap.add_argument("--reps", type=int, nargs=3, default=[5, 5, 3])
    ap.add_argument("--vacuum", type=float, default=5.0)
    ap.add_argument("--grip-width", type=float, default=2.0)
    args = ap.parse_args()

    run_pipeline(
        workdir=args.workdir,
        pp=args.pp,
        spacing=float(args.spacing),
        step=float(args.step),
        cycles=int(args.cycles),
        fmax=float(args.fmax),
        relax_steps=int(args.relax_steps),
        debug_strain=bool(args.debug_strain),
        element=args.element,
        a=float(args.a),
        reps=tuple(args.reps),
        vacuum=float(args.vacuum),
        grip_width=float(args.grip_width),
    )


if __name__ == "__main__":
    main()
