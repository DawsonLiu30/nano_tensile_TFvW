from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write
from scipy.spatial import cKDTree

from strain_engine import apply_strain
from dft_engine import relax_atoms


def _mean_nn_distance(atoms, mask: np.ndarray) -> float:
    pos = atoms.get_positions()[mask]
    if len(pos) < 2:
        return 0.0
    tree = cKDTree(pos)
    dists, _ = tree.query(pos, k=2)
    return float(dists[:, 1].mean())


def _max_nn_distance(atoms, mask: np.ndarray) -> float:
    pos = atoms.get_positions()[mask]
    if len(pos) < 2:
        return 0.0
    tree = cKDTree(pos)
    dists, _ = tree.query(pos, k=2)
    return float(dists[:, 1].max())


def _parse_stress_zz_from_out(out_file: Path) -> float | None:
    if not out_file.exists():
        return None
    txt = out_file.read_text()
    key = "TOTAL stress (GPa):"
    i = txt.find(key)
    if i < 0:
        return None
    lines = txt[i:].splitlines()
    if len(lines) < 4:
        return None
    row3 = lines[3].split()
    if len(row3) < 3:
        return None
    try:
        return float(row3[2])
    except Exception:
        return None


def _parse_energy_from_out(out_file: Path) -> float | None:
    if not out_file.exists():
        return None
    for line in out_file.read_text().splitlines():
        if line.startswith("total energy (eV)"):
            try:
                return float(line.split(":")[1].strip())
            except Exception:
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--pp", required=True)
    ap.add_argument("--spacing", type=float, required=True)
    ap.add_argument("--step", type=float, default=0.005)
    ap.add_argument("--cycles", type=int, default=200)
    ap.add_argument("--fmax", type=float, default=0.08)
    ap.add_argument("--relax-steps", type=int, default=80)
    ap.add_argument("--debug-strain", action="store_true")
    args = ap.parse_args()

    workdir = Path(args.workdir).resolve()
    init_path = Path(args.init).resolve() if Path(args.init).is_absolute() else (workdir / args.init).resolve()
    results = workdir / "results"
    results.mkdir(parents=True, exist_ok=True)

    bottom_idx = np.load(str(workdir / "bottom_idx.npy")).astype(int).ravel()
    top_idx = np.load(str(workdir / "top_idx.npy")).astype(int).ravel()
    fixed_idx = np.unique(np.concatenate([bottom_idx, top_idx])).astype(int)

    atoms0 = read(str(init_path))

    write(str(results / "cycle_000_relaxed.xyz"), atoms0)

    z0 = atoms0.get_positions()[:, 2]
    zmin0 = float(z0[bottom_idx].max())
    zmax0 = float(z0[top_idx].min())
    if zmax0 <= zmin0:
        raise RuntimeError(f"Bad grips: zmin={zmin0}, zmax={zmax0}")
    free_mask = (z0 > zmin0) & (z0 < zmax0)

    L0 = float(zmax0 - zmin0)
    d0 = _mean_nn_distance(atoms0, free_mask)
    th = 3.0 * d0
    print(f"[init] L_free0={L0:.6f} Å  d0={d0:.6f} Å  fracture_th(3*d0)={th:.6f} Å")

    summary = results / "summary.csv"
    if not summary.exists():
        summary.write_text("cycle,strain,energy_eV,sigma_zz_GPa,max_nn_A\n")

    atoms = atoms0

    for cyc in range(1, int(args.cycles) + 1):
        print(f"=== Cycle {cyc} ===")

        atoms_st = atoms.copy()
        apply_strain(
            atoms_st,
            strain_rate=float(args.step),
            bottom_idx=bottom_idx,
            top_idx=top_idx,
            axis=2,
            debug=bool(args.debug_strain),
        )
        stretched_xyz = results / f"cycle_{cyc:03d}_stretched.xyz"
        write(str(stretched_xyz), atoms_st)

        relax_log = results / f"cycle_{cyc:03d}_relax.log"
        relax_traj = results / f"cycle_{cyc:03d}_relax.traj"
        dftpy_out = results / f"cycle_{cyc:03d}_dftpy.out"

        atoms_rlx = relax_atoms(
            atoms_st,
            pp_file=args.pp,
            spacing=float(args.spacing),
            fixed_idx=fixed_idx,
            fmax=float(args.fmax),
            steps=int(args.relax_steps),
            logfile=str(relax_log),
            trajfile=str(relax_traj),
            dftpy_outfile=str(dftpy_out),
        )

        relaxed_xyz = results / f"cycle_{cyc:03d}_relaxed.xyz"
        write(str(relaxed_xyz), atoms_rlx)

        z = atoms_rlx.get_positions()[:, 2]
        zmin = float(z[bottom_idx].max())
        zmax = float(z[top_idx].min())
        L = float(zmax - zmin)
        strain = (L - L0) / L0

        E = _parse_energy_from_out(dftpy_out)
        szz = _parse_stress_zz_from_out(dftpy_out)

        max_nn = _max_nn_distance(atoms_rlx, free_mask)
        with open(summary, "a") as f:
            f.write(f"{cyc},{strain:.10f},{'' if E is None else f'{E:.12f}'},{'' if szz is None else f'{szz:.12f}'},{max_nn:.6f}\n")

        print(f"[summary] cycle={cyc:03d}  strain={strain:.6e}  E={E}  szz={szz}  max_nn={max_nn:.6f}")

        if d0 > 0.0 and max_nn > th:
            print(f">>> FRACTURE: max_nn ({max_nn:.6f}) > 3*d0 ({th:.6f}), stop.")
            break

        atoms = atoms_rlx

    print(f"Done. Results in: {results}")


if __name__ == "__main__":
    main()

