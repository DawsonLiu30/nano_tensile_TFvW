from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read, write

from strain_engine import apply_strain
from dft_engine import relax_atoms


HA_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903


def ecut_to_spacing_angstrom(ecut_ev: float) -> float:
    if ecut_ev <= 0:
        raise ValueError(f"ecut must be positive, got {ecut_ev}")

    ecut_ha = ecut_ev / HA_TO_EV
    h_bohr = math.pi / math.sqrt(2.0 * ecut_ha)
    return h_bohr * BOHR_TO_ANG


def _ensure_nonempty_idx(name: str, idx: np.ndarray) -> np.ndarray:
    arr = np.asarray(idx, dtype=int).ravel()
    if arr.size == 0:
        raise RuntimeError(f"{name} is empty.")
    return arr


def _ensure_idx_in_range(name: str, idx: np.ndarray, n_atoms: int, idx_path: Path) -> np.ndarray:
    arr = np.asarray(idx, dtype=int).ravel()
    if arr.size == 0:
        raise RuntimeError(f"{name} is empty ({idx_path}).")
    mn = int(arr.min())
    mx = int(arr.max())
    if mn < 0 or mx >= int(n_atoms):
        raise RuntimeError(
            f"{name} out of bounds ({idx_path}): min={mn}, max={mx}, n_atoms={n_atoms}"
        )
    return arr


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _grip(
    atoms,
    bottom_idx,
    top_idx,
    axis: int = 2,
    debug: bool = False,
    tag: str = "",
    outdir: Optional[Path] = None,
) -> Tuple[float, float, float]:
    z = atoms.get_positions()[:, axis]

    bottom_idx = _ensure_nonempty_idx("bottom_idx", bottom_idx)
    top_idx = _ensure_nonempty_idx("top_idx", top_idx)

    zb = float(z[bottom_idx].max())
    zt = float(z[top_idx].min())
    if zt <= zb:
        raise RuntimeError(f"Bad grips: zt ({zt}) <= zb ({zb}).")
    L = float(zt - zb)

    if debug:
        try:
            safe_tag = tag.replace(" ", "_") if tag else "grip"
            filename = f"plot_dist_{safe_tag}_{_ts()}.png"

            if outdir is not None:
                outdir = Path(outdir)
                outdir.mkdir(parents=True, exist_ok=True)
                filepath = outdir / filename
            else:
                filepath = Path(filename)

            n_atoms = z.size
            mask_bottom = np.zeros(n_atoms, dtype=bool)
            mask_top = np.zeros(n_atoms, dtype=bool)
            mask_bottom[bottom_idx] = True
            mask_top[top_idx] = True
            mask_free = ~(mask_bottom | mask_top)

            order = np.argsort(z)
            z_sorted = z[order]
            mb = mask_bottom[order]
            mf = mask_free[order]
            mt = mask_top[order]
            x = np.arange(n_atoms)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(x[mb], z_sorted[mb], s=10, alpha=0.6, label="Fixed (Bottom)")
            ax.scatter(x[mf], z_sorted[mf], s=10, alpha=0.6, label="Free (Middle)")
            ax.scatter(x[mt], z_sorted[mt], s=10, alpha=0.6, label="Fixed (Top)")

            ax.axhline(zb, linestyle="--", linewidth=1, alpha=0.7)
            ax.axhline(zt, linestyle="--", linewidth=1, alpha=0.7)
            ax.text(
                0.02,
                0.98,
                f"zb={zb:.3f} Å\nzt={zt:.3f} Å\nL={L:.3f} Å",
                transform=ax.transAxes,
                va="top",
            )
            ax.set_title(f"Atom Z-Distribution ({safe_tag})")
            ax.set_xlabel("Atoms sorted by Z")
            ax.set_ylabel("Z (Å)")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(str(filepath), dpi=120)
            plt.close(fig)

        except Exception as e:
            print(f"[Warning] Plotting failed: {e}")

    return zb, zt, L


def _free_mask(atoms, bottom_idx, top_idx, axis: int = 2, eps: float = 1e-4) -> np.ndarray:
    z = atoms.get_positions()[:, axis]
    bottom_idx = _ensure_nonempty_idx("bottom_idx", bottom_idx)
    top_idx = _ensure_nonempty_idx("top_idx", top_idx)
    zb = float(z[bottom_idx].max())
    zt = float(z[top_idx].min())
    return (z > zb + eps) & (z < zt - eps)


def _gap_stats_z(atoms, mask: np.ndarray, axis: int = 2, tol: float = 1e-8) -> Tuple[float, float, int]:
    z = atoms.get_positions()[mask, axis]
    if z.size < 2:
        return 0.0, 0.0, 0
    zz = np.sort(z)
    gaps = np.diff(zz)
    nz = gaps[gaps > tol]
    max_gap = float(gaps.max()) if gaps.size else 0.0
    mean_nz = float(nz.mean()) if nz.size else 0.0
    return max_gap, mean_nz, int(nz.size)


def _clean(atoms):
    a = atoms.copy()
    a.calc = None
    a.set_constraint([])
    return a


def _pin_fixed_positions(atoms_target, fixed_idx, ref_positions):
    fixed_idx = np.asarray(fixed_idx, dtype=int).ravel()
    pos = atoms_target.get_positions()
    pos[fixed_idx] = ref_positions
    atoms_target.set_positions(pos)
    return atoms_target


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--case", default="", help="Case name for logging")
    ap.add_argument("--workdir", default=".", help="Case working directory")
    ap.add_argument("--init", default="init.vasp", help="Input structure file")
    ap.add_argument("--pp", default="al.gga.psp", help="Pseudopotential file path")

    ap.add_argument("--bottom-idx", default="bottom_idx.npy", help="Bottom grip indices (.npy)")
    ap.add_argument("--top-idx", default="top_idx.npy", help="Top grip indices (.npy)")

    ap.add_argument("--ecut", type=float, default=400.0, help="Kinetic energy cutoff (eV). Default 400.")
    ap.add_argument("--spacing", type=float, default=None, help="Override grid spacing (Angstrom).")

    ap.add_argument("--step", type=float, default=0.005)
    ap.add_argument("--cycles", type=int, default=200)
    ap.add_argument("--fmax", type=float, default=0.08)
    ap.add_argument("--relax-steps", type=int, default=80)

    ap.add_argument("--debug-strain", action="store_true")
    ap.add_argument("--gap-tol", type=float, default=1e-6)
    ap.add_argument("--eps-free", type=float, default=1e-4)
    ap.add_argument("--no-pin-grips", action="store_true")

    ap.add_argument("--plot-grips", action="store_true")
    ap.add_argument("--plot-every", type=int, default=20)

    return ap.parse_args()


def main():
    args = parse_args()

    pin_grips = not bool(args.no_pin_grips)

    workdir = Path(args.workdir).resolve()
    if not workdir.exists():
        raise RuntimeError(f"[workdir] Not found: {workdir}")

    init_path = Path(args.init)
    init_path = init_path.resolve() if init_path.is_absolute() else (workdir / init_path).resolve()
    if not init_path.exists():
        raise RuntimeError(f"[init] Not found: {init_path}")

    pp_path = Path(args.pp).resolve()
    if not pp_path.exists():
        raise RuntimeError(f"[pp] Not found: {pp_path}")

    if args.spacing is not None:
        spacing = float(args.spacing)
        print(f"[grid] Manual spacing override: {spacing:.6f} Å")
    else:
        spacing = ecut_to_spacing_angstrom(float(args.ecut))
        print(f"[grid] ecut={float(args.ecut):.2f} eV -> spacing={spacing:.6f} Å")

    print("========================================")
    print(f"[CASE] {args.case}")
    print(f"[WORKDIR] {workdir}")
    print(f"[INIT] {init_path}")
    print(f"[PP] {pp_path}")
    print(f"[pin_grips] {pin_grips}")
    print("========================================")

    run_tag = f"{args.case or 'case'}_{_ts()}"
    results = workdir / "results" / run_tag
    results.mkdir(parents=True, exist_ok=True)
    print(f"[RESULTS] {results}")

    bottom_idx_path = Path(args.bottom_idx)
    bottom_idx_path = bottom_idx_path.resolve() if bottom_idx_path.is_absolute() else (workdir / bottom_idx_path).resolve()
    top_idx_path = Path(args.top_idx)
    top_idx_path = top_idx_path.resolve() if top_idx_path.is_absolute() else (workdir / top_idx_path).resolve()

    if not bottom_idx_path.exists():
        raise RuntimeError(f"[bottom-idx] Not found: {bottom_idx_path}")
    if not top_idx_path.exists():
        raise RuntimeError(f"[top-idx] Not found: {top_idx_path}")

    atoms0 = read(str(init_path))
    n_atoms0 = len(atoms0)

    bottom_idx = np.load(str(bottom_idx_path)).astype(int).ravel()
    top_idx = np.load(str(top_idx_path)).astype(int).ravel()

    bottom_idx = _ensure_idx_in_range("bottom_idx", bottom_idx, n_atoms0, bottom_idx_path)
    top_idx = _ensure_idx_in_range("top_idx", top_idx, n_atoms0, top_idx_path)

    bottom_idx = _ensure_nonempty_idx("bottom_idx", bottom_idx)
    top_idx = _ensure_nonempty_idx("top_idx", top_idx)

    fixed_idx = np.unique(np.concatenate([bottom_idx, top_idx])).astype(int)

    write(str(results / "cycle_000_relaxed.xyz"), atoms0)

    if args.plot_grips:
        _grip(atoms0, bottom_idx, top_idx, axis=2, debug=True, tag="cycle_000_init", outdir=results)

    zb0, zt0, L0 = _grip(atoms0, bottom_idx, top_idx, axis=2)
    free0 = _free_mask(atoms0, bottom_idx, top_idx, axis=2, eps=float(args.eps_free))

    max_gap0, d0z, n_nz0 = _gap_stats_z(atoms0, free0, axis=2, tol=float(args.gap_tol))
    if d0z <= 0.0:
        raise RuntimeError("d0z is zero. Try increasing --gap-tol.")

    th = 3.0 * d0z

    print(
        f"[init] zb={zb0:.6f} zt={zt0:.6f} L0={L0:.6f} "
        f"d0z={d0z:.6f} th={th:.6f} max_gap0={max_gap0:.6f} n_nonzero={n_nz0}"
    )

    summary = results / "summary.csv"
    if not summary.exists():
        summary.write_text("cycle,strain,energy_eV,sigma_zz_GPa,max_gap_z_A,L_stretched_A,L_relaxed_A,dL_relax_A\n")

    atoms = _clean(atoms0)

    for cyc in range(1, int(args.cycles) + 1):
        print(f"=== Cycle {cyc:03d} ===")

        zb_b, zt_b, L_b = _grip(atoms, bottom_idx, top_idx, axis=2)
        if args.debug_strain:
            print(f"[before] zb={zb_b:.6f} zt={zt_b:.6f} L={L_b:.6f}")

        atoms_st = _clean(atoms)
        apply_strain(
            atoms_st,
            strain_rate=float(args.step),
            bottom_idx=bottom_idx,
            top_idx=top_idx,
            axis=2,
            debug=bool(args.debug_strain),
        )

        zb_s, zt_s, L_s = _grip(atoms_st, bottom_idx, top_idx, axis=2)
        if args.debug_strain:
            print(f"[after strain] zb={zb_s:.6f} zt={zt_s:.6f} L={L_s:.6f}")

        write(str(results / f"cycle_{cyc:03d}_stretched.xyz"), atoms_st)

        relax_log = results / f"cycle_{cyc:03d}_relax.log"
        relax_traj = results / f"cycle_{cyc:03d}_relax.traj"
        dftpy_out = results / f"cycle_{cyc:03d}_dftpy.out"

        fixed_ref = atoms_st.get_positions()[fixed_idx].copy()
        cell_ref = atoms_st.get_cell().copy()

        atoms_rlx, energy, stress = relax_atoms(
            atoms_st,
            pp_file=str(pp_path),
            spacing=float(spacing),
            fixed_idx=fixed_idx,
            fmax=float(args.fmax),
            steps=int(args.relax_steps),
            logfile=str(relax_log),
            trajfile=str(relax_traj),
            dftpy_outfile=str(dftpy_out),
            debug_fixed=True,
        )

        if pin_grips:
            _pin_fixed_positions(atoms_rlx, fixed_idx, fixed_ref)
            atoms_rlx.set_cell(cell_ref, scale_atoms=False)

        write(str(results / f"cycle_{cyc:03d}_relaxed.xyz"), atoms_rlx)

        do_plot = False
        if args.plot_grips:
            if int(args.plot_every) > 0 and (cyc % int(args.plot_every) == 0):
                do_plot = True
            if cyc in (1, 2, 5, 10):
                do_plot = True

        zb_r, zt_r, L_r = _grip(
            atoms_rlx,
            bottom_idx,
            top_idx,
            axis=2,
            debug=do_plot,
            tag=f"cycle_{cyc:03d}_relaxed",
            outdir=results,
        )

        strain = (L_r - L0) / L0

        free_now = _free_mask(atoms_rlx, bottom_idx, top_idx, axis=2, eps=float(args.eps_free))
        max_gap, mean_nz, n_nz = _gap_stats_z(atoms_rlx, free_now, axis=2, tol=float(args.gap_tol))

        sigma_zz = 0.0
        if stress is not None:
            try:
                sigma_zz = float(stress[2, 2])
            except Exception:
                pass

        with open(summary, "a") as f:
            f.write(
                f"{cyc},{strain:.10f},"
                f"{'' if energy is None else f'{float(energy):.12f}'},"
                f"{f'{float(sigma_zz):.12f}'},"
                f"{max_gap:.6f},{L_s:.6f},{L_r:.6f},{L_r - L_s:.6f}\n"
            )

        print(
            f"[summary] cycle={cyc:03d} strain={strain:.10f} "
            f"max_gap_z={max_gap:.6f} (th={th:.6f}) "
            f"Stress={sigma_zz:.4f} GPa"
        )

        if max_gap > th:
            print(f">>> FRACTURE: max_gap_z ({max_gap:.6f}) > 3*d0z ({th:.6f})")
            break

        atoms = _clean(atoms_rlx)

    print(f"Done. Results in: {results}")


if __name__ == "__main__":
    main()
