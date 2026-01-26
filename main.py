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


# ---------- Grid Conversion: ecut(eV) -> spacing(Angstrom) ----------
# Nyquist: Gmax ~ pi / h  (h in bohr)
# Ecut(Ha) = 0.5 * Gmax^2 = pi^2 / (2 h^2)
# => h(bohr) = pi / sqrt(2*Ecut(Ha))
# Ecut(Ha) = Ecut(eV) / Ha_to_eV
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
        raise RuntimeError(f"{name} is empty. Check how you generated {name}.npy")
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
    """
    計算夾具邊界 (zb, zt) 與夾具間距 L=zt-zb。

    debug=True 時輸出 Z 分佈圖，協助驗證固定區/自由區在拉伸過程的分佈變化。
    注意：這是「證明拉伸確實發生」的視覺證據，不是 stress-strain。
    """
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

            # 依 z 排序（視覺化用）
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

            # 畫出夾具界面位置
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

            print(f"[plot] Saved: {filepath}")

        except Exception as e:
            print(f"[Warning] Plotting failed: {e}")

    return zb, zt, L


def _free_mask(atoms, bottom_idx, top_idx, axis: int = 2, eps: float = 1e-4) -> np.ndarray:
    z = atoms.get_positions()[:, axis]
    bottom_idx = _ensure_nonempty_idx("bottom_idx", bottom_idx)
    top_idx = _ensure_nonempty_idx("top_idx", top_idx)

    zb = float(z[bottom_idx].max())
    zt = float(z[top_idx].min())
    if zt <= zb:
        raise RuntimeError(f"Bad grips: zb={zb}, zt={zt}")
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
    # 明確清空 constraint（避免 ASE 版本差異造成怪行為）
    a.set_constraint([])
    return a


def _pin_fixed_positions(atoms_target, fixed_idx, ref_positions):
    fixed_idx = np.asarray(fixed_idx, dtype=int).ravel()
    pos = atoms_target.get_positions()
    pos[fixed_idx] = ref_positions
    atoms_target.set_positions(pos)
    return atoms_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--pp", required=True)

    # Choose either spacing or ecut
    ap.add_argument("--spacing", type=float, default=None,
                    help="Real-space grid spacing (Angstrom). If set, ecut is ignored.")
    ap.add_argument("--ecut", type=float, default=None,
                    help="Kinetic energy cutoff (eV). Used only if spacing is not provided.")

    ap.add_argument("--step", type=float, default=0.005)
    ap.add_argument("--cycles", type=int, default=200)
    ap.add_argument("--fmax", type=float, default=0.08)
    ap.add_argument("--relax-steps", type=int, default=80)

    ap.add_argument("--debug-strain", action="store_true")
    ap.add_argument("--gap-tol", type=float, default=1e-6)
    ap.add_argument("--eps-free", type=float, default=1e-4)
    ap.add_argument("--no-pin-grips", action="store_true")

    # Plot controls (proof of stretching)
    ap.add_argument("--plot-grips", action="store_true",
                    help="Save Z-distribution plots (fixed/free/top) into results/.")
    ap.add_argument("--plot-every", type=int, default=20,
                    help="Plot every N cycles (relaxed structure). Default=20. Use 0 to disable periodic plotting.")

    args = ap.parse_args()

    # ---------- Determine spacing ----------
    if args.spacing is None and args.ecut is None:
        ap.error("You must provide either --spacing or --ecut.")
    if args.spacing is not None and args.ecut is not None:
        ap.error("Please provide only one of --spacing or --ecut (not both).")

    if args.spacing is None:
        spacing = ecut_to_spacing_angstrom(float(args.ecut))
        print(f"[grid] Using ecut={float(args.ecut):.6f} eV -> spacing~={spacing:.6f} Å (DFTpy: setting spacing disables ecut)")
    else:
        spacing = float(args.spacing)
        print(f"[grid] Using spacing={spacing:.6f} Å (ecut disabled by design)")

    pin_grips = not bool(args.no_pin_grips)

    workdir = Path(args.workdir).resolve()
    init_path = Path(args.init)
    init_path = init_path.resolve() if init_path.is_absolute() else (workdir / init_path).resolve()

    results = workdir / "results"
    results.mkdir(parents=True, exist_ok=True)

    bottom_idx = np.load(str(workdir / "bottom_idx.npy")).astype(int).ravel()
    top_idx = np.load(str(workdir / "top_idx.npy")).astype(int).ravel()
    bottom_idx = _ensure_nonempty_idx("bottom_idx", bottom_idx)
    top_idx = _ensure_nonempty_idx("top_idx", top_idx)

    fixed_idx = np.unique(np.concatenate([bottom_idx, top_idx])).astype(int)

    atoms0 = read(str(init_path))
    write(str(results / "cycle_000_relaxed.xyz"), atoms0)

    # Plot init once if requested
    if args.plot_grips:
        _grip(atoms0, bottom_idx, top_idx, axis=2, debug=True, tag="cycle_000_init", outdir=results)

    zb0, zt0, L0 = _grip(atoms0, bottom_idx, top_idx, axis=2)
    free0 = _free_mask(atoms0, bottom_idx, top_idx, axis=2, eps=float(args.eps_free))

    max_gap0, d0z, n_nz0 = _gap_stats_z(atoms0, free0, axis=2, tol=float(args.gap_tol))
    if d0z <= 0.0:
        raise RuntimeError(
            f"d0z is zero (no nonzero z-gaps). Try increasing --gap-tol. n_nonzero={n_nz0}"
        )

    # Critical rule: 3x threshold must remain
    th = 3.0 * d0z

    print(
        f"[init] zb={zb0:.6f} zt={zt0:.6f} L0={L0:.6f} "
        f"d0z={d0z:.6f} th=3*d0z={th:.6f} max_gap0={max_gap0:.6f} n_nonzero={n_nz0} "
        f"pin_grips={pin_grips} eps_free={float(args.eps_free):.1e} gap_tol={float(args.gap_tol):.1e}"
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
            print(f"[after strain] zb={zb_s:.6f} zt={zt_s:.6f} L={L_s:.6f} dL={L_s - L_b:.6f}")

        write(str(results / f"cycle_{cyc:03d}_stretched.xyz"), atoms_st)

        relax_log = results / f"cycle_{cyc:03d}_relax.log"
        relax_traj = results / f"cycle_{cyc:03d}_relax.traj"
        dftpy_out = results / f"cycle_{cyc:03d}_dftpy.out"

        fixed_ref = atoms_st.get_positions()[fixed_idx].copy()
        cell_ref = atoms_st.get_cell().copy()

        atoms_rlx, energy, stress = relax_atoms(
            atoms_st,
            pp_file=args.pp,
            spacing=float(spacing),          # Unified control by spacing
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

        # Plot relaxed periodically if requested
        do_plot = False
        if args.plot_grips:
            if int(args.plot_every) > 0 and (cyc % int(args.plot_every) == 0):
                do_plot = True
            if cyc in (1, 2, 5, 10):  # 前期多畫幾張，方便你/老賊確認「真的有拉」
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

        sigma_zz = None
        if stress is not None:
            try:
                sigma_zz = float(stress[2, 2])
            except Exception:
                sigma_zz = None

        with open(summary, "a") as f:
            f.write(
                f"{cyc},{strain:.10f},"
                f"{'' if energy is None else f'{float(energy):.12f}'},"
                f"{'' if sigma_zz is None else f'{float(sigma_zz):.12f}'},"
                f"{max_gap:.6f},"
                f"{L_s:.6f},"
                f"{L_r:.6f},"
                f"{(L_r - L_s):.6f}\n"
            )

        print(
            f"[summary] cycle={cyc:03d} strain={strain:.10f} "
            f"max_gap_z={max_gap:.6f} (d0z={d0z:.6f}, th={th:.6f}, n_nonzero={n_nz}) "
            f"L_st={L_s:.6f} L_rlx={L_r:.6f} dL_relax={L_r - L_s:.6f}"
        )

        if max_gap > th:
            print(f">>> FRACTURE: max_gap_z ({max_gap:.6f}) > 3*d0z ({th:.6f})")
            break

        atoms = _clean(atoms_rlx)

    print(f"Done. Results in: {results}")


if __name__ == "__main__":
    main()

