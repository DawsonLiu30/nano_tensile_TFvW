from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read, write

from app.dft_engine import relax_atoms
from app.strain_engine import apply_strain


HA_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903
EV_PER_ANG3_TO_GPA = 160.21766208


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


def _ensure_balanced_grips(bottom_idx: np.ndarray, top_idx: np.ndarray) -> None:
    if bottom_idx.size != top_idx.size:
        raise RuntimeError(
            "Bottom/top grip atom counts differ, which would invalidate the finite-wire setup: "
            f"bottom={bottom_idx.size}, top={top_idx.size}"
        )


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _pick_summary_column(field_names: list[str], preferred: list[str]) -> Optional[str]:
    for key in preferred:
        if key in field_names:
            return key
    return None


def _plot_stress_strain_from_summary(summary_csv: Path, out_png: Path) -> None:
    data = np.genfromtxt(str(summary_csv), delimiter=",", names=True, dtype=None, encoding="utf-8")
    if getattr(data, "size", 0) == 0:
        print(f"[plot] Skip: no rows in {summary_csv}")
        return

    field_names = list(getattr(data, "dtype", []).names or [])
    strain_key = _pick_summary_column(field_names, ["free_region_strain", "strain"])
    stress_key = _pick_summary_column(
        field_names,
        [
            "wire_stress_free_current_GPa",
            "eng_stress_top_GPa",
            "wire_stress_free_ref_GPa",
            "sigma_zz_GPa",
        ],
    )
    if strain_key is None or stress_key is None:
        print(f"[plot] Skip: summary lacks usable strain/stress columns in {summary_csv}")
        return

    if data.shape == ():
        strain = np.array([float(data[strain_key])], dtype=float)
        stress = np.array([float(data[stress_key])], dtype=float)
    else:
        strain = np.asarray(data[strain_key], dtype=float)
        stress = np.asarray(data[stress_key], dtype=float)

    valid = np.isfinite(strain) & np.isfinite(stress)
    if not np.any(valid):
        print(f"[plot] Skip: no valid stress/strain data in {summary_csv}")
        return

    strain = strain[valid]
    stress = stress[valid]

    if np.mean(stress < 0.0) > 0.5:
        stress_plot = -stress
        stress_label = f"-{stress_key} (GPa)"
    else:
        stress_plot = stress
        stress_label = f"{stress_key} (GPa)"

    order = np.argsort(strain)
    strain = strain[order]
    stress_plot = stress_plot[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(strain, stress_plot, "-o", markersize=3, linewidth=1.4)
    ax.set_xlabel("Free-region engineering strain")
    ax.set_ylabel(stress_label)
    ax.set_title("Stress-Strain Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=150)
    plt.close(fig)
    print(f"[plot] Wrote: {out_png}")


def _xy_spans(atoms, mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    pos = atoms.get_positions()
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).ravel()
        if mask.size != len(pos):
            raise ValueError("mask length does not match atoms length.")
        if np.any(mask):
            pos = pos[mask]

    dx = float(pos[:, 0].max() - pos[:, 0].min())
    dy = float(pos[:, 1].max() - pos[:, 1].min())
    return dx, dy


def _estimate_cross_section_area_xy_bbox(atoms, mask: Optional[np.ndarray] = None) -> float:
    dx, dy = _xy_spans(atoms, mask=mask)
    return float(max(dx * dy, 0.0))


def _estimate_cross_section_area_xy_ellipse(atoms, mask: Optional[np.ndarray] = None) -> float:
    dx, dy = _xy_spans(atoms, mask=mask)
    return float(max(np.pi * dx * dy * 0.25, 0.0))


def _cell_area_xy(atoms) -> float:
    cell = atoms.get_cell().array
    return float(np.linalg.norm(np.cross(cell[0], cell[1])))


def _stress_from_cell_sigma(sigma_zz_gpa: float, cell_area_xy: float, wire_area_xy: float) -> float:
    if not np.isfinite(sigma_zz_gpa) or cell_area_xy <= 0.0 or wire_area_xy <= 0.0:
        return float("nan")
    return float(sigma_zz_gpa * (cell_area_xy / wire_area_xy))


def _grip_force_stress(
    atoms,
    top_idx: np.ndarray,
    bottom_idx: np.ndarray,
    area_ang2: float,
) -> Tuple[float, float, float]:
    forces = atoms.get_forces(apply_constraint=False)
    f_top_z = float(np.asarray(forces[top_idx, 2], dtype=float).sum())
    f_bot_z = float(np.asarray(forces[bottom_idx, 2], dtype=float).sum())
    if area_ang2 > 0.0:
        grip_force_stress = (f_top_z / area_ang2) * EV_PER_ANG3_TO_GPA
    else:
        grip_force_stress = float("nan")
    return f_top_z, f_bot_z, float(grip_force_stress)


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
                f"zb={zb:.3f} A\nzt={zt:.3f} A\nL={L:.3f} A",
                transform=ax.transAxes,
                va="top",
            )
            ax.set_title(f"Atom Z-Distribution ({safe_tag})")
            ax.set_xlabel("Atoms sorted by Z")
            ax.set_ylabel("Z (A)")
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


def _format_optional(value: float | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return f"{float(value):.12f}"


def _collect_stress_metrics(
    atoms,
    stress,
    top_idx: np.ndarray,
    bottom_idx: np.ndarray,
    free_mask: np.ndarray,
    free_area_ref_ellipse: float,
) -> dict:
    free_area_cur_ellipse = _estimate_cross_section_area_xy_ellipse(atoms, mask=free_mask)
    free_area_cur_bbox = _estimate_cross_section_area_xy_bbox(atoms, mask=free_mask)
    free_span_x, free_span_y = _xy_spans(atoms, mask=free_mask)
    cell_area_xy = _cell_area_xy(atoms)

    sigma_cell_zz = float("nan")
    wire_stress_current = float("nan")
    wire_stress_ref = float("nan")
    if stress is not None:
        try:
            sigma_cell_zz = float(stress[2, 2])
        except Exception:
            sigma_cell_zz = float("nan")

        wire_stress_current = _stress_from_cell_sigma(
            sigma_cell_zz,
            cell_area_xy=cell_area_xy,
            wire_area_xy=free_area_cur_ellipse,
        )
        wire_stress_ref = _stress_from_cell_sigma(
            sigma_cell_zz,
            cell_area_xy=cell_area_xy,
            wire_area_xy=free_area_ref_ellipse,
        )

    f_top_z = float("nan")
    f_bot_z = float("nan")
    grip_force_stress = float("nan")
    try:
        f_top_z, f_bot_z, grip_force_stress = _grip_force_stress(
            atoms,
            top_idx=top_idx,
            bottom_idx=bottom_idx,
            area_ang2=free_area_ref_ellipse,
        )
    except Exception:
        pass

    preferred_stress = wire_stress_current if np.isfinite(wire_stress_current) else grip_force_stress

    return {
        "sigma_cell_zz_GPa": sigma_cell_zz,
        "wire_stress_free_current_GPa": wire_stress_current,
        "wire_stress_free_ref_GPa": wire_stress_ref,
        "eng_stress_top_GPa": preferred_stress,
        "grip_force_stress_top_GPa": grip_force_stress,
        "f_top_z_eV_per_A": f_top_z,
        "f_bot_z_eV_per_A": f_bot_z,
        "free_area_current_ellipse_A2": free_area_cur_ellipse,
        "free_area_current_bbox_A2": free_area_cur_bbox,
        "free_span_x_A": free_span_x,
        "free_span_y_A": free_span_y,
        "cell_area_xy_A2": cell_area_xy,
    }


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
    ap.add_argument("--skip-initial-relax", action="store_true")

    ap.add_argument("--plot-grips", action="store_true")
    ap.add_argument("--plot-every", type=int, default=20)
    ap.add_argument("--plot-summary", action="store_true", help="Plot stress-strain curve at the end.")

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
        print(f"[grid] Manual spacing override: {spacing:.6f} A")
    else:
        spacing = ecut_to_spacing_angstrom(float(args.ecut))
        print(f"[grid] ecut={float(args.ecut):.2f} eV -> spacing={spacing:.6f} A")

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
    _ensure_balanced_grips(bottom_idx, top_idx)

    fixed_idx = np.unique(np.concatenate([bottom_idx, top_idx])).astype(int)
    write(str(results / "cycle_000_input.xyz"), atoms0)

    if args.plot_grips:
        _grip(atoms0, bottom_idx, top_idx, axis=2, debug=True, tag="cycle_000_input", outdir=results)

    atoms_eq = _clean(atoms0)
    energy0 = None
    stress0 = None
    if not bool(args.skip_initial_relax):
        relax_log0 = results / "cycle_000_relax.log"
        relax_traj0 = results / "cycle_000_relax.traj"
        dftpy_out0 = results / "cycle_000_dftpy.out"

        fixed_ref0 = atoms_eq.get_positions()[fixed_idx].copy()
        cell_ref0 = atoms_eq.get_cell().copy()

        atoms_eq, energy0, stress0 = relax_atoms(
            atoms_eq,
            pp_file=str(pp_path),
            spacing=float(spacing),
            fixed_idx=fixed_idx,
            fmax=float(args.fmax),
            steps=int(args.relax_steps),
            logfile=str(relax_log0),
            trajfile=str(relax_traj0),
            dftpy_outfile=str(dftpy_out0),
            debug_fixed=True,
        )

        if pin_grips:
            _pin_fixed_positions(atoms_eq, fixed_idx, fixed_ref0)
            atoms_eq.set_cell(cell_ref0, scale_atoms=False)
    else:
        print("[init] Skipping initial relaxation; using input structure as the reference state.")

    write(str(results / "cycle_000_relaxed.xyz"), atoms_eq)

    zb0, zt0, L0 = _grip(atoms_eq, bottom_idx, top_idx, axis=2)
    free0 = _free_mask(atoms_eq, bottom_idx, top_idx, axis=2, eps=float(args.eps_free))
    if not np.any(free0):
        raise RuntimeError("Free-region mask is empty. Check grip definitions.")

    max_gap0, d0z, n_nz0 = _gap_stats_z(atoms_eq, free0, axis=2, tol=float(args.gap_tol))
    if d0z <= 0.0:
        raise RuntimeError("d0z is zero. Try increasing --gap-tol.")

    th = 3.0 * d0z

    free_area_ref_bbox = _estimate_cross_section_area_xy_bbox(atoms_eq, mask=free0)
    free_area_ref_ellipse = _estimate_cross_section_area_xy_ellipse(atoms_eq, mask=free0)
    free_span0_x, free_span0_y = _xy_spans(atoms_eq, mask=free0)
    cell_area0 = _cell_area_xy(atoms_eq)

    print(
        f"[init] bottom_atoms={bottom_idx.size} top_atoms={top_idx.size} free_atoms={int(np.sum(free0))}"
    )
    print(
        f"[init] zb={zb0:.6f} zt={zt0:.6f} free_L0={L0:.6f} "
        f"d0z={d0z:.6f} th={th:.6f} max_gap0={max_gap0:.6f} n_nonzero={n_nz0}"
    )
    print(
        f"[init] free_area_ref_ellipse={free_area_ref_ellipse:.6f} A^2 "
        f"free_area_ref_bbox={free_area_ref_bbox:.6f} A^2 "
        f"cell_area_xy={cell_area0:.6f} A^2"
    )
    print(f"[init] free spans x/y = {free_span0_x:.6f} / {free_span0_y:.6f} A")

    summary = results / "summary.csv"
    summary.write_text(
        "cycle,strain,free_region_strain,energy_eV,sigma_zz_GPa,eng_stress_top_GPa,"
        "wire_stress_free_current_GPa,wire_stress_free_ref_GPa,grip_force_stress_top_GPa,"
        "f_top_z_eV_per_A,f_bot_z_eV_per_A,max_gap_z_A,L_stretched_A,L_relaxed_A,dL_relax_A,"
        "free_area_ref_ellipse_A2,free_area_current_ellipse_A2,free_area_current_bbox_A2,"
        "cell_area_xy_A2,free_span_x_A,free_span_y_A,n_free_atoms\n",
        encoding="utf-8",
    )

    stress_metrics0 = _collect_stress_metrics(
        atoms_eq,
        stress=stress0,
        top_idx=top_idx,
        bottom_idx=bottom_idx,
        free_mask=free0,
        free_area_ref_ellipse=free_area_ref_ellipse,
    )

    with open(summary, "a", encoding="utf-8") as f:
        f.write(
            ",".join(
                [
                    "0",
                    "0.0000000000",
                    "0.0000000000",
                    _format_optional(energy0),
                    _format_optional(stress_metrics0["sigma_cell_zz_GPa"]),
                    _format_optional(stress_metrics0["eng_stress_top_GPa"]),
                    _format_optional(stress_metrics0["wire_stress_free_current_GPa"]),
                    _format_optional(stress_metrics0["wire_stress_free_ref_GPa"]),
                    _format_optional(stress_metrics0["grip_force_stress_top_GPa"]),
                    _format_optional(stress_metrics0["f_top_z_eV_per_A"]),
                    _format_optional(stress_metrics0["f_bot_z_eV_per_A"]),
                    f"{max_gap0:.6f}",
                    f"{L0:.6f}",
                    f"{L0:.6f}",
                    "0.000000",
                    f"{free_area_ref_ellipse:.6f}",
                    f"{stress_metrics0['free_area_current_ellipse_A2']:.6f}",
                    f"{stress_metrics0['free_area_current_bbox_A2']:.6f}",
                    f"{stress_metrics0['cell_area_xy_A2']:.6f}",
                    f"{stress_metrics0['free_span_x_A']:.6f}",
                    f"{stress_metrics0['free_span_y_A']:.6f}",
                    str(int(np.sum(free0))),
                ]
            )
            + "\n"
        )

    atoms = _clean(atoms_eq)

    for cyc in range(1, int(args.cycles) + 1):
        print(f"=== Cycle {cyc:03d} ===")

        zb_b, zt_b, L_b = _grip(atoms, bottom_idx, top_idx, axis=2)
        if args.debug_strain:
            print(f"[before] zb={zb_b:.6f} zt={zt_b:.6f} free_L={L_b:.6f}")

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
            print(f"[after strain] zb={zb_s:.6f} zt={zt_s:.6f} free_L={L_s:.6f}")

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

        free_region_strain = (L_r - L0) / L0
        strain = free_region_strain

        free_now = _free_mask(atoms_rlx, bottom_idx, top_idx, axis=2, eps=float(args.eps_free))
        max_gap, _, _ = _gap_stats_z(atoms_rlx, free_now, axis=2, tol=float(args.gap_tol))
        stress_metrics = _collect_stress_metrics(
            atoms_rlx,
            stress=stress,
            top_idx=top_idx,
            bottom_idx=bottom_idx,
            free_mask=free_now,
            free_area_ref_ellipse=free_area_ref_ellipse,
        )

        with open(summary, "a", encoding="utf-8") as f:
            f.write(
                ",".join(
                    [
                        str(cyc),
                        f"{strain:.10f}",
                        f"{free_region_strain:.10f}",
                        _format_optional(energy),
                        _format_optional(stress_metrics["sigma_cell_zz_GPa"]),
                        _format_optional(stress_metrics["eng_stress_top_GPa"]),
                        _format_optional(stress_metrics["wire_stress_free_current_GPa"]),
                        _format_optional(stress_metrics["wire_stress_free_ref_GPa"]),
                        _format_optional(stress_metrics["grip_force_stress_top_GPa"]),
                        _format_optional(stress_metrics["f_top_z_eV_per_A"]),
                        _format_optional(stress_metrics["f_bot_z_eV_per_A"]),
                        f"{max_gap:.6f}",
                        f"{L_s:.6f}",
                        f"{L_r:.6f}",
                        f"{L_r - L_s:.6f}",
                        f"{free_area_ref_ellipse:.6f}",
                        f"{stress_metrics['free_area_current_ellipse_A2']:.6f}",
                        f"{stress_metrics['free_area_current_bbox_A2']:.6f}",
                        f"{stress_metrics['cell_area_xy_A2']:.6f}",
                        f"{stress_metrics['free_span_x_A']:.6f}",
                        f"{stress_metrics['free_span_y_A']:.6f}",
                        str(int(np.sum(free_now))),
                    ]
                )
                + "\n"
            )

        print(
            f"[summary] cycle={cyc:03d} free_strain={free_region_strain:.10f} "
            f"max_gap_z={max_gap:.6f} (th={th:.6f}) "
            f"sigma_cell_zz={stress_metrics['sigma_cell_zz_GPa']:.4f} GPa "
            f"wire_stress_free={stress_metrics['wire_stress_free_current_GPa']:.4f} GPa"
        )

        if max_gap > th:
            print(f">>> FRACTURE: max_gap_z ({max_gap:.6f}) > 3*d0z ({th:.6f})")
            break

        atoms = _clean(atoms_rlx)

    if bool(args.plot_summary):
        try:
            _plot_stress_strain_from_summary(summary, results / "stress_strain.png")
        except Exception as e:
            print(f"[Warning] Failed to plot stress-strain: {e}")

    print(f"Done. Results in: {results}")


if __name__ == "__main__":
    main()
