from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read, write

from app.strain_engine import apply_strain
from app.dft_engine import relax_atoms
from app.geometry_metrics import projected_xy_geometry


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


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _plot_stress_strain_from_summary(summary_csv: Path, out_png: Path) -> None:
    data = np.genfromtxt(str(summary_csv), delimiter=",", names=True, dtype=None, encoding="utf-8")
    if getattr(data, "size", 0) == 0:
        print(f"[plot] Skip: no rows in {summary_csv}")
        return

    # Handle both single-row and multi-row cases
    field_names = list(getattr(data, "dtype", []).names or [])
    stress_key = "eng_stress_top_GPa" if "eng_stress_top_GPa" in field_names else "sigma_zz_GPa"
    if data.shape == ():
        strain = np.array([float(data["strain"])], dtype=float)
        stress = np.array([float(data[stress_key])], dtype=float)
    else:
        strain = np.asarray(data["strain"], dtype=float)
        stress = np.asarray(data[stress_key], dtype=float)

    valid = np.isfinite(strain) & np.isfinite(stress)
    if not np.any(valid):
        print(f"[plot] Skip: no valid stress/strain data in {summary_csv}")
        return

    strain = strain[valid]
    stress = stress[valid]

    # For tensile tests, some engines report tension as negative sigma_zz.
    # If most points are negative, plot sign-flipped stress for readability.
    if np.mean(stress < 0.0) > 0.5:
        stress_plot = -stress
        stress_label = (
            "Grip-force engineering stress (GPa)"
            if stress_key == "eng_stress_top_GPa"
            else f"-{stress_key} (GPa)"
        )
    else:
        stress_plot = stress
        stress_label = (
            "Grip-force engineering stress (GPa)"
            if stress_key == "eng_stress_top_GPa"
            else f"{stress_key} (GPa)"
        )

    order = np.argsort(strain)
    strain = strain[order]
    stress_plot = stress_plot[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(strain, stress_plot, "-o", markersize=3, linewidth=1.4)
    ax.set_xlabel("Engineering strain")
    ax.set_ylabel(stress_label)
    ax.set_title("Stress-Strain Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=150)
    plt.close(fig)
    print(f"[plot] Wrote: {out_png}")


def _estimate_cross_section_area_xy(atoms) -> tuple[float, dict[str, float], str]:
    geom = projected_xy_geometry(atoms)
    hull_area = float(geom["area_hull_xy_A2"])
    bbox_area = float(geom["area_bbox_xy_A2"])
    if hull_area > 0.0:
        return hull_area, geom, "projected_hull_xy"
    return bbox_area, geom, "projected_bbox_xy"


def _axis_cell_length(atoms, axis: int = 2) -> float:
    cell = np.asarray(atoms.get_cell(), dtype=float)
    return float(np.linalg.norm(cell[axis]))


def _axial_cell_gaps(atoms, axis: int = 2) -> Tuple[float, float, float, float]:
    coord = np.asarray(atoms.get_positions()[:, axis], dtype=float)
    cell_len = _axis_cell_length(atoms, axis=axis)
    lower_gap = float(coord.min())
    upper_gap = float(cell_len - coord.max())
    span = float(coord.max() - coord.min())
    return lower_gap, upper_gap, span, cell_len


def _prepare_finite_axial_cell(atoms, axis: int = 2, axial_vacuum: float = 10.0):
    if float(axial_vacuum) < 0.0:
        raise ValueError(f"axial_vacuum must be >= 0, got {axial_vacuum}")

    lower_gap0, upper_gap0, span, cell_len0 = _axial_cell_gaps(atoms, axis=axis)
    if span <= 0.0:
        raise RuntimeError("Input structure has zero span along the tensile axis.")

    target_cell_len = max(float(cell_len0), float(span + 2.0 * float(axial_vacuum)))
    target_lower_gap = 0.5 * (target_cell_len - span)

    pos = atoms.get_positions().copy()
    shift = target_lower_gap - float(pos[:, axis].min())
    pos[:, axis] += shift
    atoms.set_positions(pos)

    cell = np.asarray(atoms.get_cell(), dtype=float).copy()
    axis_vec = cell[axis].copy()
    axis_norm = float(np.linalg.norm(axis_vec))
    if axis_norm <= 0.0:
        raise RuntimeError(f"Cell axis {axis} has zero length.")
    cell[axis] = axis_vec * (target_cell_len / axis_norm)
    atoms.set_cell(cell, scale_atoms=False)
    atoms.pbc = [False, False, False]

    lower_gap1, upper_gap1, span1, cell_len1 = _axial_cell_gaps(atoms, axis=axis)
    info = {
        "old_lower_gap_A": float(lower_gap0),
        "old_upper_gap_A": float(upper_gap0),
        "old_cell_A": float(cell_len0),
        "new_lower_gap_A": float(lower_gap1),
        "new_upper_gap_A": float(upper_gap1),
        "new_cell_A": float(cell_len1),
        "span_A": float(span1),
        "target_axial_vacuum_A": float(axial_vacuum),
    }
    return atoms, info


def _stress_is_reliable(atoms) -> bool:
    return bool(np.all(np.asarray(atoms.pbc, dtype=bool)))


def _infer_init_state(init_path: Path) -> str:
    name = init_path.name.lower()
    if any(token in name for token in ("restart", "ckpt", "checkpoint")):
        return "checkpoint"
    if any(token in name for token in ("relaxed", "pre_relaxed", "fully_relaxed")):
        return "relaxed"
    return "raw"


def _nearest_nonzero_distance(points, tol: float = 1e-8) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0

    try:
        from scipy.spatial import cKDTree

        dists, _ = cKDTree(pts).query(pts, k=2)
        nn = np.asarray(dists[:, 1], dtype=float)
        nn = nn[np.isfinite(nn) & (nn > tol)]
        if nn.size:
            return float(nn.min())
    except Exception:
        pass

    min_dist = float("inf")
    tol2 = float(tol) ** 2
    for i in range(len(pts) - 1):
        diff = pts[i + 1 :] - pts[i]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        valid = dist2 > tol2
        if not np.any(valid):
            continue
        cand = float(np.sqrt(dist2[valid].min()))
        if cand < min_dist:
            min_dist = cand

    return 0.0 if not np.isfinite(min_dist) else float(min_dist)


def _bond_components(points, cutoff: float) -> list[np.ndarray]:
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n == 0:
        return []
    if cutoff <= 0.0:
        return [np.asarray([i], dtype=int) for i in range(n)]

    parent = np.arange(n, dtype=int)

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri = _find(i)
        rj = _find(j)
        if ri != rj:
            parent[rj] = ri

    try:
        from scipy.spatial import cKDTree

        pairs = cKDTree(pts).query_pairs(r=float(cutoff))
        for i, j in pairs:
            _union(int(i), int(j))
    except Exception:
        tol2 = float(cutoff) ** 2
        for i in range(n - 1):
            diff = pts[i + 1 :] - pts[i]
            dist2 = np.einsum("ij,ij->i", diff, diff)
            near = np.where(dist2 <= tol2)[0]
            for off in near:
                _union(i, i + 1 + int(off))

    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = _find(i)
        groups.setdefault(root, []).append(i)

    return [
        np.asarray(idx, dtype=int)
        for idx in sorted(groups.values(), key=lambda arr: (min(arr), len(arr)))
    ]


def _load_path_state(
    atoms,
    bottom_idx: np.ndarray,
    top_idx: np.ndarray,
    bond_cutoff_A: float,
    axis: int = 2,
    tol: float = 1e-8,
) -> dict[str, float | int | list[float]]:
    components = _bond_components(atoms.get_positions(), cutoff=float(bond_cutoff_A))
    z = np.asarray(atoms.get_positions()[:, axis], dtype=float)
    n_atoms = len(z)
    bottom_mask = np.zeros(n_atoms, dtype=bool)
    top_mask = np.zeros(n_atoms, dtype=bool)
    bottom_mask[np.asarray(bottom_idx, dtype=int).ravel()] = True
    top_mask[np.asarray(top_idx, dtype=int).ravel()] = True

    component_max_gaps: list[float] = []
    n_bridging_paths = 0
    max_gap_bridging = 0.0

    for idx in components:
        idx = np.asarray(idx, dtype=int).ravel()
        if idx.size == 0:
            continue
        if not bool(bottom_mask[idx].any()) or not bool(top_mask[idx].any()):
            continue

        n_bridging_paths += 1
        zz = np.sort(z[idx])
        gaps = np.diff(zz)
        max_gap = float(gaps.max()) if gaps.size else 0.0
        component_max_gaps.append(max_gap)
        max_gap_bridging = max(max_gap_bridging, max_gap)

    return {
        "n_components": int(len(components)),
        "n_bridging_paths": int(n_bridging_paths),
        "max_gap_z_A": float(max_gap_bridging),
        "component_max_gaps_A": [float(v) for v in component_max_gaps],
    }


def _grip_force_stress(atoms, top_idx: np.ndarray, bottom_idx: np.ndarray, area_ang2: float) -> Tuple[float, float, float]:
    forces = atoms.get_forces(apply_constraint=False)  # eV/Angstrom
    f_top_z = float(np.asarray(forces[top_idx, 2], dtype=float).sum())
    f_bot_z = float(np.asarray(forces[bottom_idx, 2], dtype=float).sum())
    if area_ang2 > 0.0:
        eng_stress_top = (f_top_z / area_ang2) * EV_PER_ANG3_TO_GPA
    else:
        eng_stress_top = float("nan")
    return f_top_z, f_bot_z, float(eng_stress_top)


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
    ap.add_argument("--pp", default="al.gga.recpot", help="Reciprocal-space pseudopotential file path")

    ap.add_argument("--bottom-idx", default="bottom_idx.npy", help="Bottom grip indices (.npy)")
    ap.add_argument("--top-idx", default="top_idx.npy", help="Top grip indices (.npy)")

    ap.add_argument("--ecut", type=float, default=1000.0, help="Kinetic energy cutoff (eV). Default 1000.")
    ap.add_argument("--spacing", type=float, default=None, help="Override grid spacing (Angstrom).")

    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--cycles", type=int, default=200)
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--relax-steps", type=int, default=200)
    ap.add_argument("--initial-relax-steps", type=int, default=200)
    ap.add_argument("--skip-initial-relax", action="store_true")
    ap.add_argument(
        "--init-state",
        choices=["auto", "raw", "relaxed", "checkpoint"],
        default="auto",
        help="How to treat --init: raw (new), relaxed (skip init-relax), checkpoint (resume zero-strain relax), or auto.",
    )
    ap.add_argument("--bfgs-maxstep", type=float, default=0.04)
    ap.add_argument("--fire-maxstep", type=float, default=0.04)

    ap.add_argument("--debug-strain", action="store_true")
    ap.add_argument("--gap-tol", type=float, default=1e-6)
    ap.add_argument("--axial-vacuum", type=float, default=10.0)
    ap.add_argument("--fracture-gap-factor", type=float, default=3.0)
    ap.add_argument(
        "--bond-cutoff",
        type=float,
        default=None,
        help="Override the fracture/load-path bond cutoff in Angstrom. Default: fracture-gap-factor * initial nearest-neighbor distance.",
    )
    ap.add_argument("--no-pin-grips", action="store_true")

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

    fixed_idx = np.unique(np.concatenate([bottom_idx, top_idx])).astype(int)
    atoms0, finite_info = _prepare_finite_axial_cell(
        atoms0, axis=2, axial_vacuum=float(args.axial_vacuum)
    )
    print(
        "[finite-cell] "
        f"cell_z {finite_info['old_cell_A']:.6f} -> {finite_info['new_cell_A']:.6f} A | "
        f"lower_gap {finite_info['old_lower_gap_A']:.6f} -> {finite_info['new_lower_gap_A']:.6f} A | "
        f"upper_gap {finite_info['old_upper_gap_A']:.6f} -> {finite_info['new_upper_gap_A']:.6f} A | "
        f"pbc={atoms0.pbc.tolist()}"
    )

    write(str(results / "cycle_000_raw.xyz"), atoms0)

    init_state = str(args.init_state)
    if init_state == "auto":
        init_state = _infer_init_state(init_path)
    print(f"[init-state] {init_state}")

    should_run_initial_relax = not bool(args.skip_initial_relax) and init_state in {"raw", "checkpoint"}
    if not should_run_initial_relax:
        reason = "--skip-initial-relax" if bool(args.skip_initial_relax) else f"init-state={init_state}"
        print(f"[init-relax] skipped ({reason})")
    else:
        init_relax_label = "resuming zero-strain relaxation from checkpoint" if init_state == "checkpoint" else "starting zero-strain relaxation"
        print(
            f"[init-relax] {init_relax_label} "
            f"(steps={int(args.initial_relax_steps)} fmax={float(args.fmax):.4f})"
        )
        init_relax_log = results / "init_relax.log"
        init_relax_traj = results / "init_relax.traj"
        init_dftpy_out = results / "init_dftpy.out"
        atoms0, init_energy, init_stress = relax_atoms(
            atoms0,
            pp_file=str(pp_path),
            spacing=float(spacing),
            fixed_idx=fixed_idx,
            fmax=float(args.fmax),
            steps=int(args.initial_relax_steps),
            bfgs_maxstep=float(args.bfgs_maxstep),
            fire_maxstep=float(args.fire_maxstep),
            logfile=str(init_relax_log),
            trajfile=str(init_relax_traj),
            dftpy_outfile=str(init_dftpy_out),
            debug_fixed=True,
        )
        print(
            "[init-relax] done "
            f"energy_eV={float(init_energy):.6f} "
            f"sigma_zz_GPa={float(init_stress[2, 2]) if init_stress is not None else float('nan'):.6f}"
        )

    write(str(results / "cycle_000_relaxed.xyz"), atoms0)

    if args.plot_grips:
        _grip(atoms0, bottom_idx, top_idx, axis=2, debug=True, tag="cycle_000_init", outdir=results)

    zb0, zt0, L0 = _grip(atoms0, bottom_idx, top_idx, axis=2)
    area0, geom0, area_method = _estimate_cross_section_area_xy(atoms0)
    if area0 <= 0.0:
        raise RuntimeError("Initial projected cross-section area is zero.")

    d_ref = _nearest_nonzero_distance(atoms0.get_positions(), tol=float(args.gap_tol))
    if d_ref <= 0.0:
        raise RuntimeError("Initial nearest-neighbor distance could not be inferred.")
    th = float(args.bond_cutoff) if args.bond_cutoff is not None else float(args.fracture_gap_factor) * d_ref

    load_path0 = _load_path_state(
        atoms0,
        bottom_idx=bottom_idx,
        top_idx=top_idx,
        bond_cutoff_A=th,
        axis=2,
        tol=float(args.gap_tol),
    )
    max_gap0 = float(load_path0["max_gap_z_A"])
    d0z = float(d_ref)
    n_nz0 = int(load_path0["n_bridging_paths"])

    print(
        f"[init] zb={zb0:.6f} zt={zt0:.6f} L0={L0:.6f} "
        f"d_ref={d0z:.6f} th={th:.6f} "
        f"(fracture_gap_factor={float(args.fracture_gap_factor):.3f}) "
        f"max_gap0={max_gap0:.6f} bridging_paths={n_nz0}"
    )
    print(
        f"[init] load-path components={int(load_path0['n_components'])} "
        f"bridging_paths={int(load_path0['n_bridging_paths'])}"
    )
    print(
        f"[init] A0_{area_method}={area0:.6f} A^2 "
        f"(bbox={float(geom0['area_bbox_xy_A2']):.6f} A^2, "
        f"hull={float(geom0['area_hull_xy_A2']):.6f} A^2, "
        f"d_eq={float(geom0['equiv_diameter_hull_nm']):.6f} nm)"
    )

    summary = results / "summary.csv"
    if not summary.exists():
        summary.write_text(
            "cycle,strain,energy_eV,sigma_zz_GPa,eng_stress_top_GPa,f_top_z_eV_per_A,f_bot_z_eV_per_A,"
            "eng_stress_area_A2,max_gap_z_A,n_bridging_paths,n_intact_paths,"
            "L_stretched_A,L_relaxed_A,dL_relax_A\n"
        )

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
            bfgs_maxstep=float(args.bfgs_maxstep),
            fire_maxstep=float(args.fire_maxstep),
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

        load_path_now = _load_path_state(
            atoms_rlx,
            bottom_idx=bottom_idx,
            top_idx=top_idx,
            bond_cutoff_A=th,
            axis=2,
            tol=float(args.gap_tol),
        )
        max_gap = float(load_path_now["max_gap_z_A"])
        n_bridging_paths = int(load_path_now["n_bridging_paths"])
        n_intact_paths = int(load_path_now["n_bridging_paths"])

        sigma_zz = float("nan")
        eng_stress_top = float("nan")
        f_top_z = float("nan")
        f_bot_z = float("nan")
        if stress is not None and _stress_is_reliable(atoms_rlx):
            try:
                sigma_zz = float(stress[2, 2])
            except Exception:
                pass
        if stress is not None:
            try:
                f_top_z, f_bot_z, eng_stress_top = _grip_force_stress(
                    atoms_rlx, top_idx=top_idx, bottom_idx=bottom_idx, area_ang2=area0
                )
            except Exception:
                pass

        with open(summary, "a") as f:
            f.write(
                f"{cyc},{strain:.10f},"
                f"{'' if energy is None else f'{float(energy):.12f}'},"
                f"{'' if not np.isfinite(sigma_zz) else f'{float(sigma_zz):.12f}'},"
                f"{'' if not np.isfinite(eng_stress_top) else f'{float(eng_stress_top):.12f}'},"
                f"{'' if not np.isfinite(f_top_z) else f'{float(f_top_z):.12f}'},"
                f"{'' if not np.isfinite(f_bot_z) else f'{float(f_bot_z):.12f}'},"
                f"{area0:.6f},{max_gap:.6f},{n_bridging_paths},{n_intact_paths},"
                f"{L_s:.6f},{L_r:.6f},{L_r - L_s:.6f}\n"
            )

        print(
            f"[summary] cycle={cyc:03d} strain={strain:.10f} "
            f"max_gap_z={max_gap:.6f} (th={th:.6f}) "
            f"bridging_paths={n_bridging_paths} "
            f"sigma_zz={sigma_zz:.4f} GPa eng_stress_top={eng_stress_top:.4f} GPa"
        )

        if n_bridging_paths == 0:
            print(
                ">>> FRACTURE: "
                f"no grip-to-grip load path remains "
                f"(bond cutoff={th:.6f} A)"
            )
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
