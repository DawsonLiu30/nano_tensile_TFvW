from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read, write

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dft_engine import evaluate_atoms, normalize_kedf_name, relax_atoms


HA_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903
EV_PER_A3_TO_GPA = 160.21766208


def ecut_to_spacing_angstrom(ecut_ev: float) -> float:
    if ecut_ev <= 0:
        raise ValueError(f"ecut must be positive, got {ecut_ev}")
    ecut_ha = ecut_ev / HA_TO_EV
    return math.pi / math.sqrt(2.0 * ecut_ha) * BOHR_TO_ANG


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _xy_spans(atoms) -> tuple[float, float]:
    pos = atoms.get_positions()
    return float(pos[:, 0].max() - pos[:, 0].min()), float(pos[:, 1].max() - pos[:, 1].min())


def _ellipse_area_xy(atoms) -> float:
    span_x, span_y = _xy_spans(atoms)
    return float(np.pi * span_x * span_y * 0.25)


def _cell_area_xy(atoms) -> float:
    cell = atoms.get_cell().array
    return float(np.linalg.norm(np.cross(cell[0], cell[1])))


def _load_metadata(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ["bottom_grip_indices", "top_grip_indices", "grip_distance_ref_A"]:
        if key not in data:
            raise ValueError(f"Missing {key} in {path}")
    return data


def _apply_incremental_grip_strain(atoms, *, step: float, bottom_idx: np.ndarray, top_idx: np.ndarray):
    strained = atoms.copy()
    # Relaxed structures carry FixAtoms constraints from the previous cycle.
    # Clear them before moving the grips, then relax_atoms() reapplies them.
    strained.set_constraint(None)
    pos = strained.get_positions()
    bottom_center = float(pos[bottom_idx, 2].mean())
    top_center = float(pos[top_idx, 2].mean())
    previous_distance = top_center - bottom_center
    if previous_distance <= 0.0:
        raise RuntimeError(f"Invalid grip distance before stretching: {previous_distance:.12f} A")
    center = 0.5 * (bottom_center + top_center)
    scale = 1.0 + float(step)
    new_distance = previous_distance * scale
    half_delta = 0.5 * (new_distance - previous_distance)

    all_idx = np.arange(len(strained), dtype=int)
    fixed_idx = np.unique(np.concatenate([bottom_idx, top_idx])).astype(int)
    free_idx = np.setdiff1d(all_idx, fixed_idx, assume_unique=False)

    # Real tensile grips translate as rigid bodies.  We only affinely stretch
    # the mobile region to provide a reasonable initial guess for relaxation.
    pos[free_idx, 2] = center + (pos[free_idx, 2] - center) * scale
    pos[bottom_idx, 2] -= half_delta
    pos[top_idx, 2] += half_delta
    strained.set_positions(pos)
    measured_distance = _grip_distance(strained, bottom_idx, top_idx)
    expected_distance = new_distance
    tol = max(1e-8, abs(expected_distance) * 1e-8)
    if not np.isfinite(measured_distance) or abs(measured_distance - expected_distance) > tol:
        raise RuntimeError(
            "Grip displacement check failed: "
            f"expected L={expected_distance:.12f} A, got L={measured_distance:.12f} A. "
            "A stale constraint may still be blocking the imposed strain."
        )
    return strained


def _grip_distance(atoms, bottom_idx: np.ndarray, top_idx: np.ndarray) -> float:
    pos = atoms.get_positions()
    return float(pos[top_idx, 2].mean() - pos[bottom_idx, 2].mean())


def _raw_forces(atoms) -> np.ndarray:
    return np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)


def _collect_metrics(atoms, stress, *, bottom_idx: np.ndarray, top_idx: np.ndarray, area_ref: float, l0: float) -> dict[str, float]:
    forces = _raw_forces(atoms)
    top_force_z = float(np.sum(forces[top_idx, 2]))
    bottom_force_z = float(np.sum(forces[bottom_idx, 2]))
    # Reaction stresses are positive in tension when the sample pulls the top grip downward
    # and the bottom grip upward.
    stress_top = -top_force_z * EV_PER_A3_TO_GPA / area_ref
    stress_bottom = bottom_force_z * EV_PER_A3_TO_GPA / area_ref
    stress_avg = 0.5 * (stress_top + stress_bottom)
    span_x, span_y = _xy_spans(atoms)
    return {
        "strain": _grip_distance(atoms, bottom_idx, top_idx) / float(l0) - 1.0,
        "sigma_cell_zz_GPa": float(stress[2, 2]),
        "grip_force_top_z_eVA": top_force_z,
        "grip_force_bottom_z_eVA": bottom_force_z,
        "grip_stress_top_GPa": float(stress_top),
        "grip_stress_bottom_GPa": float(stress_bottom),
        "grip_stress_avg_GPa": float(stress_avg),
        "wire_area_ref_ellipse_A2": float(area_ref),
        "wire_area_current_ellipse_A2": _ellipse_area_xy(atoms),
        "cell_area_xy_A2": _cell_area_xy(atoms),
        "wire_span_x_A": span_x,
        "wire_span_y_A": span_y,
        "grip_distance_A": _grip_distance(atoms, bottom_idx, top_idx),
    }


def _summary_header() -> str:
    return (
        "cycle,strain,energy_eV,sigma_cell_zz_GPa,"
        "grip_force_top_z_eVA,grip_force_bottom_z_eVA,"
        "grip_stress_top_GPa,grip_stress_bottom_GPa,grip_stress_avg_GPa,"
        "wire_area_ref_ellipse_A2,wire_area_current_ellipse_A2,cell_area_xy_A2,"
        "wire_span_x_A,wire_span_y_A,grip_distance_A,n_atoms\n"
    )


def _append_summary_row(path: Path, row: dict[str, float]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                int(row["cycle"]),
                f"{row['strain']:.12f}",
                f"{row['energy_eV']:.12f}",
                f"{row['sigma_cell_zz_GPa']:.12f}",
                f"{row['grip_force_top_z_eVA']:.12f}",
                f"{row['grip_force_bottom_z_eVA']:.12f}",
                f"{row['grip_stress_top_GPa']:.12f}",
                f"{row['grip_stress_bottom_GPa']:.12f}",
                f"{row['grip_stress_avg_GPa']:.12f}",
                f"{row['wire_area_ref_ellipse_A2']:.12f}",
                f"{row['wire_area_current_ellipse_A2']:.12f}",
                f"{row['cell_area_xy_A2']:.12f}",
                f"{row['wire_span_x_A']:.12f}",
                f"{row['wire_span_y_A']:.12f}",
                f"{row['grip_distance_A']:.12f}",
                int(row["n_atoms"]),
            ]
        )


def _read_summary_rows(summary_csv: Path) -> list[dict[str, str]]:
    with open(summary_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _max_z_gap(z_values: np.ndarray) -> float:
    z_sorted = np.sort(np.asarray(z_values, dtype=float).ravel())
    if z_sorted.size < 2:
        return 0.0
    return float(np.max(np.diff(z_sorted)))


def _component_sizes(atoms, *, cutoff_A: float) -> list[int]:
    pos = atoms.get_positions()
    n_atoms = len(pos)
    if n_atoms == 0:
        return []
    parent = list(range(n_atoms))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(i: int, j: int) -> None:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    cutoff = float(cutoff_A)
    for i in range(n_atoms):
        distances = np.linalg.norm(pos[i + 1 :] - pos[i], axis=1)
        for offset in np.where(distances < cutoff)[0]:
            union(i, i + 1 + int(offset))

    groups: dict[int, int] = {}
    for idx in range(n_atoms):
        root = find(idx)
        groups[root] = groups.get(root, 0) + 1
    return sorted(groups.values(), reverse=True)


def _fracture_metrics(atoms, *, fixed_idx: np.ndarray, d0z: float, gap_factor: float) -> dict[str, float | bool | int]:
    z = atoms.get_positions()[:, 2]
    all_idx = np.arange(len(atoms), dtype=int)
    free_idx = np.setdiff1d(all_idx, np.asarray(fixed_idx, dtype=int), assume_unique=False)
    z_free = z[free_idx] if free_idx.size else z
    threshold = float(gap_factor) * float(d0z)
    max_gap_all = _max_z_gap(z)
    max_gap_free = _max_z_gap(z_free)
    component_sizes = _component_sizes(atoms, cutoff_A=3.35)
    largest_component = int(component_sizes[0]) if component_sizes else 0
    second_component = int(component_sizes[1]) if len(component_sizes) > 1 else 0
    # A major second cluster is a robust complete-separation marker; it avoids
    # treating one isolated surface atom as a fully fractured wire.
    major_component_min = max(3, int(round(0.10 * len(atoms))))
    gap_fractured = bool(max_gap_all > threshold)
    component_fractured = bool(second_component >= major_component_min)
    return {
        "max_gap_all_A": max_gap_all,
        "max_gap_free_A": max_gap_free,
        "free_z_span_A": float(np.max(z_free) - np.min(z_free)) if z_free.size else 0.0,
        "d0z_A": float(d0z),
        "gap_factor": float(gap_factor),
        "gap_threshold_A": threshold,
        "n_components": int(len(component_sizes)),
        "largest_component_atoms": largest_component,
        "second_component_atoms": second_component,
        "gap_fractured": gap_fractured,
        "component_fractured": component_fractured,
        "fractured": bool(gap_fractured or component_fractured),
    }


def _fracture_header() -> str:
    return (
        "cycle,strain,max_gap_all_A,max_gap_free_A,free_z_span_A,"
        "d0z_A,gap_factor,gap_threshold_A,n_components,largest_component_atoms,"
        "second_component_atoms,gap_fractured,component_fractured,fractured\n"
    )


def _append_fracture_row(path: Path, *, cycle: int, strain: float, metrics: dict[str, float | bool]) -> None:
    if not path.exists():
        path.write_text(_fracture_header(), encoding="utf-8")
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                int(cycle),
                f"{float(strain):.12f}",
                f"{float(metrics['max_gap_all_A']):.12f}",
                f"{float(metrics['max_gap_free_A']):.12f}",
                f"{float(metrics['free_z_span_A']):.12f}",
                f"{float(metrics['d0z_A']):.12f}",
                f"{float(metrics['gap_factor']):.12f}",
                f"{float(metrics['gap_threshold_A']):.12f}",
                int(metrics["n_components"]),
                int(metrics["largest_component_atoms"]),
                int(metrics["second_component_atoms"]),
                bool(metrics["gap_fractured"]),
                bool(metrics["component_fractured"]),
                bool(metrics["fractured"]),
            ]
        )


def _rebuild_fracture_status(
    *,
    summary_csv: Path,
    results: Path,
    fixed_idx: np.ndarray,
    d0z: float,
    gap_factor: float,
) -> tuple[bool, int | None]:
    rows = _read_summary_rows(summary_csv)
    status_csv = results / "fracture_status.csv"
    status_csv.write_text(_fracture_header(), encoding="utf-8")
    first_fracture: int | None = None
    for row in rows:
        cycle = int(row["cycle"])
        xyz = results / f"cycle_{cycle:03d}_relaxed.xyz"
        if not xyz.exists():
            continue
        atoms = read(str(xyz))
        metrics = _fracture_metrics(atoms, fixed_idx=fixed_idx, d0z=d0z, gap_factor=gap_factor)
        _append_fracture_row(status_csv, cycle=cycle, strain=float(row["strain"]), metrics=metrics)
        if bool(metrics["fractured"]) and first_fracture is None:
            first_fracture = cycle
    return first_fracture is not None, first_fracture


def _plot_summary(summary_csv: Path, out_png: Path) -> None:
    data = np.genfromtxt(str(summary_csv), delimiter=",", names=True, dtype=None, encoding="utf-8")
    if getattr(data, "size", 0) == 0:
        return
    if data.shape == ():
        strain = np.array([float(data["strain"])])
        stress = np.array([float(data["grip_stress_avg_GPa"])])
    else:
        strain = np.asarray(data["strain"], dtype=float)
        stress = np.asarray(data["grip_stress_avg_GPa"], dtype=float)
    valid = np.isfinite(strain) & np.isfinite(stress)
    if not np.any(valid):
        return
    strain = strain[valid] * 100.0
    stress = stress[valid]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(strain, stress, "-o", markersize=3.5, linewidth=1.5)
    if strain.size:
        peak_idx = int(np.nanargmax(stress))
        ax.scatter([strain[peak_idx]], [stress[peak_idx]], s=45, color="#c84c09", zorder=5)
        info = f"Peak = {stress[peak_idx]:.2f} GPa\nstrain = {strain[peak_idx]:.2f}%"
        ax.text(
            0.03,
            0.97,
            info,
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.65", "alpha": 0.9},
        )
    ax.axhline(0.0, color="0.55", linewidth=0.8)
    ax.set_xlabel("Engineering strain from grip separation (%)")
    ax.set_ylabel("Grip reaction stress (GPa)")
    ax.set_title("Finite-Grip Vacancy Nanowire Tensile Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=150)
    plt.close(fig)
    print(f"[plot] Wrote: {out_png}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run displacement-controlled finite-grip tensile loading.")
    ap.add_argument("--case", default="")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--init", required=True)
    ap.add_argument("--metadata", default="inputs/grip_metadata.json")
    ap.add_argument("--pp", default="al.gga.recpot")
    ap.add_argument("--kedf", default="TFVW")
    ap.add_argument("--ecut", type=float, default=1000.0)
    ap.add_argument("--spacing", type=float, default=None)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--relax-steps", type=int, default=80)
    ap.add_argument("--plot-summary", action="store_true")
    ap.add_argument("--resume-results", default="")
    ap.add_argument("--a0", type=float, default=4.118877004246)
    ap.add_argument("--fracture-d0z", type=float, default=None, help="Axial reference spacing for fracture. Default: a0/sqrt(3).")
    ap.add_argument("--fracture-gap-factor", type=float, default=3.0)
    ap.add_argument("--disable-fracture-stop", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    if not workdir.exists():
        raise RuntimeError(f"[workdir] Not found: {workdir}")
    init_path = Path(args.init)
    init_path = init_path.resolve() if init_path.is_absolute() else (workdir / init_path).resolve()
    if not init_path.exists():
        raise RuntimeError(f"[init] Not found: {init_path}")
    metadata_path = Path(args.metadata)
    metadata_path = metadata_path.resolve() if metadata_path.is_absolute() else (workdir / metadata_path).resolve()
    metadata = _load_metadata(metadata_path)
    pp_path = Path(args.pp).resolve()
    if not pp_path.exists():
        raise RuntimeError(f"[pp] Not found: {pp_path}")

    bottom_idx = np.asarray(metadata["bottom_grip_indices"], dtype=int)
    top_idx = np.asarray(metadata["top_grip_indices"], dtype=int)
    fixed_idx = np.unique(np.concatenate([bottom_idx, top_idx])).astype(int)
    l0 = float(metadata["grip_distance_ref_A"])
    area_ref = float(metadata.get("area_ref_ellipse_A2", 0.0))
    if l0 <= 0.0 or area_ref <= 0.0:
        raise ValueError("Invalid reference grip distance or area in metadata.")
    d0z = float(args.fracture_d0z) if args.fracture_d0z is not None else float(args.a0) / math.sqrt(3.0)
    gap_factor = float(args.fracture_gap_factor)
    if d0z <= 0.0 or gap_factor <= 0.0:
        raise ValueError("fracture d0z and gap factor must be positive.")

    kedf_name = normalize_kedf_name(args.kedf)
    spacing = float(args.spacing) if args.spacing is not None else ecut_to_spacing_angstrom(float(args.ecut))
    print("========================================")
    print(f"[CASE] {args.case}")
    print(f"[WORKDIR] {workdir}")
    print(f"[INIT] {init_path}")
    print(f"[METADATA] {metadata_path}")
    print(f"[KEDF] {args.kedf} -> {kedf_name}")
    print(f"[grid] spacing={spacing:.6f} A")
    print(
        f"[fracture] enabled={not bool(args.disable_fracture_stop)} "
        f"gap=max_atomic_z_gap>{gap_factor:.3f}*d111 "
        f"(d111={d0z:.6f} A, threshold={gap_factor * d0z:.6f} A); "
        "also stops if a major disconnected cluster appears"
    )
    print("========================================")

    if args.resume_results:
        results = Path(args.resume_results)
        results = results.resolve() if results.is_absolute() else (workdir / results).resolve()
        if not results.exists():
            raise RuntimeError(f"[resume-results] Not found: {results}")
        print(f"[RESUME] {results}")
    else:
        run_tag = f"{args.case or 'grip_tensile'}_{_ts()}"
        results = workdir / "results" / run_tag
        results.mkdir(parents=True, exist_ok=True)
    print(f"[RESULTS] {results}")

    summary = results / "summary.csv"
    if args.resume_results:
        rows = _read_summary_rows(summary)
        if not rows:
            raise RuntimeError(f"[resume] summary.csv is empty: {summary}")
        already_fractured, first_fracture = _rebuild_fracture_status(
            summary_csv=summary,
            results=results,
            fixed_idx=fixed_idx,
            d0z=d0z,
            gap_factor=gap_factor,
        )
        if already_fractured and not bool(args.disable_fracture_stop):
            print(f"[fracture] Existing result already fractured at cycle {first_fracture}; nothing to resume.")
            if bool(args.plot_summary):
                _plot_summary(summary, results / "stress_strain.png")
            return
        last_cycle = int(rows[-1]["cycle"])
        atoms_eq = read(str(results / f"cycle_{last_cycle:03d}_relaxed.xyz"))
        start_cycle = last_cycle + 1
        print(f"[resume] Continuing from cycle {last_cycle:03d}; next cycle is {start_cycle:03d}")
    else:
        atoms_eq = read(str(init_path))
        write(str(results / "cycle_000_input.xyz"), atoms_eq)
        atoms_eq, energy0, stress0 = evaluate_atoms(
            atoms_eq,
            pp_file=str(pp_path),
            spacing=spacing,
            kedf=kedf_name,
            dftpy_outfile=str(results / "cycle_000_dftpy.out"),
        )
        write(str(results / "cycle_000_relaxed.xyz"), atoms_eq)
        summary.write_text(_summary_header(), encoding="utf-8")
        metrics0 = _collect_metrics(atoms_eq, stress0, bottom_idx=bottom_idx, top_idx=top_idx, area_ref=area_ref, l0=l0)
        _append_summary_row(
            summary,
            {"cycle": 0, "energy_eV": float(energy0), "n_atoms": int(len(atoms_eq)), **metrics0},
        )
        fracture0 = _fracture_metrics(atoms_eq, fixed_idx=fixed_idx, d0z=d0z, gap_factor=gap_factor)
        _append_fracture_row(results / "fracture_status.csv", cycle=0, strain=float(metrics0["strain"]), metrics=fracture0)
        if bool(fracture0["fractured"]) and not bool(args.disable_fracture_stop):
            print(
                f">>> FRACTURE: cycle=000 max_gap_all="
                f"{float(fracture0['max_gap_all_A']):.6f} A, "
                f"threshold={float(fracture0['gap_threshold_A']):.6f} A, "
                f"components={int(fracture0['n_components'])} "
                f"(largest/second={int(fracture0['largest_component_atoms'])}/"
                f"{int(fracture0['second_component_atoms'])})"
            )
            if bool(args.plot_summary):
                _plot_summary(summary, results / "stress_strain.png")
            print(f"Done. Results in: {results}")
            return
        start_cycle = 1

    print(f"[init] grip L0={l0:.6f} A area_ref={area_ref:.6f} A^2 atoms={len(atoms_eq)}")
    print(f"[init] grip atoms bottom/top={bottom_idx.size}/{top_idx.size}")

    for cyc in range(start_cycle, int(args.cycles) + 1):
        cycle_tag = f"cycle_{cyc:03d}"
        atoms_st = _apply_incremental_grip_strain(
            atoms_eq,
            step=float(args.step),
            bottom_idx=bottom_idx,
            top_idx=top_idx,
        )
        write(str(results / f"{cycle_tag}_stretched.xyz"), atoms_st)
        atoms_rlx, energy, stress = relax_atoms(
            atoms_st,
            pp_file=str(pp_path),
            spacing=spacing,
            fixed_idx=fixed_idx,
            kedf=kedf_name,
            fmax=float(args.fmax),
            steps=int(args.relax_steps),
            logfile=str(results / f"{cycle_tag}_relax.log"),
            trajfile=str(results / f"{cycle_tag}_relax.traj"),
            dftpy_outfile=str(results / f"{cycle_tag}_dftpy.out"),
            debug_fixed=True,
        )
        write(str(results / f"{cycle_tag}_relaxed.xyz"), atoms_rlx)
        metrics = _collect_metrics(atoms_rlx, stress, bottom_idx=bottom_idx, top_idx=top_idx, area_ref=area_ref, l0=l0)
        imposed_distance = _grip_distance(atoms_st, bottom_idx, top_idx)
        relaxed_distance = float(metrics["grip_distance_A"])
        if abs(relaxed_distance - imposed_distance) > 1e-6:
            raise RuntimeError(
                "Fixed-grip check failed after relaxation: "
                f"imposed L={imposed_distance:.12f} A, relaxed L={relaxed_distance:.12f} A."
            )
        row = {"cycle": cyc, "energy_eV": float(energy), "n_atoms": int(len(atoms_rlx)), **metrics}
        _append_summary_row(summary, row)
        fracture = _fracture_metrics(atoms_rlx, fixed_idx=fixed_idx, d0z=d0z, gap_factor=gap_factor)
        _append_fracture_row(results / "fracture_status.csv", cycle=cyc, strain=float(row["strain"]), metrics=fracture)
        print(
            f"[summary] cycle={cyc:03d} strain={row['strain']:.6f} "
            f"grip_stress_avg={row['grip_stress_avg_GPa']:+.6f} GPa "
            f"top/bottom={row['grip_stress_top_GPa']:+.6f}/{row['grip_stress_bottom_GPa']:+.6f} GPa "
            f"max_gap_all={float(fracture['max_gap_all_A']):.6f} A "
            f"free_gap={float(fracture['max_gap_free_A']):.6f} A "
            f"(th={float(fracture['gap_threshold_A']):.6f} A) "
            f"components={int(fracture['n_components'])} "
            f"largest/second={int(fracture['largest_component_atoms'])}/"
            f"{int(fracture['second_component_atoms'])}"
        )
        atoms_eq = atoms_rlx
        if bool(fracture["fractured"]) and not bool(args.disable_fracture_stop):
            print(
                f">>> FRACTURE: cycle={cyc:03d} max_gap_all="
                f"{float(fracture['max_gap_all_A']):.6f} A, "
                f"threshold={float(fracture['gap_threshold_A']):.6f} A, "
                f"components={int(fracture['n_components'])} "
                f"(largest/second={int(fracture['largest_component_atoms'])}/"
                f"{int(fracture['second_component_atoms'])})"
            )
            break

    if bool(args.plot_summary):
        _plot_summary(summary, results / "stress_strain.png")

    print(f"Done. Results in: {results}")


if __name__ == "__main__":
    main()
