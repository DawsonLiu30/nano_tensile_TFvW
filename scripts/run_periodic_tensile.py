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


def ecut_to_spacing_angstrom(ecut_ev: float) -> float:
    if ecut_ev <= 0:
        raise ValueError(f"ecut must be positive, got {ecut_ev}")

    ecut_ha = ecut_ev / HA_TO_EV
    h_bohr = math.pi / math.sqrt(2.0 * ecut_ha)
    return h_bohr * BOHR_TO_ANG


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _pick_summary_column(field_names: list[str], preferred: list[str]) -> str | None:
    for key in preferred:
        if key in field_names:
            return key
    return None


def _xy_spans(atoms) -> tuple[float, float]:
    pos = atoms.get_positions()
    dx = float(pos[:, 0].max() - pos[:, 0].min())
    dy = float(pos[:, 1].max() - pos[:, 1].min())
    return dx, dy


def _estimate_cross_section_area_xy_bbox(atoms) -> float:
    dx, dy = _xy_spans(atoms)
    return float(max(dx * dy, 0.0))


def _estimate_cross_section_area_xy_ellipse(atoms) -> float:
    dx, dy = _xy_spans(atoms)
    return float(max(np.pi * dx * dy * 0.25, 0.0))


def _convex_hull_area_xy(atoms) -> float:
    points = np.asarray(atoms.get_positions()[:, :2], dtype=float)
    if len(points) < 3:
        return float("nan")
    points = np.unique(points, axis=0)
    if len(points) < 3:
        return float("nan")
    order = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[order]

    def cross(o, a, b) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    if len(hull) < 3:
        return float("nan")
    x = hull[:, 0]
    y = hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _estimate_cross_section_area_xy_model(atoms, cross_section_shape: str) -> float:
    shape = str(cross_section_shape).strip().lower()
    if shape in {"hexagon", "triangle"}:
        area = _convex_hull_area_xy(atoms)
        if np.isfinite(area) and area > 0.0:
            return float(area)
    return _estimate_cross_section_area_xy_ellipse(atoms)


def _cell_area_xy(atoms) -> float:
    cell = atoms.get_cell().array
    return float(np.linalg.norm(np.cross(cell[0], cell[1])))


def _stress_from_cell_sigma(sigma_zz_gpa: float, cell_area_xy: float, wire_area_xy: float) -> float:
    if not np.isfinite(sigma_zz_gpa) or cell_area_xy <= 0.0 or wire_area_xy <= 0.0:
        return float("nan")
    return float(sigma_zz_gpa * (cell_area_xy / wire_area_xy))


def _collect_stress_metrics(atoms, stress, area_ref_model: float, cross_section_shape: str) -> dict[str, float]:
    area_cur_ellipse = _estimate_cross_section_area_xy_ellipse(atoms)
    area_cur_hull = _convex_hull_area_xy(atoms)
    area_cur_model = _estimate_cross_section_area_xy_model(atoms, cross_section_shape)
    area_cur_bbox = _estimate_cross_section_area_xy_bbox(atoms)
    span_x, span_y = _xy_spans(atoms)
    cell_area_xy = _cell_area_xy(atoms)

    sigma_cell_zz = float(stress[2, 2])
    wire_stress_current = _stress_from_cell_sigma(
        sigma_cell_zz,
        cell_area_xy=cell_area_xy,
        wire_area_xy=area_cur_model,
    )
    wire_stress_ref = _stress_from_cell_sigma(
        sigma_cell_zz,
        cell_area_xy=cell_area_xy,
        wire_area_xy=area_ref_model,
    )
    return {
        "sigma_cell_zz_GPa": sigma_cell_zz,
        "cauchy_wire_zz_GPa": wire_stress_current,
        "nominal_wire_zz_GPa": wire_stress_ref,
        "wire_stress_current_GPa": wire_stress_current,
        "wire_stress_ref_GPa": wire_stress_ref,
        "wire_area_current_model_A2": area_cur_model,
        "wire_area_current_hull_A2": area_cur_hull,
        "wire_area_current_ellipse_A2": area_cur_ellipse,
        "wire_area_current_bbox_A2": area_cur_bbox,
        "wire_span_x_A": span_x,
        "wire_span_y_A": span_y,
        "cell_area_xy_A2": cell_area_xy,
    }


def _read_summary_rows(summary_csv: Path) -> list[dict[str, str]]:
    with open(summary_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_cross_section_shape(workdir: Path) -> str:
    for name in ["paper_periodic_manifest.json", "vacancy_branch_manifest.json"]:
        path = workdir / "inputs" / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        shape = data.get("geometry", {}).get("cross_section_shape")
        if shape:
            return str(shape).strip().lower()
    return "circle"


def _plot_periodic_summary(summary_csv: Path, out_png: Path) -> None:
    data = np.genfromtxt(str(summary_csv), delimiter=",", names=True, dtype=None, encoding="utf-8")
    if getattr(data, "size", 0) == 0:
        print(f"[plot] Skip: no rows in {summary_csv}")
        return

    field_names = list(getattr(data, "dtype", []).names or [])
    strain_key = _pick_summary_column(field_names, ["strain"])
    stress_key = _pick_summary_column(
        field_names,
        ["cauchy_wire_zz_GPa", "wire_stress_current_GPa", "nominal_wire_zz_GPa", "wire_stress_ref_GPa", "sigma_cell_zz_GPa"],
    )
    if strain_key is None or stress_key is None:
        print(f"[plot] Skip: summary lacks usable periodic stress/strain columns in {summary_csv}")
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
    order = np.argsort(strain)
    strain = strain[order]
    stress = stress[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(strain, stress, "-o", markersize=3, linewidth=1.4)
    ax.set_xlabel("Axial engineering strain")
    ax.set_ylabel(f"{stress_key} (GPa)")
    ax.set_title("Periodic Wire Stress-Strain Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=150)
    plt.close(fig)
    print(f"[plot] Wrote: {out_png}")


def _axial_strain(atoms, lz_ref: float, axis: int) -> float:
    lz = float(atoms.get_cell().lengths()[axis])
    return float(lz / lz_ref - 1.0)


def _true_strain(engineering_strain: float) -> float:
    strain = float(engineering_strain)
    if strain <= -1.0:
        return float("nan")
    return float(math.log1p(strain))


def _apply_periodic_strain(atoms, strain_step: float, axis: int):
    strained = atoms.copy()
    cell = strained.get_cell().array.copy()
    cell[int(axis)] *= 1.0 + float(strain_step)
    strained.set_cell(cell, scale_atoms=True)
    return strained


def _append_summary_row(summary: Path, row: dict[str, float]) -> None:
    with open(summary, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                int(row["cycle"]),
                f"{row['strain']:.12f}",
                f"{row.get('true_strain', _true_strain(row['strain'])):.12f}",
                f"{row['energy_eV']:.12f}",
                f"{row['sigma_cell_zz_GPa']:.12f}",
                f"{row.get('cauchy_wire_zz_GPa', row['wire_stress_current_GPa']):.12f}",
                f"{row.get('nominal_wire_zz_GPa', row['wire_stress_ref_GPa']):.12f}",
                f"{row['wire_stress_current_GPa']:.12f}",
                f"{row['wire_stress_ref_GPa']:.12f}",
                f"{row['wire_area_ref_ellipse_A2']:.12f}",
                f"{row.get('wire_area_ref_model_A2', row['wire_area_ref_ellipse_A2']):.12f}",
                f"{row.get('wire_area_current_model_A2', row['wire_area_current_ellipse_A2']):.12f}",
                f"{row.get('wire_area_current_hull_A2', float('nan')):.12f}",
                f"{row['wire_area_current_ellipse_A2']:.12f}",
                f"{row['wire_area_current_bbox_A2']:.12f}",
                f"{row['cell_area_xy_A2']:.12f}",
                f"{row['wire_span_x_A']:.12f}",
                f"{row['wire_span_y_A']:.12f}",
                int(row["n_atoms"]),
            ]
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run quasistatic periodic tensile loading for axially periodic nanocolumn/nanocrystal prisms."
    )
    ap.add_argument("--case", default="", help="Case name for logging.")
    ap.add_argument("--workdir", default=".", help="Case working directory.")
    ap.add_argument("--init", required=True, help="Input equilibrium structure file.")
    ap.add_argument("--pp", default="al.gga.recpot", help="Pseudopotential file path.")
    ap.add_argument("--ecut", type=float, default=1000.0, help="Kinetic energy cutoff (eV).")
    ap.add_argument("--spacing", type=float, default=None, help="Override grid spacing (Angstrom).")
    ap.add_argument("--kedf", default="TFVW", help="DFTpy KEDF name. Examples: TFVW, SM, WT.")
    ap.add_argument("--step", type=float, default=0.01, help="Engineering strain increment per cycle.")
    ap.add_argument("--cycles", type=int, default=20, help="Number of quasistatic strain cycles.")
    ap.add_argument("--fmax", type=float, default=0.002, help="Force convergence for each relax.")
    ap.add_argument("--relax-steps", type=int, default=120, help="Maximum relax steps per cycle.")
    ap.add_argument("--axis", type=int, default=2, choices=[0, 1, 2], help="Axial strain direction.")
    ap.add_argument("--plot-summary", action="store_true", help="Plot periodic stress-strain summary at the end.")
    ap.add_argument("--resume-results", default="", help="Existing results directory to resume into.")
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

    pp_path = Path(args.pp).resolve()
    if not pp_path.exists():
        raise RuntimeError(f"[pp] Not found: {pp_path}")

    kedf_name = normalize_kedf_name(args.kedf)
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
    print(f"[KEDF] {args.kedf} -> {kedf_name}")
    print("========================================")
    cross_section_shape = _load_cross_section_shape(workdir)
    print(f"[GEOMETRY] cross_section_shape={cross_section_shape}")

    resume_results = None
    resume_rows: list[dict[str, str]] = []
    if args.resume_results:
        resume_results = Path(args.resume_results)
        resume_results = resume_results.resolve() if resume_results.is_absolute() else (workdir / resume_results).resolve()
        if not resume_results.exists():
            raise RuntimeError(f"[resume-results] Not found: {resume_results}")
        results = resume_results
        print(f"[RESUME] {results}")
    else:
        run_tag = f"{args.case or 'periodic_case'}_{_ts()}"
        results = workdir / "results" / run_tag
        results.mkdir(parents=True, exist_ok=True)
    print(f"[RESULTS] {results}")

    atoms_eq = None
    energy0 = None
    stress0 = None
    summary = results / "summary.csv"
    if resume_results is not None:
        if not summary.exists():
            raise RuntimeError(f"[resume] summary.csv not found in {results}")
        cycle0_relaxed = results / "cycle_000_relaxed.xyz"
        if not cycle0_relaxed.exists():
            raise RuntimeError(f"[resume] cycle_000_relaxed.xyz not found in {results}")
        atoms_eq = read(str(cycle0_relaxed))
        resume_rows = _read_summary_rows(summary)
        if not resume_rows:
            raise RuntimeError(f"[resume] summary.csv is empty in {results}")
    else:
        atoms_eq = read(str(init_path))
        write(str(results / "cycle_000_input.xyz"), atoms_eq)
        atoms_eq, energy0, stress0 = evaluate_atoms(
            atoms_eq,
            pp_file=str(pp_path),
            spacing=float(spacing),
            kedf=kedf_name,
            dftpy_outfile=str(results / "cycle_000_dftpy.out"),
        )
        write(str(results / "cycle_000_relaxed.xyz"), atoms_eq)

    lz_ref = float(atoms_eq.get_cell().lengths()[int(args.axis)])
    area_ref_ellipse = _estimate_cross_section_area_xy_ellipse(atoms_eq)
    area_ref_model = _estimate_cross_section_area_xy_model(atoms_eq, cross_section_shape)
    area_ref_bbox = _estimate_cross_section_area_xy_bbox(atoms_eq)
    span0_x, span0_y = _xy_spans(atoms_eq)
    cell_area0 = _cell_area_xy(atoms_eq)

    print(
        f"[init] L0={lz_ref:.6f} A area_ref_model={area_ref_model:.6f} A^2 "
        f"area_ref_ellipse={area_ref_ellipse:.6f} A^2 "
        f"area_ref_bbox={area_ref_bbox:.6f} A^2 cell_area_xy={cell_area0:.6f} A^2"
    )
    print(f"[init] spans x/y = {span0_x:.6f} / {span0_y:.6f} A atoms={len(atoms_eq)}")

    start_cycle = 1
    if resume_results is None:
        summary.write_text(
            "cycle,strain,true_strain,energy_eV,sigma_cell_zz_GPa,cauchy_wire_zz_GPa,nominal_wire_zz_GPa,"
            "wire_stress_current_GPa,wire_stress_ref_GPa,"
            "wire_area_ref_ellipse_A2,wire_area_ref_model_A2,wire_area_current_model_A2,wire_area_current_hull_A2,"
            "wire_area_current_ellipse_A2,wire_area_current_bbox_A2,"
            "cell_area_xy_A2,wire_span_x_A,wire_span_y_A,n_atoms\n",
            encoding="utf-8",
        )
        stress_metrics0 = _collect_stress_metrics(
            atoms_eq,
            stress=stress0,
            area_ref_model=area_ref_model,
            cross_section_shape=cross_section_shape,
        )
        _append_summary_row(
            summary,
            {
                "cycle": 0,
                "strain": 0.0,
                "true_strain": 0.0,
                "energy_eV": float(energy0),
                "wire_area_ref_ellipse_A2": area_ref_ellipse,
                "wire_area_ref_model_A2": area_ref_model,
                "n_atoms": int(len(atoms_eq)),
                **stress_metrics0,
            },
        )
    else:
        last_complete = resume_rows[-1]
        start_cycle = int(last_complete["cycle"]) + 1
        cycle_prev = int(last_complete["cycle"])
        prev_relaxed = results / f"cycle_{cycle_prev:03d}_relaxed.xyz"
        if not prev_relaxed.exists():
            raise RuntimeError(f"[resume] missing {prev_relaxed}")
        atoms_eq = read(str(prev_relaxed))
        print(f"[resume] Continuing from cycle {cycle_prev:03d}; next cycle is {start_cycle:03d}")

    for cyc in range(start_cycle, int(args.cycles) + 1):
        atoms_st = _apply_periodic_strain(atoms_eq, strain_step=float(args.step), axis=int(args.axis))
        cycle_tag = f"cycle_{cyc:03d}"
        write(str(results / f"{cycle_tag}_stretched.xyz"), atoms_st)

        atoms_rlx, energy, stress = relax_atoms(
            atoms_st,
            pp_file=str(pp_path),
            spacing=float(spacing),
            fixed_idx=np.array([], dtype=int),
            kedf=kedf_name,
            fmax=float(args.fmax),
            steps=int(args.relax_steps),
            logfile=str(results / f"{cycle_tag}_relax.log"),
            trajfile=str(results / f"{cycle_tag}_relax.traj"),
            dftpy_outfile=str(results / f"{cycle_tag}_dftpy.out"),
            debug_fixed=False,
        )
        write(str(results / f"{cycle_tag}_relaxed.xyz"), atoms_rlx)

        strain = _axial_strain(atoms_rlx, lz_ref=lz_ref, axis=int(args.axis))
        stress_metrics = _collect_stress_metrics(
            atoms_rlx,
            stress=stress,
            area_ref_model=area_ref_model,
            cross_section_shape=cross_section_shape,
        )

        row = {
            "cycle": cyc,
            "strain": strain,
            "true_strain": _true_strain(strain),
            "energy_eV": float(energy),
            "wire_area_ref_ellipse_A2": area_ref_ellipse,
            "wire_area_ref_model_A2": area_ref_model,
            "n_atoms": int(len(atoms_rlx)),
            **stress_metrics,
        }
        _append_summary_row(summary, row)
        print(
            f"[summary] cycle={cyc:03d} strain={strain:.6f} "
            f"sigma_zz={row['sigma_cell_zz_GPa']:+.6f} GPa "
            f"cauchy_wire={row['cauchy_wire_zz_GPa']:+.6f} GPa"
        )

        atoms_eq = atoms_rlx

    if bool(args.plot_summary):
        _plot_periodic_summary(summary, results / "stress_strain.png")

    print(f"Done. Results in: {results}")


if __name__ == "__main__":
    main()
