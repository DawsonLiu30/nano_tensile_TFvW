from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from ase.io import read, write

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ase_nanocrystal import build_circular_nanowire
from app.dft_engine import normalize_kedf_name, relax_atoms


HA_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903


def ecut_to_spacing_angstrom(ecut_ev: float) -> float:
    if ecut_ev <= 0:
        raise ValueError(f"ecut must be positive, got {ecut_ev}")
    ecut_ha = ecut_ev / HA_TO_EV
    return math.pi / math.sqrt(2.0 * ecut_ha) * BOHR_TO_ANG


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_structure_pair(base: Path, atoms) -> None:
    write(str(base.with_suffix(".xyz")), atoms)
    write(str(base.with_suffix(".vasp")), atoms, vasp5=True, direct=True)


def _latest_bulk_validation_csv() -> Path | None:
    candidates = sorted(
        (ROOT / "results").glob("bulk_Al_fcc_TFVW*/bulk_validation.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _read_bulk_mu(path: Path | None) -> tuple[float | None, float | None, str]:
    if path is None or not path.exists():
        return None, None, ""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None, str(path)
    zero_row = min(rows, key=lambda row: abs(float(row["strain"])))
    return float(zero_row["energy_per_atom_eV"]), float(zero_row["strain"]), str(path)


def _xy_center(atoms) -> tuple[float, float]:
    cell = atoms.get_cell().array
    return float(cell[0, 0] * 0.5), float(cell[1, 1] * 0.5)


def _xy_spans(atoms) -> tuple[float, float]:
    pos = atoms.get_positions()
    return float(pos[:, 0].max() - pos[:, 0].min()), float(pos[:, 1].max() - pos[:, 1].min())


def _ellipse_area_xy(atoms) -> float:
    span_x, span_y = _xy_spans(atoms)
    return float(np.pi * span_x * span_y * 0.25)


def _add_z_vacuum_and_center(atoms, z_vacuum: float):
    finite = atoms.copy()
    pos = finite.get_positions()
    zmin = float(pos[:, 2].min())
    zmax = float(pos[:, 2].max())
    wire_span_z = zmax - zmin
    if wire_span_z <= 0.0:
        raise ValueError("Wire z span is not positive.")

    cell = finite.get_cell().array.copy()
    cell[2, :] = [0.0, 0.0, wire_span_z + 2.0 * float(z_vacuum)]
    finite.set_cell(cell, scale_atoms=False)

    pos = finite.get_positions()
    pos[:, 2] += float(z_vacuum) - float(pos[:, 2].min())
    finite.set_positions(pos)
    finite.pbc = [True, True, True]
    return finite


def build_finite_grip_wire(
    *,
    a0: float,
    diameter_nm: float,
    wire_length: float,
    xy_vacuum: float,
    z_vacuum: float,
    orientation: str,
):
    wire = build_circular_nanowire(
        a0=float(a0),
        diameter_nm=float(diameter_nm),
        length_z=float(wire_length),
        vacuum=float(xy_vacuum),
        orientation=str(orientation),
    )
    return _add_z_vacuum_and_center(wire, z_vacuum=float(z_vacuum))


def select_grip_indices(atoms, grip_thickness: float) -> tuple[np.ndarray, np.ndarray]:
    z = atoms.get_positions()[:, 2]
    zmin = float(z.min())
    zmax = float(z.max())
    thickness = float(grip_thickness)
    if thickness <= 0.0:
        raise ValueError("grip_thickness must be positive.")
    bottom = np.where(z <= zmin + thickness)[0].astype(int)
    top = np.where(z >= zmax - thickness)[0].astype(int)
    if bottom.size == 0 or top.size == 0:
        raise ValueError(
            f"Empty grip selection: bottom={bottom.size}, top={top.size}, thickness={thickness}"
        )
    overlap = np.intersect1d(bottom, top)
    if overlap.size > 0:
        raise ValueError("Top and bottom grip selections overlap; increase wire length or reduce grip thickness.")
    return bottom, top


def _remap_after_delete(indices: np.ndarray, removed: int) -> np.ndarray:
    out: list[int] = []
    for idx in np.asarray(indices, dtype=int).ravel():
        if int(idx) == int(removed):
            continue
        out.append(int(idx) - 1 if int(idx) > int(removed) else int(idx))
    return np.asarray(out, dtype=int)


def choose_free_surface_vacancy(
    atoms,
    *,
    bottom_idx: np.ndarray,
    top_idx: np.ndarray,
    z_window_fraction: float,
) -> tuple[int, dict[str, float]]:
    pos = atoms.get_positions()
    n_atoms = len(atoms)
    fixed_mask = np.zeros(n_atoms, dtype=bool)
    fixed_mask[np.asarray(bottom_idx, dtype=int)] = True
    fixed_mask[np.asarray(top_idx, dtype=int)] = True
    free_idx = np.where(~fixed_mask)[0]
    if free_idx.size == 0:
        raise ValueError("No free atoms remain after grip selection.")

    free_z = pos[free_idx, 2]
    free_z_min = float(free_z.min())
    free_z_max = float(free_z.max())
    free_center = 0.5 * (free_z_min + free_z_max)
    free_length = free_z_max - free_z_min
    if free_length <= 0.0:
        raise ValueError("Free-region z span is not positive.")

    window = max(1e-6, min(0.5 * free_length, float(z_window_fraction) * free_length))
    central = free_idx[np.abs(pos[free_idx, 2] - free_center) <= window]
    if central.size == 0:
        central = free_idx

    cx, cy = _xy_center(atoms)
    radial = np.hypot(pos[:, 0] - cx, pos[:, 1] - cy)
    chosen = int(central[int(np.argmax(radial[central]))])
    site = {
        "index": chosen,
        "x_A": float(pos[chosen, 0]),
        "y_A": float(pos[chosen, 1]),
        "z_A": float(pos[chosen, 2]),
        "radial_distance_A": float(radial[chosen]),
        "free_z_min_A": free_z_min,
        "free_z_max_A": free_z_max,
        "free_length_A": float(free_length),
        "free_center_z_A": float(free_center),
        "z_center_offset_A": float(abs(pos[chosen, 2] - free_center)),
        "z_window_A": float(window),
    }
    return chosen, site


def _delete_atom(atoms, atom_index: int):
    defect = atoms.copy()
    del defect[int(atom_index)]
    return defect


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Prepare a finite [111] nanowire with fixed end grips and a vacancy in the central free region."
    )
    ap.add_argument("--case", default="", help="Case directory name. Defaults to a timestamped finite-grip case.")
    ap.add_argument("--diameter-nm", type=float, required=True)
    ap.add_argument("--orientation", choices=["111", "100", "110"], default="111")
    ap.add_argument("--a0", type=float, default=4.118877004246)
    ap.add_argument("--wire-length", type=float, default=21.0, help="Target finite wire length before z vacuum, in Angstrom.")
    ap.add_argument("--min-wire-span", type=float, default=10.0, help="Minimum atomistic wire z-span in Angstrom.")
    ap.add_argument("--xy-vacuum", type=float, default=10.0, help="Vacuum padding in x/y, in Angstrom.")
    ap.add_argument("--z-vacuum", type=float, default=10.0, help="Vacuum padding at both z ends, in Angstrom.")
    ap.add_argument("--grip-thickness", type=float, default=3.0, help="Thickness of each fixed grip region in Angstrom.")
    ap.add_argument("--vacancy-z-window-fraction", type=float, default=0.35)
    ap.add_argument("--pp", default="al.gga.recpot")
    ap.add_argument("--kedf", default="TFVW")
    ap.add_argument("--ecut", type=float, default=1000.0)
    ap.add_argument("--spacing", type=float, default=None)
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--relax-steps", type=int, default=120)
    ap.add_argument("--outdir", default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pp_path = Path(args.pp).resolve()
    if not pp_path.exists():
        raise FileNotFoundError(f"pp file not found: {pp_path}")

    kedf_name = normalize_kedf_name(args.kedf)
    spacing = float(args.spacing) if args.spacing is not None else ecut_to_spacing_angstrom(float(args.ecut))
    case_name = args.case or f"finite_grip_{args.orientation}_{float(args.diameter_nm):.1f}nm_vacancy_tfvw_{_ts()}"
    outdir = Path(args.outdir).resolve() if str(args.outdir).strip() else ROOT / "cases" / case_name / "inputs"
    outdir.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("[grip-prepare] finite grip vacancy wire")
    print(f"[grip-prepare] case       : {case_name}")
    print(f"[grip-prepare] outdir     : {outdir}")
    print(f"[grip-prepare] diameter   : {float(args.diameter_nm):.3f} nm")
    print(f"[grip-prepare] wire length: {float(args.wire_length):.3f} A")
    print(f"[grip-prepare] grip thick : {float(args.grip_thickness):.3f} A")
    print(f"[grip-prepare] KEDF       : {args.kedf} -> {kedf_name}")
    print(f"[grip-prepare] spacing    : {spacing:.6f} A")
    print("========================================")

    pristine_raw = build_finite_grip_wire(
        a0=float(args.a0),
        diameter_nm=float(args.diameter_nm),
        wire_length=float(args.wire_length),
        xy_vacuum=float(args.xy_vacuum),
        z_vacuum=float(args.z_vacuum),
        orientation=str(args.orientation),
    )
    raw_pos = pristine_raw.get_positions()
    raw_wire_span = float(raw_pos[:, 2].max() - raw_pos[:, 2].min())
    if raw_wire_span < float(args.min_wire_span):
        raise RuntimeError(
            f"Finite wire z-span {raw_wire_span:.6f} A is below required {float(args.min_wire_span):.6f} A"
        )
    bottom_raw, top_raw = select_grip_indices(pristine_raw, grip_thickness=float(args.grip_thickness))
    fixed_raw = np.unique(np.concatenate([bottom_raw, top_raw])).astype(int)
    _write_structure_pair(outdir / "pristine_raw", pristine_raw)

    pristine_eq, pristine_energy, pristine_stress = relax_atoms(
        pristine_raw,
        pp_file=str(pp_path),
        spacing=spacing,
        fixed_idx=fixed_raw,
        kedf=kedf_name,
        fmax=float(args.fmax),
        steps=int(args.relax_steps),
        logfile=str(outdir / "pristine_grip_relax.log"),
        trajfile=str(outdir / "pristine_grip_relax.traj"),
        dftpy_outfile=str(outdir / "pristine_grip_dftpy.out"),
        debug_fixed=True,
    )
    _write_structure_pair(outdir / "pristine_equilibrium", pristine_eq)

    vacancy_idx, vacancy_site = choose_free_surface_vacancy(
        pristine_eq,
        bottom_idx=bottom_raw,
        top_idx=top_raw,
        z_window_fraction=float(args.vacancy_z_window_fraction),
    )
    vacancy_start = _delete_atom(pristine_eq, vacancy_idx)
    bottom_vac = _remap_after_delete(bottom_raw, vacancy_idx)
    top_vac = _remap_after_delete(top_raw, vacancy_idx)
    fixed_vac = np.unique(np.concatenate([bottom_vac, top_vac])).astype(int)
    _write_structure_pair(outdir / "vacancy_start", vacancy_start)

    vacancy_eq, vacancy_energy, vacancy_stress = relax_atoms(
        vacancy_start,
        pp_file=str(pp_path),
        spacing=spacing,
        fixed_idx=fixed_vac,
        kedf=kedf_name,
        fmax=float(args.fmax),
        steps=int(args.relax_steps),
        logfile=str(outdir / "vacancy_grip_relax.log"),
        trajfile=str(outdir / "vacancy_grip_relax.traj"),
        dftpy_outfile=str(outdir / "vacancy_grip_dftpy.out"),
        debug_fixed=True,
    )
    _write_structure_pair(outdir / "vacancy_equilibrium", vacancy_eq)

    bulk_mu, bulk_mu_strain, bulk_mu_source = _read_bulk_mu(_latest_bulk_validation_csv())
    formation = None if bulk_mu is None else float(vacancy_energy - pristine_energy + bulk_mu)

    pos_eq = vacancy_eq.get_positions()
    bottom_center = float(pos_eq[bottom_vac, 2].mean())
    top_center = float(pos_eq[top_vac, 2].mean())
    grip_distance = float(top_center - bottom_center)
    wire_span = float(pos_eq[:, 2].max() - pos_eq[:, 2].min())
    metadata = {
        "case": case_name,
        "boundary_condition": "finite_grip_displacement_control",
        "bottom_grip_indices": [int(i) for i in bottom_vac],
        "top_grip_indices": [int(i) for i in top_vac],
        "fixed_indices": [int(i) for i in fixed_vac],
        "grip_thickness_A": float(args.grip_thickness),
        "bottom_grip_center_z_A": bottom_center,
        "top_grip_center_z_A": top_center,
        "grip_distance_ref_A": grip_distance,
        "wire_span_ref_A": wire_span,
        "area_ref_ellipse_A2": _ellipse_area_xy(vacancy_eq),
        "cell_lengths_A": [float(x) for x in vacancy_eq.get_cell().lengths()],
    }
    (outdir / "grip_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "case": case_name,
        "branch": "finite_grip_vacancy",
        "geometry": {
            "builder": "paper_circular_finite_grip",
            "orientation": str(args.orientation),
            "diameter_nm": float(args.diameter_nm),
            "a0_input_A": float(args.a0),
            "wire_length_target_A": float(args.wire_length),
            "raw_wire_span_A": raw_wire_span,
            "vacancy_wire_span_A": wire_span,
            "xy_vacuum_A": float(args.xy_vacuum),
            "z_vacuum_A": float(args.z_vacuum),
            "min_wire_span_requirement_A": float(args.min_wire_span),
        },
        "grips": {
            "grip_thickness_A": float(args.grip_thickness),
            "bottom_count": int(bottom_vac.size),
            "top_count": int(top_vac.size),
            "free_atom_count": int(len(vacancy_eq) - fixed_vac.size),
            "grip_distance_ref_A": grip_distance,
            "boundary_condition": "bottom/top grip atoms fixed during relaxation; grips are displaced during tensile loading",
        },
        "dft": {
            "pp": str(pp_path),
            "kedf": kedf_name,
            "spacing_A": spacing,
            "fmax": float(args.fmax),
            "relax_steps": int(args.relax_steps),
        },
        "pristine": {
            "n_atoms": int(len(pristine_eq)),
            "energy_eV": float(pristine_energy),
            "energy_per_atom_eV": float(pristine_energy / len(pristine_eq)),
            "sigma_zz_GPa": float(pristine_stress[2, 2]),
        },
        "vacancy": {
            "n_atoms": int(len(vacancy_eq)),
            "energy_eV": float(vacancy_energy),
            "energy_per_atom_eV": float(vacancy_energy / len(vacancy_eq)),
            "sigma_zz_GPa": float(vacancy_stress[2, 2]),
            "formation_energy_bulk_mu_eV": formation,
            "bulk_mu_eV_per_atom": bulk_mu,
            "bulk_mu_source_csv": bulk_mu_source,
            "bulk_mu_reference_strain": bulk_mu_strain,
            "selected_site": vacancy_site,
            "selection_rule": "outermost atom inside the central free region, excluding fixed grip atoms",
        },
        "artifacts": {
            "pristine_raw": str(outdir / "pristine_raw.vasp"),
            "pristine_equilibrium": str(outdir / "pristine_equilibrium.vasp"),
            "vacancy_start": str(outdir / "vacancy_start.vasp"),
            "vacancy_equilibrium": str(outdir / "vacancy_equilibrium.vasp"),
            "grip_metadata": str(outdir / "grip_metadata.json"),
        },
    }
    (outdir / "grip_vacancy_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[grip-prepare] pristine atoms       : {len(pristine_eq)}")
    print(f"[grip-prepare] vacancy atoms        : {len(vacancy_eq)}")
    print(f"[grip-prepare] vacancy wire span    : {wire_span:.6f} A")
    print(f"[grip-prepare] grip distance ref    : {grip_distance:.6f} A")
    print(f"[grip-prepare] bottom/top grip atoms: {bottom_vac.size} / {top_vac.size}")
    print(f"[grip-prepare] vacancy index        : {vacancy_idx}")
    print(f"[grip-prepare] vacancy free z range : {vacancy_site['free_z_min_A']:.6f} / {vacancy_site['free_z_max_A']:.6f} A")
    print(f"[grip-prepare] Wrote: {outdir / 'vacancy_equilibrium.vasp'}")


if __name__ == "__main__":
    main()
