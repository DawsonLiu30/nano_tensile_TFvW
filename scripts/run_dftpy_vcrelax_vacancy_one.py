from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dft_engine import relax_atoms_and_cell


def repeat_label(manifest: dict[str, object]) -> str:
    repeat = manifest.get("conventional_repeat", ["?", "?", "?"])
    if isinstance(repeat, list) and len(repeat) == 3:
        return f"conv_{int(repeat[0]):02d}x{int(repeat[1]):02d}x{int(repeat[2]):02d}"
    return str(manifest.get("conventional_repeat_label", "unknown"))


def resolve_case(rootdir: Path, setting: str, scan: str) -> Path:
    candidates: list[Path] = []
    if scan in {"auto", "spacing"}:
        candidates.append(rootdir / "spacing_scan" / setting)
    if scan in {"auto", "size"}:
        candidates.append(rootdir / "size_scan" / setting)
    if scan in {"auto", "weight"}:
        candidates.append(rootdir / "weight_scan" / setting)
    if scan in {"auto", "pair"}:
        candidates.append(rootdir / "pair_scan" / setting)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find setting. Tried:\n" + "\n".join(str(x) for x in candidates))


def write_structure_pair(base: Path, atoms) -> None:
    write(str(base.with_suffix(".xyz")), atoms)
    write(str(base.with_suffix(".vasp")), atoms, direct=True, vasp5=True)


def write_calculator_provenance(
    path: Path,
    *,
    structure_file: str,
    pp_file: Path,
    spacing: float,
    kedf: str,
    xc: str,
    kedf_x: float,
    kedf_y: float,
    fmax: float,
    steps: int,
    pressure_gpa: float,
    ase_optimizer: str,
) -> None:
    data = {
        "structure_file": structure_file,
        "dftpy_calculator": {
            "JOB": {"calctype": "Energy Force Stress"},
            "PATH": {"pppath": str(pp_file.parent)},
            "PP": {"Al": pp_file.name},
            "GRID": {"spacing": spacing},
            "EXC": {"xc": xc},
            "KEDF": {"kedf": kedf, "x": kedf_x, "y": kedf_y},
            "OPT": {"method": "LBFGS"},
        },
        "ase_full_relaxation": {
            "cell_filter": "FrechetCellFilter",
            "optimizer": ase_optimizer,
            "fmax_eV_A": fmax,
            "max_steps": steps,
            "scalar_pressure_GPa": pressure_gpa,
        },
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run DFTpy full atom+cell relaxation for one vacancy case.")
    ap.add_argument("--rootdir", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--scan", choices=["auto", "spacing", "size", "weight", "pair"], default="auto")
    ap.add_argument("--pressure-gpa", type=float, default=0.0)
    ap.add_argument(
        "--ase-optimizer",
        choices=["BFGS", "LBFGS", "BFGSLineSearch", "SciPyFminBFGS", "SciPyFminCG", "MDMin"],
        default="BFGS",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rootdir = Path(args.rootdir).expanduser().resolve()
    case_dir = resolve_case(rootdir, str(args.setting), str(args.scan))
    manifest = json.loads((case_dir / "point_manifest.json").read_text(encoding="utf-8"))

    pp_declared = Path(str(manifest["pp_file"])).expanduser()
    pp_candidates = [
        pp_declared if pp_declared.is_absolute() else case_dir / pp_declared,
        case_dir / pp_declared.name,
        rootdir / pp_declared.name,
    ]
    pp_file = next((candidate.resolve() for candidate in pp_candidates if candidate.exists()), None)
    if pp_file is None:
        raise FileNotFoundError(
            "Missing DFTpy pseudopotential. Tried:\n" + "\n".join(str(path) for path in pp_candidates)
        )

    spacing = float(manifest["spacing_A"])
    kedf = str(manifest["kedf"])
    xc = str(manifest.get("xc", "PBE")).strip().upper()
    kedf_x = float(manifest.get("kedf_x", 1.0))
    kedf_y = float(manifest.get("kedf_y", 1.0))
    fmax = float(manifest["fmax_eV_per_A"])
    steps = int(manifest["relax_steps"])

    is_pair = str(manifest.get("scan_type", "")) == "pair"
    defect_label = "divacancy" if is_pair else "vacancy"
    defect_start = case_dir / f"{defect_label}_start.vasp"
    if not defect_start.exists() and is_pair:
        # Backward compatibility for packages prepared before the explicit
        # divacancy naming fix.
        defect_start = case_dir / "vacancy_start.vasp"

    pristine = read(str(case_dir / "pristine_raw.vasp"))
    defect = read(str(defect_start))

    write_calculator_provenance(
        case_dir / "dftpy_pristine_calculator_config.json",
        structure_file="pristine_raw.vasp",
        pp_file=pp_file,
        spacing=spacing,
        kedf=kedf,
        xc=xc,
        kedf_x=kedf_x,
        kedf_y=kedf_y,
        fmax=fmax,
        steps=steps,
        pressure_gpa=float(args.pressure_gpa),
        ase_optimizer=str(args.ase_optimizer),
    )
    write_calculator_provenance(
        case_dir / f"dftpy_{defect_label}_calculator_config.json",
        structure_file=defect_start.name,
        pp_file=pp_file,
        spacing=spacing,
        kedf=kedf,
        xc=xc,
        kedf_x=kedf_x,
        kedf_y=kedf_y,
        fmax=fmax,
        steps=steps,
        pressure_gpa=float(args.pressure_gpa),
        ase_optimizer=str(args.ase_optimizer),
    )

    pristine_relaxed, pristine_energy, pristine_stress = relax_atoms_and_cell(
        pristine,
        pp_file=pp_file,
        spacing=spacing,
        kedf=kedf,
        xc=xc,
        kedf_x=kedf_x,
        kedf_y=kedf_y,
        fmax=fmax,
        steps=steps,
        logfile=str(case_dir / "pristine_relax.log"),
        trajfile=str(case_dir / "pristine_relax.traj"),
        dftpy_outfile=str(case_dir / "pristine_dftpy.out"),
        scalar_pressure_gpa=float(args.pressure_gpa),
        ase_optimizer=str(args.ase_optimizer),
    )
    defect_relaxed, defect_energy, defect_stress = relax_atoms_and_cell(
        defect,
        pp_file=pp_file,
        spacing=spacing,
        kedf=kedf,
        xc=xc,
        kedf_x=kedf_x,
        kedf_y=kedf_y,
        fmax=fmax,
        steps=steps,
        logfile=str(case_dir / f"{defect_label}_relax.log"),
        trajfile=str(case_dir / f"{defect_label}_relax.traj"),
        dftpy_outfile=str(case_dir / f"{defect_label}_dftpy.out"),
        scalar_pressure_gpa=float(args.pressure_gpa),
        ase_optimizer=str(args.ase_optimizer),
    )

    write_structure_pair(case_dir / "pristine_vc_relaxed", pristine_relaxed)
    write_structure_pair(case_dir / f"{defect_label}_vc_relaxed", defect_relaxed)
    write_structure_pair(case_dir / "pristine_relaxed", pristine_relaxed)
    write_structure_pair(case_dir / f"{defect_label}_relaxed", defect_relaxed)

    n_pristine = int(manifest["pristine_n_atoms"])
    n_vacancy = int(manifest["vacancy_n_atoms"])
    vacancy_count = n_pristine - n_vacancy
    ef_vac = float(defect_energy - (n_vacancy / n_pristine) * pristine_energy)
    pristine_fmax = float(np.linalg.norm(pristine_relaxed.get_forces(), axis=1).max())
    defect_fmax = float(np.linalg.norm(defect_relaxed.get_forces(), axis=1).max())

    result = {
        "setting": str(manifest["setting"]),
        "scan_type": str(manifest.get("scan_type", "unknown")),
        "relaxation_mode": "full_atom_and_cell_relaxation_vc_relax_equivalent",
        "cell_basis": str(manifest["cell_basis"]),
        "conventional_repeat_label": repeat_label(manifest),
        "conventional_repeat": manifest.get("conventional_repeat", []),
        "pristine_n_atoms": n_pristine,
        "vacancy_n_atoms": n_vacancy,
        "vacancy_count": vacancy_count,
        "defect_label": defect_label,
        "vacancy_concentration_fraction": float(vacancy_count) / float(n_pristine),
        "vacancy_concentration_percent": 100.0 * float(vacancy_count) / float(n_pristine),
        "pair_distance_A": manifest.get("pair_distance_A"),
        "spacing_A": spacing,
        "ecut_analogue_eV": float(manifest.get("ecut_analogue_eV", 0.0)),
        "kedf": kedf,
        "kedf_x": kedf_x,
        "kedf_y": kedf_y,
        "xc": xc,
        "fmax_eV_per_A": fmax,
        "target_pressure_GPa": float(args.pressure_gpa),
        "ase_optimizer": str(args.ase_optimizer),
        "pristine_energy_eV": float(pristine_energy),
        "vacancy_energy_eV": float(defect_energy),
        f"{defect_label}_energy_eV": float(defect_energy),
        "vacancy_formation_energy_eV": ef_vac,
        "pristine_stress_GPa": pristine_stress.tolist(),
        "vacancy_stress_GPa": defect_stress.tolist(),
        f"{defect_label}_stress_GPa": defect_stress.tolist(),
        "pristine_final_fmax_eV_A": pristine_fmax,
        "vacancy_final_fmax_eV_A": defect_fmax,
        f"{defect_label}_final_fmax_eV_A": defect_fmax,
        "pristine_cell_lengths_A": [float(x) for x in pristine_relaxed.cell.lengths()],
        "vacancy_cell_lengths_A": [float(x) for x in defect_relaxed.cell.lengths()],
        f"{defect_label}_cell_lengths_A": [float(x) for x in defect_relaxed.cell.lengths()],
        "pristine_cell_angles_deg": [float(x) for x in pristine_relaxed.cell.angles()],
        "vacancy_cell_angles_deg": [float(x) for x in defect_relaxed.cell.angles()],
        f"{defect_label}_cell_angles_deg": [float(x) for x in defect_relaxed.cell.angles()],
        "formula": "E_f^defect = E_full-relax_defect^(N-nvac) - ((N-nvac)/N) E_full-relax_pristine^N",
    }
    (case_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
