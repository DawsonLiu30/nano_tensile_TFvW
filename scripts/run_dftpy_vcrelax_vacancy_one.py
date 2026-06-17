from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

    pp_file = Path(str(manifest["pp_file"])).expanduser().resolve()
    if not pp_file.exists():
        raise FileNotFoundError(f"Missing DFTpy pseudopotential: {pp_file}")

    spacing = float(manifest["spacing_A"])
    kedf = str(manifest["kedf"])
    xc = str(manifest.get("xc", "PBE")).strip().upper()
    kedf_x = float(manifest.get("kedf_x", 1.0))
    kedf_y = float(manifest.get("kedf_y", 1.0))
    fmax = float(manifest["fmax_eV_per_A"])
    steps = int(manifest["relax_steps"])

    pristine = read(str(case_dir / "pristine_raw.vasp"))
    vacancy = read(str(case_dir / "vacancy_start.vasp"))

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
    vacancy_relaxed, vacancy_energy, vacancy_stress = relax_atoms_and_cell(
        vacancy,
        pp_file=pp_file,
        spacing=spacing,
        kedf=kedf,
        xc=xc,
        kedf_x=kedf_x,
        kedf_y=kedf_y,
        fmax=fmax,
        steps=steps,
        logfile=str(case_dir / "vacancy_relax.log"),
        trajfile=str(case_dir / "vacancy_relax.traj"),
        dftpy_outfile=str(case_dir / "vacancy_dftpy.out"),
        scalar_pressure_gpa=float(args.pressure_gpa),
        ase_optimizer=str(args.ase_optimizer),
    )

    write_structure_pair(case_dir / "pristine_vc_relaxed", pristine_relaxed)
    write_structure_pair(case_dir / "vacancy_vc_relaxed", vacancy_relaxed)
    write_structure_pair(case_dir / "pristine_relaxed", pristine_relaxed)
    write_structure_pair(case_dir / "vacancy_relaxed", vacancy_relaxed)

    n_pristine = int(manifest["pristine_n_atoms"])
    n_vacancy = int(manifest["vacancy_n_atoms"])
    vacancy_count = n_pristine - n_vacancy
    ef_vac = float(vacancy_energy - (n_vacancy / n_pristine) * pristine_energy)

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
        "vacancy_energy_eV": float(vacancy_energy),
        "vacancy_formation_energy_eV": ef_vac,
        "pristine_stress_GPa": pristine_stress.tolist(),
        "vacancy_stress_GPa": vacancy_stress.tolist(),
        "pristine_cell_lengths_A": [float(x) for x in pristine_relaxed.cell.lengths()],
        "vacancy_cell_lengths_A": [float(x) for x in vacancy_relaxed.cell.lengths()],
        "pristine_cell_angles_deg": [float(x) for x in pristine_relaxed.cell.angles()],
        "vacancy_cell_angles_deg": [float(x) for x in vacancy_relaxed.cell.angles()],
        "formula": "E_f^defect = E_full-relax_defect^(N-nvac) - ((N-nvac)/N) E_full-relax_pristine^N",
    }
    (case_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
