from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ase.io import read, write

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dft_engine import evaluate_atoms, relax_atoms


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run one prepared DFTpy vacancy-convergence setting.")
    ap.add_argument("--rootdir", required=True)
    ap.add_argument("--setting", required=True, help="Example: ecut_0300eV")
    return ap.parse_args()


def _write_structure_pair(base: Path, atoms) -> None:
    write(str(base.with_suffix(".xyz")), atoms)
    write(str(base.with_suffix(".vasp")), atoms, direct=True, vasp5=True)


def main() -> None:
    args = parse_args()
    rootdir = Path(args.rootdir).expanduser().resolve()
    case_dir = rootdir / "spacing_scan" / str(args.setting)
    if not case_dir.exists():
        legacy_case_dir = rootdir / "ecut_scan" / str(args.setting)
        if legacy_case_dir.exists():
            case_dir = legacy_case_dir
        else:
            raise FileNotFoundError(f"Missing case dir for setting '{args.setting}' under spacing_scan/ or ecut_scan/.")

    point_manifest = json.loads((case_dir / "point_manifest.json").read_text(encoding="utf-8"))
    pp_file = Path(str(point_manifest["pp_file"])).expanduser().resolve()
    spacing = float(point_manifest["spacing_A"])
    kedf = str(point_manifest["kedf"])
    fmax = float(point_manifest["fmax_eV_per_A"])
    relax_steps = int(point_manifest["relax_steps"])

    pristine = read(str(case_dir / "pristine_raw.vasp"))
    vacancy = read(str(case_dir / "vacancy_start.vasp"))

    _, pristine_energy_eV, pristine_stress = evaluate_atoms(
        pristine,
        pp_file=pp_file,
        spacing=spacing,
        kedf=kedf,
        dftpy_outfile=str(case_dir / "pristine_dftpy.out"),
    )

    relaxed_vacancy, vacancy_energy_eV, vacancy_stress = relax_atoms(
        vacancy,
        pp_file=pp_file,
        spacing=spacing,
        fixed_idx=[],
        kedf=kedf,
        fmax=fmax,
        steps=relax_steps,
        logfile=str(case_dir / "vacancy_relax.log"),
        trajfile=str(case_dir / "vacancy_relax.traj"),
        dftpy_outfile=str(case_dir / "vacancy_dftpy.out"),
        debug_fixed=False,
    )
    _write_structure_pair(case_dir / "vacancy_relaxed", relaxed_vacancy)

    pristine_atoms = int(point_manifest["pristine_n_atoms"])
    ef_vac = float(vacancy_energy_eV - ((pristine_atoms - 1) / pristine_atoms) * pristine_energy_eV)
    result = {
        "setting": point_manifest["setting"],
        "grid_parameter": str(point_manifest.get("grid_parameter", "spacing")),
        "spacing_A": float(point_manifest["spacing_A"]),
        "ecut_analogue_eV": float(point_manifest.get("ecut_analogue_eV", point_manifest.get("ecut_eV", 0.0))),
        "pristine_energy_eV": float(pristine_energy_eV),
        "vacancy_energy_eV": float(vacancy_energy_eV),
        "vacancy_formation_energy_eV": float(ef_vac),
        "pristine_stress_GPa": pristine_stress.tolist(),
        "vacancy_stress_GPa": vacancy_stress.tolist(),
    }
    (case_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
