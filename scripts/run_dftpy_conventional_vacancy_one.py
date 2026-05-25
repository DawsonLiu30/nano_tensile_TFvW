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


def repeat_label(manifest: dict[str, object]) -> str:
    if "conventional_repeat_label" in manifest:
        return str(manifest["conventional_repeat_label"])
    if "conventional_repeat_n" in manifest:
        n = int(manifest["conventional_repeat_n"])
        return f"conv_{n:02d}x{n:02d}x{n:02d}"
    repeat = manifest.get("conventional_repeat", ["?", "?", "?"])
    if isinstance(repeat, list) and len(repeat) == 3:
        return f"conv_{int(repeat[0]):02d}x{int(repeat[1]):02d}x{int(repeat[2]):02d}"
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one VESTA-checked conventional fcc DFTpy vacancy case."
    )
    parser.add_argument("--rootdir", required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument(
        "--scan",
        choices=["auto", "spacing", "size"],
        default="auto",
        help="Scan folder to search. Auto checks spacing_scan then size_scan.",
    )
    return parser.parse_args()


def resolve_case(rootdir: Path, setting: str, scan: str) -> Path:
    candidates: list[Path] = []
    if scan in {"auto", "spacing"}:
        candidates.append(rootdir / "spacing_scan" / setting)
    if scan in {"auto", "size"}:
        candidates.append(rootdir / "size_scan" / setting)
    for case_dir in candidates:
        if case_dir.exists():
            return case_dir
    tried = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find setting '{setting}'. Tried:\n{tried}")


def write_structure_pair(base: Path, atoms) -> None:
    write(str(base.with_suffix(".xyz")), atoms)
    write(str(base.with_suffix(".vasp")), atoms, direct=True, vasp5=True)


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
    fmax = float(manifest["fmax_eV_per_A"])
    relax_steps = int(manifest["relax_steps"])

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
    write_structure_pair(case_dir / "vacancy_relaxed", relaxed_vacancy)

    n_pristine = int(manifest["pristine_n_atoms"])
    n_vacancy = int(manifest["vacancy_n_atoms"])
    ef_vac = float(vacancy_energy_eV - (n_vacancy / n_pristine) * pristine_energy_eV)

    result = {
        "setting": str(manifest["setting"]),
        "scan_type": str(manifest.get("scan_type", "unknown")),
        "cell_basis": str(manifest["cell_basis"]),
        "conventional_repeat_label": repeat_label(manifest),
        "conventional_repeat": manifest.get("conventional_repeat", []),
        "pristine_n_atoms": n_pristine,
        "vacancy_n_atoms": n_vacancy,
        "vacancy_concentration_fraction": 1.0 / float(n_pristine),
        "vacancy_concentration_percent": 100.0 / float(n_pristine),
        "spacing_A": spacing,
        "ecut_analogue_eV": float(manifest.get("ecut_analogue_eV", 0.0)),
        "kedf": kedf,
        "fmax_eV_per_A": fmax,
        "pristine_energy_eV": float(pristine_energy_eV),
        "vacancy_energy_eV": float(vacancy_energy_eV),
        "vacancy_formation_energy_eV": ef_vac,
        "pristine_stress_GPa": pristine_stress.tolist(),
        "vacancy_stress_GPa": vacancy_stress.tolist(),
    }
    (case_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
