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
    ap = argparse.ArgumentParser(description="Run one prepared DFTpy primitive-size vacancy case.")
    ap.add_argument("--rootdir", required=True)
    ap.add_argument("--setting", required=True)
    return ap.parse_args()


def _write_structure_pair(base: Path, atoms) -> None:
    write(str(base.with_suffix(".xyz")), atoms)
    write(str(base.with_suffix(".vasp")), atoms, direct=True, vasp5=True)


def _resolve_pp(pp_text: str, case_dir: Path) -> Path:
    candidate = Path(str(pp_text)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    root_candidate = (ROOT / candidate).resolve()
    if root_candidate.exists():
        return root_candidate
    case_candidate = (case_dir / candidate).resolve()
    return case_candidate


def main() -> None:
    args = parse_args()
    rootdir = Path(args.rootdir).expanduser().resolve()
    case_dir = rootdir / "size_scan" / str(args.setting)
    if not case_dir.exists():
        raise FileNotFoundError(f"Missing case dir: {case_dir}")

    manifest = json.loads((case_dir / "point_manifest.json").read_text(encoding="utf-8"))
    pp_file = _resolve_pp(str(manifest["pp_file"]), case_dir)
    if not pp_file.exists():
        raise FileNotFoundError(f"pp_file not found: {pp_file}")

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
    _write_structure_pair(case_dir / "vacancy_relaxed", relaxed_vacancy)

    n_pristine = int(manifest["pristine_n_atoms"])
    n_vacancy = int(manifest["vacancy_n_atoms"])
    ef_vac = float(vacancy_energy_eV - (n_vacancy / n_pristine) * pristine_energy_eV)

    result = {
        "setting": str(manifest["setting"]),
        "cell_basis": str(manifest["cell_basis"]),
        "a0_A": float(manifest["a0_A"]),
        "repeat_n": int(manifest["repeat_n"]),
        "pristine_n_atoms": n_pristine,
        "vacancy_n_atoms": n_vacancy,
        "volume_A3": float(manifest["volume_A3"]),
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
