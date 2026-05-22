from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from ase.io import read, write
from ase.neighborlist import neighbor_list


ROOT = Path(__file__).resolve().parents[1]
HA_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903


def spacing_angstrom_to_ecut_ev(spacing_A: float) -> float:
    h_bohr = float(spacing_A) / BOHR_TO_ANG
    ecut_ha = (math.pi / h_bohr) ** 2 / 2.0
    return ecut_ha * HA_TO_EV


def _parse_float_list(text: str) -> list[float]:
    values = []
    for chunk in str(text).split(","):
        token = chunk.strip()
        if token:
            values.append(float(token))
    if not values:
        raise ValueError("No spacing values were provided.")
    return values


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_structure_pair(base: Path, atoms) -> None:
    write(str(base.with_suffix(".xyz")), atoms)
    write(str(base.with_suffix(".vasp")), atoms, direct=True, vasp5=True)


def _spacing_label(spacing_A: float) -> str:
    return f"spacing_{spacing_A:.2f}A".replace(".", "p")


def _write_config_ini(path: Path, *, pp_filename: str, spacing_A: float, kedf: str, cellfile: str) -> None:
    text = f"""[JOB]
task = Optdensity
calctype = Energy Force Stress

[PATH]
pppath = ./
cellpath = ./

[MATH]
linearie = T
linearii = T

[PP]
Al = {pp_filename}

[CELL]
cellfile = {cellfile}
format = vasp

[GRID]
spacing = {spacing_A:.8f}

[DENSITY]
densityoutput = den.xsf

[EXC]
xc = PBE

[KEDF]
kedf = {kedf}
x = 1.0
y = 1.0

[OPT]
method = TN
maxiter = 300
econv = 1e-6
"""
    _write_text(path, text)


def _geometry_summary(atoms) -> dict[str, float | int | str]:
    atoms = atoms.copy()
    atoms.pbc = [True, True, True]
    summary: dict[str, float | int | str] = {
        "n_atoms": int(len(atoms)),
        "formula": atoms.get_chemical_formula(),
        "volume_A3": float(atoms.get_volume()),
        "a_A": float(atoms.cell.lengths()[0]),
        "b_A": float(atoms.cell.lengths()[1]),
        "c_A": float(atoms.cell.lengths()[2]),
    }
    if len(atoms) > 1:
        i, j, d = neighbor_list("ijd", atoms, cutoff=4.0)
        summary["min_distance_A"] = float(d.min()) if len(d) else float("nan")
    else:
        summary["min_distance_A"] = float("nan")
    z = sorted(float(v) % float(atoms.cell.lengths()[2]) for v in atoms.positions[:, 2])
    if z:
        gaps = [z[k + 1] - z[k] for k in range(len(z) - 1)]
        gaps.append(float(atoms.cell.lengths()[2]) - z[-1] + z[0])
        summary["max_z_gap_A"] = float(max(gaps))
    else:
        summary["max_z_gap_A"] = float("nan")
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Prepare a DFTpy vacancy spacing convergence package by reusing "
            "already-validated QE pristine/vacancy structures instead of "
            "regenerating the atomic geometry independently."
        )
    )
    ap.add_argument("--qe-pristine", required=True, help="Validated QE pristine VASP structure.")
    ap.add_argument("--qe-vacancy", required=True, help="Validated QE vacancy VASP structure.")
    ap.add_argument(
        "--outdir",
        default=str(ROOT / "results" / "dftpy_vacancy_convergence_from_qe_structures_20260522"),
    )
    ap.add_argument("--spacing-list", default="0.30,0.25,0.22,0.20,0.18")
    ap.add_argument("--pp", default=str(ROOT / "al.gga.recpot"))
    ap.add_argument("--kedf", default="TFVW")
    ap.add_argument("--fmax", type=float, default=0.002)
    ap.add_argument("--relax-steps", type=int, default=500)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pristine_source = Path(args.qe_pristine).expanduser().resolve()
    vacancy_source = Path(args.qe_vacancy).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    pp_path = Path(args.pp).expanduser().resolve()

    if not pristine_source.exists():
        raise FileNotFoundError(f"Missing QE pristine structure: {pristine_source}")
    if not vacancy_source.exists():
        raise FileNotFoundError(f"Missing QE vacancy structure: {vacancy_source}")
    if not pp_path.exists():
        raise FileNotFoundError(f"Missing pseudopotential: {pp_path}")

    pristine = read(str(pristine_source))
    vacancy = read(str(vacancy_source))
    n_pristine = int(len(pristine))
    n_vacancy = int(len(vacancy))
    if n_vacancy != n_pristine - 1:
        raise ValueError(
            f"Expected one vacancy: pristine has {n_pristine} atoms, vacancy has {n_vacancy} atoms."
        )

    spacing_values = _parse_float_list(args.spacing_list)
    outdir.mkdir(parents=True, exist_ok=True)

    top_manifest = {
        "workflow": "dftpy_vacancy_spacing_from_validated_qe_structures",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Avoid independent DFTpy structure generation after VESTA review; reuse validated QE geometry.",
        "qe_pristine_source": str(pristine_source),
        "qe_vacancy_source": str(vacancy_source),
        "pristine_geometry": _geometry_summary(pristine),
        "vacancy_geometry": _geometry_summary(vacancy),
        "grid_parameter": "spacing",
        "spacing_values_A": [float(v) for v in spacing_values],
        "ecut_analogue_eV": [spacing_angstrom_to_ecut_ev(v) for v in spacing_values],
        "kedf": str(args.kedf),
        "pp_file": str(pp_path),
        "fmax_eV_per_A": float(args.fmax),
        "relax_steps": int(args.relax_steps),
        "formation_energy_formula": "E_f^vac = E_vac^(N-1) - ((N-1)/N) E_pristine^(N)",
        "vesta_rule": "Open pristine_raw.vasp and vacancy_start.vasp before submitting production jobs.",
    }
    _write_text(outdir / "manifest.json", json.dumps(top_manifest, indent=2))

    settings = []
    for spacing in spacing_values:
        label = _spacing_label(float(spacing))
        settings.append(label)
        case_dir = outdir / "spacing_scan" / label
        case_dir.mkdir(parents=True, exist_ok=True)

        _write_structure_pair(case_dir / "pristine_raw", pristine)
        _write_structure_pair(case_dir / "vacancy_start", vacancy)
        _write_config_ini(
            case_dir / "dftpy_pristine_input.ini",
            pp_filename=pp_path.name,
            spacing_A=float(spacing),
            kedf=str(args.kedf),
            cellfile="pristine_raw.vasp",
        )
        _write_config_ini(
            case_dir / "dftpy_vacancy_input.ini",
            pp_filename=pp_path.name,
            spacing_A=float(spacing),
            kedf=str(args.kedf),
            cellfile="vacancy_start.vasp",
        )
        point_manifest = {
            "setting": label,
            "source": "validated QE structures",
            "spacing_A": float(spacing),
            "ecut_analogue_eV": spacing_angstrom_to_ecut_ev(float(spacing)),
            "pristine_n_atoms": n_pristine,
            "vacancy_n_atoms": n_vacancy,
            "pristine_source": str(pristine_source),
            "vacancy_source": str(vacancy_source),
            "pp_file": str(pp_path),
            "kedf": str(args.kedf),
            "fmax_eV_per_A": float(args.fmax),
            "relax_steps": int(args.relax_steps),
        }
        _write_text(case_dir / "point_manifest.json", json.dumps(point_manifest, indent=2))

    _write_text(outdir / "settings_spacing_scan.txt", "\n".join(settings) + "\n")
    _write_text(
        outdir / "README.md",
        "# DFTpy Vacancy Spacing From Validated QE Structures\n\n"
        "This package deliberately reuses QE pristine/vacancy structures instead of "
        "regenerating the DFTpy starting geometry. Open the copied VASP files in "
        "VESTA before submitting production jobs.\n",
    )
    print("============================================================")
    print("DFTpy vacancy spacing package prepared from QE structures")
    print("============================================================")
    print(f"Root: {outdir}")
    print(f"Settings: {outdir / 'settings_spacing_scan.txt'}")


if __name__ == "__main__":
    main()
