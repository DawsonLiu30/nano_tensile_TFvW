from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

from ase.build import bulk
from ase.io import write

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.aluminum_defaults import AL_FCC_A0_TFVW_ANG


HA_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903


def ecut_to_spacing_angstrom(ecut_ev: float) -> float:
    ecut_ha = float(ecut_ev) / HA_TO_EV
    h_bohr = math.pi / math.sqrt(2.0 * ecut_ha)
    return h_bohr * BOHR_TO_ANG


def spacing_angstrom_to_ecut_ev(spacing_A: float) -> float:
    h_bohr = float(spacing_A) / BOHR_TO_ANG
    ecut_ha = (math.pi / h_bohr) ** 2 / 2.0
    return ecut_ha * HA_TO_EV


def _parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for chunk in str(text).split(","):
        token = chunk.strip()
        if token:
            values.append(float(token))
    if not values:
        raise ValueError("No valid floating-point values were provided.")
    return values


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_structure_pair(base: Path, atoms) -> None:
    write(str(base.with_suffix(".xyz")), atoms)
    write(str(base.with_suffix(".vasp")), atoms, direct=True, vasp5=True)


def _spacing_label(spacing_A: float) -> str:
    return f"spacing_{spacing_A:.2f}A".replace(".", "p")


def _write_config_ini(path: Path, *, pp_filename: str, spacing_A: float, kedf: str) -> None:
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
cellfile = pristine_raw.vasp
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


def _choose_vacancy_index(atoms) -> tuple[int, dict[str, float]]:
    pos = atoms.get_positions()
    cell = atoms.get_cell().array
    center = 0.5 * (cell[0] + cell[1] + cell[2])
    dists = ((pos - center[None, :]) ** 2).sum(axis=1) ** 0.5
    idx = int(dists.argmin())
    return idx, {
        "index": idx,
        "x_A": float(pos[idx, 0]),
        "y_A": float(pos[idx, 1]),
        "z_A": float(pos[idx, 2]),
        "distance_to_center_A": float(dists[idx]),
    }


def _latest_bulk_summary_path() -> Path:
    candidates = sorted(
        (ROOT / "results").glob("bulk_Al_fcc_TFVW*/summary.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No bulk TFvW summary.txt was found under results/.")
    return candidates[0]


def _read_a0_from_summary(path: Path) -> float:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("a0_ref_A="):
            return float(line.split("=", 1)[1].strip())
    raise ValueError(f"Could not find a0_ref_A in {path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare a DFTpy primitive-cell vacancy convergence workflow for direct comparison to QE.")
    ap.add_argument("--outdir", default=str(ROOT / "results" / "dftpy_vacancy_convergence_primitive4_20260508"))
    ap.add_argument("--supercell-n", type=int, default=4, help="Primitive-cell repetition N for the comparison supercell.")
    ap.add_argument("--spacing-list", default="0.30,0.25,0.22,0.20,0.18", help="Comma-separated DFTpy grid spacings in Angstrom.")
    ap.add_argument("--ecut-list", default="", help="Optional comma-separated ecut-like grid labels in eV. Used only if spacing-list is empty.")
    ap.add_argument("--a0", type=float, default=None, help="Manual lattice constant override in Angstrom.")
    ap.add_argument("--bulk-summary", default="", help="Optional TFvW bulk summary.txt used to read a0_ref_A.")
    ap.add_argument("--pp", default=str(ROOT / "al.gga.recpot"))
    ap.add_argument("--kedf", default="TFVW")
    ap.add_argument("--fmax", type=float, default=0.002)
    ap.add_argument("--relax-steps", type=int, default=200)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    pp_path = Path(args.pp).expanduser().resolve()
    if not pp_path.exists():
        raise FileNotFoundError(f"Pseudopotential not found: {pp_path}")

    bulk_summary = _latest_bulk_summary_path() if not str(args.bulk_summary).strip() else Path(args.bulk_summary).expanduser().resolve()
    a0_A = float(args.a0) if args.a0 is not None else _read_a0_from_summary(bulk_summary)
    if str(args.spacing_list).strip():
        spacing_values = _parse_float_list(args.spacing_list)
        ecut_values = [spacing_angstrom_to_ecut_ev(v) for v in spacing_values]
    else:
        ecut_values = _parse_float_list(args.ecut_list)
        spacing_values = [ecut_to_spacing_angstrom(v) for v in ecut_values]
    n = int(args.supercell_n)

    pristine = bulk("Al", "fcc", a=float(a0_A), cubic=False).repeat((n, n, n))
    pristine_n_atoms = int(len(pristine))
    vacancy_index, vacancy_site = _choose_vacancy_index(pristine)
    vacancy_start = pristine.copy()
    del vacancy_start[vacancy_index]

    top_manifest = {
        "workflow": "dftpy_vacancy_convergence_primitive",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "comparison_target": "QE primitive 4x4x4 vacancy workflow",
        "cell_basis": "primitive",
        "supercell_n": n,
        "pristine_n_atoms": pristine_n_atoms,
        "vacancy_n_atoms": pristine_n_atoms - 1,
        "a0_ref_A": float(a0_A),
        "pp_file": str(pp_path),
        "kedf": str(args.kedf),
        "grid_parameter": "spacing",
        "spacing_values_A": [float(v) for v in spacing_values],
        "ecut_values_eV": [float(v) for v in ecut_values],
        "fmax_eV_per_A": float(args.fmax),
        "relax_steps": int(args.relax_steps),
        "note": "DFTpy has no Brillouin-zone k-point sampling; the primary numerical parameter tested here is the real-space grid spacing. ecut_values_eV are stored only as eV-like reference labels.",
    }
    _write_text(outdir / "manifest.json", json.dumps(top_manifest, indent=2))

    readme = f"""# DFTpy primitive 4x4x4 vacancy convergence package

Purpose:

- supplement the QE vacancy line with a DFTpy result on the same primitive-cell supercell geometry
- test the DFTpy real-space grid parameter **spacing**
- keep an eV-like label only as a convenience when comparing the grid density to QE-style cutoff language
- keep the large primitive 8x8x8 to 14x14x14 size-convergence line separate

Important:

- this package answers the professor's question: "what is the DFTpy vacancy result on the corresponding comparison cell?"
- DFTpy has **no k-point sampling**
- according to the official DFTpy OF-DFT tutorial, the numerical grid parameter is `[GRID] spacing`
- in this package, the primary sweep is `spacing_scan/spacing_XXA`
- an eV-like reference value is still stored in `point_manifest.json` as `ecut_analogue_eV`
- a provenance `dftpy_input.ini` is written for each setting, while the actual ionic relaxation is performed through the official ASE/DFTpyCalculator relaxation route
- for large-size DFTpy supercells, use the already-prepared folder:
  `results/bulk_fcc_vacancy_primitive_8to14_20260506`
"""
    _write_text(outdir / "README.md", readme)

    run_all_lines = [
        "# Suggested sequential runs on iservice",
        "# (run one setting at a time, then collect after all completed)",
        "",
    ]

    for spacing, ecut in zip(spacing_values, ecut_values):
        label = _spacing_label(float(spacing))
        case_dir = outdir / "spacing_scan" / label
        case_dir.mkdir(parents=True, exist_ok=True)

        _write_structure_pair(case_dir / "pristine_raw", pristine)
        _write_structure_pair(case_dir / "vacancy_start", vacancy_start)
        _write_config_ini(case_dir / "dftpy_input.ini", pp_filename=pp_path.name, spacing_A=float(spacing), kedf=str(args.kedf))

        point_manifest = {
            "setting": label,
            "grid_parameter": "spacing",
            "spacing_A": float(spacing),
            "ecut_analogue_eV": float(ecut),
            "cell_basis": "primitive",
            "supercell_n": n,
            "a0_A": float(a0_A),
            "pristine_n_atoms": pristine_n_atoms,
            "vacancy_n_atoms": pristine_n_atoms - 1,
            "vacancy_site": vacancy_site,
            "pp_file": str(pp_path),
            "kedf": str(args.kedf),
            "fmax_eV_per_A": float(args.fmax),
            "relax_steps": int(args.relax_steps),
            "formation_energy_formula": "E_f^vac = E_vac^(N-1) - ((N-1)/N) E_pristine^(N)",
            "expected_outputs": [
                "pristine_dftpy.out",
                "vacancy_dftpy.out",
                "vacancy_relax.log",
                "vacancy_relax.traj",
                "vacancy_relaxed.xyz",
                "vacancy_relaxed.vasp",
                "result.json",
            ],
        }
        _write_text(case_dir / "point_manifest.json", json.dumps(point_manifest, indent=2))

        run_text = f"""# Run this setting
# Primary DFTpy parameter: spacing_A stored in point_manifest.json
python scripts/run_dftpy_vacancy_convergence.py ^
  --rootdir {outdir} ^
  --setting {label}
"""
        _write_text(case_dir / "RUN_COMMAND.txt", run_text)
        run_all_lines.append(f"python scripts/run_dftpy_vacancy_convergence.py --rootdir {outdir} --setting {label}")

    run_all_lines.extend(
        [
            "",
            "# Collect after all settings finish",
            f"python scripts/collect_dftpy_vacancy_convergence.py --rootdir {outdir}",
        ]
    )
    _write_text(outdir / "RUN_ALL_COMMANDS.txt", "\n".join(run_all_lines) + "\n")

    print(f"[prepare-dftpy-vacancy] Prepared workflow under: {outdir}")


if __name__ == "__main__":
    main()
