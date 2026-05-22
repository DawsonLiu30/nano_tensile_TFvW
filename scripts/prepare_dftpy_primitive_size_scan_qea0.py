from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import write


ROOT = Path(__file__).resolve().parents[1]
BOHR_TO_ANG = 0.529177210903
HA_TO_EV = 27.211386245988


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare a DFTpy primitive-supercell vacancy size scan at the QE bulk lattice constant.")
    ap.add_argument(
        "--outdir",
        default=str(ROOT / "results" / "dftpy_vacancy_size_primitive_qe_a0_spacing020_20260511"),
    )
    ap.add_argument("--a0", type=float, default=4.039825, help="QE bulk reference lattice constant in Angstrom.")
    ap.add_argument("--spacing", type=float, default=0.20, help="DFTpy real-space grid spacing in Angstrom.")
    ap.add_argument("--kedf", default="TFVW")
    ap.add_argument("--pp", default="al.gga.recpot", help="Portable relative pseudopotential path for remote runs.")
    ap.add_argument("--fmax", type=float, default=0.002)
    ap.add_argument("--relax-steps", type=int, default=500)
    ap.add_argument("--repeats", default="4,6,8,10,12,14")
    return ap.parse_args()


def _parse_int_list(text: str) -> list[int]:
    vals = sorted({int(tok.strip()) for tok in str(text).split(",") if tok.strip()})
    if not vals:
        raise ValueError("No repeats were provided.")
    if any(v <= 0 for v in vals):
        raise ValueError(f"Invalid repeat list: {vals}")
    return vals


def _spacing_to_ecut_like_ev(spacing_A: float) -> float:
    h_bohr = float(spacing_A) / BOHR_TO_ANG
    ecut_ha = (math.pi / h_bohr) ** 2 / 2.0
    return ecut_ha * HA_TO_EV


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_structure_pair(base: Path, atoms) -> None:
    write(str(base.with_suffix(".xyz")), atoms)
    write(str(base.with_suffix(".vasp")), atoms, direct=True, vasp5=True)


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


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    scan_dir = outdir / "size_scan"
    scan_dir.mkdir(parents=True, exist_ok=True)

    repeats = _parse_int_list(args.repeats)
    prim = bulk("Al", "fcc", a=float(args.a0), cubic=False)
    settings: list[str] = []

    top_manifest = {
        "workflow": "dftpy_vacancy_primitive_size_scan",
        "cell_basis": "primitive fcc",
        "a0_A": float(args.a0),
        "spacing_A": float(args.spacing),
        "ecut_analogue_eV": _spacing_to_ecut_like_ev(float(args.spacing)),
        "kedf": str(args.kedf),
        "pp_file": str(args.pp),
        "fmax_eV_per_A": float(args.fmax),
        "relax_steps": int(args.relax_steps),
        "repeats": repeats,
        "note": "This is the professor-requested primitive-cell supercell language, not an Angstrom-cubed label.",
    }
    _write_text(outdir / "scan_manifest.json", json.dumps(top_manifest, indent=2))

    for n in repeats:
        case_dir = scan_dir / f"prim_{n:02d}x{n:02d}x{n:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        pristine = prim.repeat((n, n, n))
        pristine.wrap()

        scaled = pristine.get_scaled_positions(wrap=True)
        target = np.array([0.5, 0.5, 0.5], dtype=float)
        diff = scaled - target[None, :]
        diff -= np.round(diff)
        remove_idx = int(np.argmin(np.sum(diff * diff, axis=1)))

        vacancy = pristine.copy()
        removed_scaled = scaled[remove_idx].copy()
        removed_cart = pristine.positions[remove_idx].copy()
        del vacancy[remove_idx]

        _write_structure_pair(case_dir / "pristine_raw", pristine)
        _write_structure_pair(case_dir / "vacancy_start", vacancy)
        _write_config_ini(
            case_dir / "dftpy_input.ini",
            pp_filename=Path(str(args.pp)).name,
            spacing_A=float(args.spacing),
            kedf=str(args.kedf),
        )

        manifest = {
            "setting": case_dir.name,
            "cell_basis": "primitive fcc",
            "a0_A": float(args.a0),
            "repeat_n": int(n),
            "pristine_n_atoms": int(len(pristine)),
            "vacancy_n_atoms": int(len(vacancy)),
            "volume_A3": float(pristine.get_volume()),
            "spacing_A": float(args.spacing),
            "ecut_analogue_eV": _spacing_to_ecut_like_ev(float(args.spacing)),
            "pp_file": str(args.pp),
            "kedf": str(args.kedf),
            "fmax_eV_per_A": float(args.fmax),
            "relax_steps": int(args.relax_steps),
            "removed_atom_index": int(remove_idx),
            "removed_atom_scaled": [float(v) for v in removed_scaled.tolist()],
            "removed_atom_cart_A": [float(v) for v in removed_cart.tolist()],
            "formation_energy_formula": "E_f^vac = E_vac^(N-1) - ((N-1)/N) E_pristine^(N)",
        }
        _write_text(case_dir / "point_manifest.json", json.dumps(manifest, indent=2))
        settings.append(case_dir.name)

    _write_text(outdir / "settings_size_scan.txt", "\n".join(settings) + "\n")
    print("============================================================")
    print("DFTpy primitive size scan prepared")
    print("============================================================")
    print(f"Root         : {outdir}")
    print(f"Settings file: {outdir / 'settings_size_scan.txt'}")


if __name__ == "__main__":
    main()
