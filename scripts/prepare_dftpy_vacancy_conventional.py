from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import write
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


def _parse_int_list(text: str) -> list[int]:
    values = sorted({int(chunk.strip()) for chunk in str(text).split(",") if chunk.strip()})
    if not values:
        raise ValueError("No conventional repeat values were provided.")
    if any(v <= 0 for v in values):
        raise ValueError(f"Repeat values must be positive: {values}")
    return values


def _parse_repeat(text: str) -> tuple[int, int, int]:
    token = str(text).strip().lower().replace(",", "x")
    parts = token.split("x")
    if len(parts) == 1:
        n = int(parts[0])
        repeat = (n, n, n)
    elif len(parts) == 3:
        repeat = tuple(int(part.strip()) for part in parts)
    else:
        raise ValueError(f"Invalid conventional repeat: {text}")
    if any(value <= 0 for value in repeat):
        raise ValueError(f"Repeat values must be positive: {text}")
    return repeat


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_structure_pair(base: Path, atoms) -> None:
    write(str(base.with_suffix(".xyz")), atoms)
    write(str(base.with_suffix(".vasp")), atoms, direct=True, vasp5=True)


def _spacing_label(spacing_A: float) -> str:
    return f"spacing_{spacing_A:.2f}A".replace(".", "p")


def _conv_label(n: int) -> str:
    return f"conv_{int(n):02d}x{int(n):02d}x{int(n):02d}"


def _repeat_label(repeat: tuple[int, int, int]) -> str:
    return f"conv_{repeat[0]:02d}x{repeat[1]:02d}x{repeat[2]:02d}"


def _geometry_summary(atoms) -> dict[str, float | int | str]:
    atoms = atoms.copy()
    atoms.pbc = [True, True, True]
    lengths = atoms.cell.lengths()
    angles = atoms.cell.angles()
    summary: dict[str, float | int | str] = {
        "n_atoms": int(len(atoms)),
        "formula": atoms.get_chemical_formula(),
        "volume_A3": float(atoms.get_volume()),
        "a_A": float(lengths[0]),
        "b_A": float(lengths[1]),
        "c_A": float(lengths[2]),
        "alpha_deg": float(angles[0]),
        "beta_deg": float(angles[1]),
        "gamma_deg": float(angles[2]),
    }
    if len(atoms) > 1:
        _, _, d = neighbor_list("ijd", atoms, cutoff=4.2)
        summary["min_distance_A"] = float(d.min()) if len(d) else float("nan")
    else:
        summary["min_distance_A"] = float("nan")
    z = sorted(float(v) % float(lengths[2]) for v in atoms.positions[:, 2])
    if z:
        gaps = [z[k + 1] - z[k] for k in range(len(z) - 1)]
        gaps.append(float(lengths[2]) - z[-1] + z[0])
        summary["max_z_gap_A"] = float(max(gaps))
    else:
        summary["max_z_gap_A"] = float("nan")
    return summary


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


def _build_conventional_pair(*, a0_A: float, repeat: tuple[int, int, int]):
    conventional = bulk("Al", "fcc", a=float(a0_A), cubic=True)
    pristine = conventional.repeat(tuple(int(value) for value in repeat))
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
    vacancy.wrap()

    return pristine, vacancy, {
        "removed_atom_index": int(remove_idx),
        "removed_atom_scaled": [float(v) for v in removed_scaled.tolist()],
        "removed_atom_cart_A": [float(v) for v in removed_cart.tolist()],
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Prepare VESTA-friendly conventional cubic fcc Al vacancy structures "
            "for DFTpy spacing and optional supercell-size convergence."
        )
    )
    ap.add_argument(
        "--outdir",
        default=str(ROOT / "results" / "dftpy_vacancy_conventional_qe_a0_20260522"),
    )
    ap.add_argument("--a0", type=float, default=4.039825, help="Conventional cubic fcc lattice constant in Angstrom.")
    ap.add_argument("--spacing-list", default="0.30,0.25,0.22,0.20,0.18")
    ap.add_argument(
        "--spacing-repeat",
        default="4",
        help=(
            "Conventional repeat used for the spacing scan. Use 4 for cubic "
            "4x4x4 = 256/255 atoms, or 2x2x4 for the QE-matched 64/63 cell."
        ),
    )
    ap.add_argument(
        "--size-repeats",
        default="",
        help="Optional conventional repeats for size scan, e.g. 2,3,4,5. Leave empty to skip.",
    )
    ap.add_argument("--pp", default=str(ROOT / "al.gga.recpot"))
    ap.add_argument("--kedf", default="TFVW")
    ap.add_argument("--fmax", type=float, default=0.002)
    ap.add_argument("--relax-steps", type=int, default=500)
    return ap.parse_args()


def _write_case(
    case_dir: Path,
    *,
    pristine,
    vacancy,
    pp_path: Path,
    spacing_A: float,
    kedf: str,
    fmax: float,
    relax_steps: int,
    extra_manifest: dict[str, object],
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_structure_pair(case_dir / "pristine_raw", pristine)
    _write_structure_pair(case_dir / "vacancy_start", vacancy)
    _write_config_ini(
        case_dir / "dftpy_pristine_input.ini",
        pp_filename=pp_path.name,
        spacing_A=float(spacing_A),
        kedf=str(kedf),
        cellfile="pristine_raw.vasp",
    )
    _write_config_ini(
        case_dir / "dftpy_vacancy_input.ini",
        pp_filename=pp_path.name,
        spacing_A=float(spacing_A),
        kedf=str(kedf),
        cellfile="vacancy_start.vasp",
    )
    manifest = {
        **extra_manifest,
        "cell_basis": "conventional cubic fcc",
        "pristine_n_atoms": int(len(pristine)),
        "vacancy_n_atoms": int(len(vacancy)),
        "spacing_A": float(spacing_A),
        "ecut_analogue_eV": spacing_angstrom_to_ecut_ev(float(spacing_A)),
        "pp_file": str(pp_path),
        "kedf": str(kedf),
        "fmax_eV_per_A": float(fmax),
        "relax_steps": int(relax_steps),
        "formation_energy_formula": "E_f^vac = E_vac^(N-1) - ((N-1)/N) E_pristine^(N)",
        "pristine_geometry": _geometry_summary(pristine),
        "vacancy_geometry": _geometry_summary(vacancy),
    }
    _write_text(case_dir / "point_manifest.json", json.dumps(manifest, indent=2))


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    pp_path = Path(args.pp).expanduser().resolve()
    if not pp_path.exists():
        raise FileNotFoundError(f"Missing pseudopotential: {pp_path}")

    spacing_values = _parse_float_list(args.spacing_list)
    outdir.mkdir(parents=True, exist_ok=True)

    spacing_repeat = _parse_repeat(args.spacing_repeat)
    pristine, vacancy, removed = _build_conventional_pair(a0_A=float(args.a0), repeat=spacing_repeat)
    spacing_settings = []
    for spacing in spacing_values:
        label = _spacing_label(spacing)
        spacing_settings.append(label)
        _write_case(
            outdir / "spacing_scan" / label,
            pristine=pristine,
            vacancy=vacancy,
            pp_path=pp_path,
            spacing_A=float(spacing),
            kedf=str(args.kedf),
            fmax=float(args.fmax),
            relax_steps=int(args.relax_steps),
            extra_manifest={
                "setting": label,
                "scan_type": "spacing",
                "conventional_repeat": list(spacing_repeat),
                "conventional_repeat_label": _repeat_label(spacing_repeat),
                **removed,
            },
        )
    _write_text(outdir / "settings_spacing_scan.txt", "\n".join(spacing_settings) + "\n")

    size_settings = []
    size_repeats = _parse_int_list(args.size_repeats) if str(args.size_repeats).strip() else []
    for n in size_repeats:
        repeat_n = (int(n), int(n), int(n))
        pristine_n, vacancy_n, removed_n = _build_conventional_pair(a0_A=float(args.a0), repeat=repeat_n)
        label = _conv_label(n)
        size_settings.append(label)
        _write_case(
            outdir / "size_scan" / label,
            pristine=pristine_n,
            vacancy=vacancy_n,
            pp_path=pp_path,
            spacing_A=0.20,
            kedf=str(args.kedf),
            fmax=float(args.fmax),
            relax_steps=int(args.relax_steps),
            extra_manifest={
                "setting": label,
                "scan_type": "size",
                "conventional_repeat": list(repeat_n),
                "conventional_repeat_label": _repeat_label(repeat_n),
                **removed_n,
            },
        )
    if size_settings:
        _write_text(outdir / "settings_size_scan.txt", "\n".join(size_settings) + "\n")

    top_manifest = {
        "workflow": "dftpy_vacancy_conventional_cubic",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Use VESTA-friendly conventional cubic fcc supercells instead of rhombohedral primitive cells.",
        "cell_basis": "conventional cubic fcc",
        "a0_A": float(args.a0),
        "spacing_repeat": list(spacing_repeat),
        "spacing_repeat_label": _repeat_label(spacing_repeat),
        "spacing_scan_atoms": {
            "pristine": int(len(pristine)),
            "vacancy": int(len(vacancy)),
            "rule": "N_pristine = 4 * nx * ny * nz for conventional fcc repeats",
        },
        "spacing_values_A": [float(v) for v in spacing_values],
        "size_repeats": [int(v) for v in size_repeats],
        "kedf": str(args.kedf),
        "pp_file": str(pp_path),
        "fmax_eV_per_A": float(args.fmax),
        "relax_steps": int(args.relax_steps),
        "vesta_rule": "Open spacing_scan/spacing_0p20A/pristine_raw.vasp and vacancy_start.vasp before submitting.",
        "note": "This intentionally changes the comparison cell from primitive 4x4x4 (64/63 atoms, 60-degree rhombohedral cell) to conventional 4x4x4 (256/255 atoms, 90-degree cubic cell).",
    }
    _write_text(outdir / "manifest.json", json.dumps(top_manifest, indent=2))
    _write_text(
        outdir / "README.md",
        "# DFTpy Conventional Cubic Vacancy Convergence\n\n"
        "This package uses conventional cubic fcc Al cells for VESTA-friendly "
        "structure inspection. The default spacing scan uses conventional "
        "4x4x4 = 256 pristine atoms and 255 vacancy atoms.\n",
    )

    print("============================================================")
    print("DFTpy conventional cubic vacancy package prepared")
    print("============================================================")
    print(f"Root: {outdir}")
    print(f"Spacing settings: {outdir / 'settings_spacing_scan.txt'}")
    if size_settings:
        print(f"Size settings   : {outdir / 'settings_size_scan.txt'}")
    print(f"Pristine atoms  : {len(pristine)}")
    print(f"Vacancy atoms   : {len(vacancy)}")


if __name__ == "__main__":
    main()
