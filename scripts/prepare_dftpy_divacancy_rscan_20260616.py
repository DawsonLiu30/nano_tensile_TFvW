from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

from ase.io import write


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from divacancy_geometry import (
    DFTPY_A0_A, build_centered_pristine, direction_label, enumerate_pairs,
    pair_geometry_metadata, parse_direction, parse_repeat, prepare_output_directory,
    remove_two_atoms, sha256_file, validate_positive,
)


def spacing_to_ecut_analogue_ev(spacing_a: float) -> float:
    # Same convention used by the existing DFTpy vacancy preparation scripts.
    return (math.pi / spacing_a) ** 2 * 3.80998212


def safe_distance_label(distance_a: float) -> str:
    return f"r{distance_a:.4f}A".replace(".", "p")


def write_structure_pair(base: Path, atoms) -> None:
    write(str(base.with_suffix(".vasp")), atoms, direct=True, vasp5=True)
    write(str(base.with_suffix(".xyz")), atoms)


def write_dftpy_provenance_input(
    path: Path,
    *,
    pp_filename: str,
    structure_filename: str,
    spacing_a: float,
    xc: str,
    kedf: str,
    kedf_x: float,
    kedf_y: float,
) -> None:
    """Write the human-readable equivalent of the programmatic calculator input."""

    path.write_text(
        f"""# Provenance input equivalent to the DftpyCalculator configuration.
# Ionic and cell relaxation is performed by ASE FrechetCellFilter + BFGS.

[JOB]
task = Optdensity
calctype = Energy Force Stress

[PATH]
pppath = ../../
cellpath = ./

[PP]
Al = {pp_filename}

[CELL]
cellfile = {structure_filename}
format = vasp

[GRID]
spacing = {spacing_a:.8f}

[EXC]
xc = {str(xc).strip().upper()}

[KEDF]
kedf = {kedf}
x = {kedf_x:.8f}
y = {kedf_y:.8f}

[OPT]
method = LBFGS
""",
        encoding="utf-8",
    )


def write_case_readme(
    path: Path,
    *,
    setting: str,
    pair_distance_a: float,
    pair_direction: tuple[int, int, int],
    n_pristine: int = 108,
    pair_selection: str = "fixed_direction",
) -> None:
    path.write_text(
        f"""DFTpy divacancy case: {setting}

Pair distance:
  {pair_distance_a:.8f} A (initial minimum-image distance under PBC)

Crystallographic direction (selection protocol: {pair_selection}):
  {direction_label(pair_direction)} (parallel and antiparallel sites are equivalent)

Starting structures:
  pristine_raw.vasp       {n_pristine}-atom pristine cell
  divacancy_start.vasp    {n_pristine - 2}-atom cell with two vacancies

DFTpy provenance inputs:
  dftpy_pristine_input.ini
  dftpy_divacancy_input.ini

Expected calculation outputs:
  pristine_dftpy.out
  divacancy_dftpy.out
  pristine_relax.log
  divacancy_relax.log
  pristine_vc_relaxed.vasp
  divacancy_vc_relaxed.vasp
  result.json

The .ini files document the DftpyCalculator settings. The ionic and cell
relaxation is driven programmatically through ASE FrechetCellFilter and BFGS.
""",
        encoding="utf-8",
    )


def enumerate_same_height_pairs(pristine, center_index, direction=(1, 1, 0),
                                z_tol=1e-6, distance_tol=1e-4, direction_tol=1e-6):
    """Compatibility wrapper; new code may call enumerate_pairs explicitly."""
    return enumerate_pairs(pristine, center_index, direction=direction, z_tol=z_tol,
                           distance_tol=distance_tol, direction_tol=direction_tol)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a DFTpy full atom+cell relaxation scan for same-height Al "
            "divacancy pairs along one fixed crystallographic direction."
        )
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--a0", type=float, default=DFTPY_A0_A)
    parser.add_argument("--repeat", type=parse_repeat, default=(3, 3, 3))
    parser.add_argument("--pp", default="al.lda.recpot")
    parser.add_argument("--xc", default="LDA")
    parser.add_argument("--kedf", default="TFVW")
    parser.add_argument("--kedf-x", type=float, default=0.9)
    parser.add_argument("--kedf-y", type=float, default=0.1)
    parser.add_argument("--spacing", type=float, default=0.20)
    parser.add_argument("--fmax", type=float, default=0.005)
    parser.add_argument("--relax-steps", type=int, default=5000)
    parser.add_argument(
        "--direction",
        type=parse_direction,
        default=(1, 1, 0),
        help="fixed cubic crystallographic direction (default: 1,1,0)",
    )
    parser.add_argument("--pair-selection", choices=["fixed_direction", "shells"], default="fixed_direction",
                        help="fixed_direction radial scan, or independent 3D FCC shell/direction representatives")
    parser.add_argument("--z-tol", type=float, default=1.0e-6)
    parser.add_argument("--distance-tol", type=float, default=1.0e-4)
    parser.add_argument(
        "--direction-tol",
        type=float,
        default=1.0e-6,
        help="maximum perpendicular displacement from the fixed direction in A",
    )
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--account", default="MST114175")
    parser.add_argument("--partition", default="ctest")
    parser.add_argument("--time-limit", default="02:00:00")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--mem", default="96G")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument(
        "--ase-optimizer",
        default="BFGS",
        choices=["BFGS", "LBFGS", "BFGSLineSearch", "SciPyFminBFGS", "SciPyFminCG", "MDMin"],
        help="ASE optimizer passed to the DFTpy vc-relax-equivalent runner.",
    )
    args = parser.parse_args(argv)
    try:
        validate_positive(a0=args.a0, spacing=args.spacing, fmax=args.fmax,
                          relax_steps=args.relax_steps, z_tol=args.z_tol,
                          distance_tol=args.distance_tol, direction_tol=args.direction_tol,
                          cpus=args.cpus, max_parallel=args.max_parallel)
        if args.max_pairs < 0:
            raise ValueError("max-pairs must be nonnegative")
        if any(not math.isfinite(v) or v < 0 for v in (args.kedf_x, args.kedf_y)):
            raise ValueError("KEDF weights must be finite and nonnegative")
        if args.kedf_x == args.kedf_y == 0:
            raise ValueError("both KEDF weights cannot be zero")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    pp_source = Path(args.pp).expanduser().resolve()
    if not pp_source.is_file():
        raise FileNotFoundError(f"Missing pseudopotential: {pp_source}")

    pristine, center_index, shift = build_centered_pristine(args.a0, args.repeat)
    pairs = enumerate_pairs(
        pristine, center_index, selection=args.pair_selection, direction=args.direction,
        z_tol=args.z_tol, distance_tol=args.distance_tol, direction_tol=args.direction_tol,
    )
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise RuntimeError(
            f"No same-height pairs found along {direction_label(args.direction)} "
            f"for repeat={args.repeat}"
        )

    prepare_output_directory(outdir)
    pp_path = outdir / pp_source.name
    shutil.copy2(pp_source, pp_path)
    pair_root = outdir / "pair_scan"
    pair_root.mkdir()
    write_structure_pair(outdir / "pristine_raw", pristine)
    write_structure_pair(outdir / "pristine_start", pristine)

    settings = []
    rows = []
    ecut_analogue = spacing_to_ecut_analogue_ev(args.spacing)
    n_pristine = len(pristine)

    for case_idx, (distance, second_index, delta) in enumerate(pairs, start=1):
        setting = f"pair_{case_idx:02d}_{safe_distance_label(distance)}"
        case_dir = pair_root / setting
        case_dir.mkdir(parents=True)
        divacancy = remove_two_atoms(pristine, center_index, second_index)
        geometry = pair_geometry_metadata(pristine, center_index, second_index, delta, args.a0)
        write_structure_pair(case_dir / "pristine_raw", pristine)
        write_structure_pair(case_dir / "divacancy_start", divacancy)
        write_dftpy_provenance_input(
            case_dir / "dftpy_pristine_input.ini",
            pp_filename=pp_path.name,
            structure_filename="pristine_raw.vasp",
            spacing_a=args.spacing,
            xc=args.xc,
            kedf=args.kedf,
            kedf_x=args.kedf_x,
            kedf_y=args.kedf_y,
        )
        write_dftpy_provenance_input(
            case_dir / "dftpy_divacancy_input.ini",
            pp_filename=pp_path.name,
            structure_filename="divacancy_start.vasp",
            spacing_a=args.spacing,
            xc=args.xc,
            kedf=args.kedf,
            kedf_x=args.kedf_x,
            kedf_y=args.kedf_y,
        )
        write_case_readme(
            case_dir / "README_CASE.txt",
            setting=setting,
            pair_distance_a=distance,
            pair_direction=tuple(geometry["pair_direction_indices"]),
            n_pristine=n_pristine,
            pair_selection=args.pair_selection,
        )

        manifest = {
            "setting": setting,
            "scan_type": "pair",
            "pair_selection": args.pair_selection,
            **geometry,
            "scan_point_index": case_idx,
            "requested_direction_indices": list(args.direction) if args.pair_selection == "fixed_direction" else None,
            "direction_tolerance_A": args.direction_tol,
            "cell_basis": "conventional cubic fcc",
            "a0_start_A": args.a0,
            "conventional_repeat": list(args.repeat),
            "conventional_repeat_label": "conv_" + "x".join(f"{v:02d}" for v in args.repeat),
            "cell_lengths_A": [float(x) for x in pristine.cell.lengths()],
            "cell_angles_deg": [float(x) for x in pristine.cell.angles()],
            "cell_volume_A3": float(pristine.cell.volume),
            "all_cell_lengths_exceed_10_A": all(float(x) > 10.0 for x in pristine.cell.lengths()),
            "pristine_n_atoms": n_pristine,
            "vacancy_n_atoms": len(divacancy),
            "vacancy_count": n_pristine - len(divacancy),
            "vacancy_concentration_percent": 100.0 * (n_pristine - len(divacancy)) / n_pristine,
            "first_vacancy_index": center_index,
            "second_vacancy_index": second_index,
            "pair_distance_A": distance,
            "pair_delta_A": [float(x) for x in delta],
            "spacing_A": args.spacing,
            "ecut_analogue_eV": ecut_analogue,
            "pp_file": "../../" + pp_path.name,
            "pp_sha256": sha256_file(pp_path),
            "pp_source_at_preparation": str(pp_source),
            "ase_optimizer": args.ase_optimizer,
            "generator_sha256": sha256_file(Path(__file__)),
            "geometry_helper_sha256": sha256_file(SCRIPT_DIR / "divacancy_geometry.py"),
            "xc": args.xc,
            "kedf": args.kedf,
            "kedf_x": args.kedf_x,
            "kedf_y": args.kedf_y,
            "fmax_eV_per_A": args.fmax,
            "relax_steps": args.relax_steps,
            "formation_energy_formula": "E_f^2vac(r) = E_divac^(N-2,r) - ((N-2)/N) E_pristine^N",
            "per_vacancy_formula": "E_f^2vac(r) / 2",
        }
        (case_dir / "point_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        settings.append(setting)
        rows.append(
            {
                "case": setting,
                "r_A": f"{distance:.8f}",
                "dx_A": f"{float(delta[0]):.8f}",
                "dy_A": f"{float(delta[1]):.8f}",
                "dz_A": f"{float(delta[2]):.8f}",
                "direction": geometry["pair_direction_family"],
                "fcc_shell_index": geometry["fcc_shell_index"],
                "pair_selection": args.pair_selection,
                "N_pristine": n_pristine,
                "N_divacancy": len(divacancy),
                "vacancy_count": n_pristine - len(divacancy),
                "vacancy_concentration_percent": f"{100.0 * (n_pristine - len(divacancy)) / n_pristine:.8f}",
                "source_dir": str(case_dir),
                "input_files": "pristine_raw.vasp; divacancy_start.vasp; dftpy_pristine_input.ini; dftpy_divacancy_input.ini; point_manifest.json",
                "expected_output_files": "pristine_dftpy.out; divacancy_dftpy.out; pristine_relax.log; divacancy_relax.log; result.json",
            }
        )

    (outdir / "settings_pair_scan.txt").write_text("\n".join(settings) + "\n", encoding="utf-8")
    with (outdir / "divacancy_pair_plan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    array_end = max(len(settings) - 1, 0)
    submit_text = f"""#!/bin/bash
#SBATCH -J dftpyDVac
#SBATCH -A {args.account}
#SBATCH -p {args.partition}
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c {args.cpus}
#SBATCH --mem={args.mem}
#SBATCH -t {args.time_limit}
#SBATCH --array=0-{array_end}%{args.max_parallel}
#SBATCH -o logs_ctest/%x_%A_%a.out
#SBATCH -e logs_ctest/%x_%A_%a.err

set -euo pipefail

ROOT="${{ROOT:-/work/dawson666/dftpy_project/relax/dftpy45}}"
SERIES_DIR="${{SERIES_DIR:-{outdir}}}"
ASE_OPTIMIZER="${{ASE_OPTIMIZER:-{args.ase_optimizer}}}"
SETTING_FILE="${{SERIES_DIR}}/settings_pair_scan.txt"

mkdir -p "${{ROOT}}/logs_ctest"
cd "${{ROOT}}"

source /home/dawson666/miniconda3/etc/profile.d/conda.sh
conda activate dftpy-env

export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-1}}"
export MKL_NUM_THREADS="${{MKL_NUM_THREADS:-1}}"
export OPENBLAS_NUM_THREADS="${{OPENBLAS_NUM_THREADS:-1}}"

SETTING=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SETTING_FILE")
if [ -z "$SETTING" ]; then
  echo "[ERROR] Empty setting from $SETTING_FILE"
  exit 1
fi

echo "[INFO] SERIES_DIR=$SERIES_DIR"
echo "[INFO] SETTING=$SETTING"
echo "[INFO] ASE_OPTIMIZER=$ASE_OPTIMIZER"
echo "[INFO] OMP_NUM_THREADS=$OMP_NUM_THREADS"

python scripts/run_dftpy_vcrelax_vacancy_one.py \\
  --rootdir "${{SERIES_DIR}}" \\
  --setting "$SETTING" \\
  --scan pair \\
  --ase-optimizer "$ASE_OPTIMIZER"
"""
    submit_path = outdir / "submit_dftpy_divacancy_pair_array.sh"
    submit_path.write_text(submit_text, encoding="utf-8")

    top_manifest = {
        "workflow": "dftpy_al_divacancy_" + args.pair_selection + "_rscan",
        "pair_selection": args.pair_selection,
        "distance_convention": "initial minimum-image distance under PBC",
        "shell_note": "scan_point_index is not the FCC neighbour-shell index; see fcc_shell_index in each case",
        "generator_sha256": sha256_file(Path(__file__)),
        "geometry_helper_sha256": sha256_file(SCRIPT_DIR / "divacancy_geometry.py"),
        "root": str(outdir),
        "pair_count": len(settings),
        "pair_direction_family": direction_label(args.direction) if args.pair_selection == "fixed_direction" else None,
        "pair_direction_indices": list(args.direction) if args.pair_selection == "fixed_direction" else None,
        "direction_tolerance_A": args.direction_tol,
        "settings_file": str(outdir / "settings_pair_scan.txt"),
        "submit_script": str(submit_path),
        "cell_lengths_A": [float(x) for x in pristine.cell.lengths()],
        "all_cell_lengths_exceed_10_A": all(float(x) > 10.0 for x in pristine.cell.lengths()),
        "method": {
            "code": "DFTpy",
            "xc": args.xc,
            "kedf": args.kedf,
            "kedf_x": args.kedf_x,
            "kedf_y": args.kedf_y,
            "pp": pp_path.name,
            "pp_sha256": sha256_file(pp_path),
            "a0_start_A": args.a0,
            "ase_optimizer": args.ase_optimizer,
            "spacing_A": args.spacing,
            "relaxation": "full atom+cell relaxation / vc-relax equivalent",
            "fmax_eV_A": args.fmax,
        },
    }
    (outdir / "manifest.json").write_text(json.dumps(top_manifest, indent=2), encoding="utf-8")

    source_dir = outdir / "preparation_sources"
    source_dir.mkdir()
    for source in (Path(__file__), SCRIPT_DIR / "divacancy_geometry.py"):
        shutil.copy2(source, source_dir / source.name)
    print(json.dumps(top_manifest, indent=2))


if __name__ == "__main__":
    main()
