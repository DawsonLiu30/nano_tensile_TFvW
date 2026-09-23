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

from prepare_qe_vacancy_vcrelax_3x3x3 import (  # noqa: E402
    DEFAULT_A0_A,
    cell_summary,
    parse_kmesh_list,
    write_vcrelax_input,
)


from divacancy_geometry import (
    build_centered_pristine, direction_label, enumerate_pairs, pair_geometry_metadata,
    parse_direction, parse_repeat, prepare_output_directory, remove_two_atoms,
    sha256_file, validate_positive,
)


def safe_distance_label(distance_a: float) -> str:
    return f"r{distance_a:.4f}A".replace(".", "p")


def write_structure_pair(base: Path, atoms) -> None:
    write(str(base.with_suffix(".vasp")), atoms, direct=True, vasp5=True)
    write(str(base.with_suffix(".xyz")), atoms)


def enumerate_same_height_pairs(pristine, center_index, z_tol=1e-6, distance_tol=1e-4,
                                direction=(1, 1, 0), direction_tol=1e-6):
    """Fixed-direction compatibility entry; use selection='shells' explicitly for shells."""
    return enumerate_pairs(pristine, center_index, direction=direction, z_tol=z_tol,
                           distance_tol=distance_tol, direction_tol=direction_tol)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_divacancy_group_job(
    path: Path,
    *,
    job_name: str,
    partition: str,
    ntasks: int,
    time_limit: str,
    mem: str,
) -> None:
    text = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --mem={mem}
#SBATCH --partition={partition}
#SBATCH --no-requeue
#SBATCH --account=MST114175

set -euo pipefail

module purge
module load intel/2021
module load intelmpi/2021.11

PWX="${{PWX:-/work/dawson666/q-e-qe-7.3.1/PW/src/pw.x}}"
QELIB="${{QELIB:-/home/dawson666/miniconda3/envs/abinit-env/lib}}"
export LD_LIBRARY_PATH="${{QELIB}}:${{LD_LIBRARY_PATH:-}}"
export LD_PRELOAD="${{QELIB}}/libgfortran.so.5.0.0${{LD_PRELOAD:+:$LD_PRELOAD}}"

if [ ! -x "$PWX" ]; then
  echo "[ERROR] QE binary not found: $PWX"
  exit 2
fi

run_qe() {{
  local folder="$1"
  if [ -s "$folder/vc-relax.out" ] && grep -q "JOB DONE" "$folder/vc-relax.out" && grep -Eq 'bfgs converged|End of BFGS Geometry Optimization' "$folder/vc-relax.out"; then
    echo "[SKIP] $folder already completed"
    return
  fi
  echo "[RUN] $folder"
  (
    cd "$folder"
    if [ -e vc-relax.out ] || [ -e tmp ] || [ -e CRASH ]; then
      archive="attempt_$(date -u +%Y%m%dT%H%M%SZ)_${{SLURM_JOB_ID:-local}}"
      mkdir "$archive"
      for item in vc-relax.out tmp CRASH; do
        [ ! -e "$item" ] || mv -- "$item" "$archive/"
      done
    fi
    mkdir -p tmp
    mpirun "$PWX" -in vc-relax.in > vc-relax.out
  )
}}

echo "[INFO] host=$(hostname) job=${{SLURM_JOB_ID}} pwx=$PWX"
run_qe pristine_vcrelax
run_qe divacancy_vcrelax
"""
    write_text(path, text)


def write_divacancy_array(
    path: Path,
    *,
    settings: list[str],
    max_parallel: int,
    partition: str,
    ntasks: int,
    time_limit: str,
    mem: str,
) -> None:
    settings_file = path.with_suffix(".settings")
    write_text(settings_file, "\n".join(settings) + "\n")
    text = f"""#!/bin/bash
#SBATCH --job-name=QEDivac
#SBATCH --output=logs_submit/%x_%A_%a.out
#SBATCH --error=logs_submit/%x_%A_%a.err
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --mem={mem}
#SBATCH --partition={partition}
#SBATCH --no-requeue
#SBATCH --account=MST114175
#SBATCH --array=0-{len(settings) - 1}%{max_parallel}

set -euo pipefail

ROOT="${{ROOT:-${{SLURM_SUBMIT_DIR:-$(pwd -P)}}}}"
SETTING=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${{ROOT}}/{settings_file.name}")
if [ -z "$SETTING" ] || [ ! -f "${{ROOT}}/${{SETTING}}/group_job.sh" ]; then
  echo "[ERROR] Invalid array setting: $SETTING" >&2
  exit 2
fi
mkdir -p "${{ROOT}}/logs_submit"
cd "${{ROOT}}/${{SETTING}}"
bash group_job.sh
"""
    write_text(path, text)


def prepare_case(
    root: Path,
    rel_setting: str,
    *,
    pristine,
    divacancy,
    ecut_ev: float,
    kmesh: tuple[int, int, int],
    pseudo_name: str,
    force_conv_eva: float,
    press_conv_kbar: float,
    partition: str,
    ntasks: int,
    time_limit: str,
    mem: str,
) -> None:
    setting_dir = root / rel_setting
    write_vcrelax_input(
        setting_dir / "pristine_vcrelax" / "vc-relax.in",
        prefix=f"Al_pristine_divac_{rel_setting.replace('/', '_')}",
        atoms=pristine,
        ecut_ev=ecut_ev,
        kmesh=kmesh,
        pseudo_name=pseudo_name,
        force_conv_eva=force_conv_eva,
        press_conv_kbar=press_conv_kbar,
    )
    write_vcrelax_input(
        setting_dir / "divacancy_vcrelax" / "vc-relax.in",
        prefix=f"Al_divac_{rel_setting.replace('/', '_')}",
        atoms=divacancy,
        ecut_ev=ecut_ev,
        kmesh=kmesh,
        pseudo_name=pseudo_name,
        force_conv_eva=force_conv_eva,
        press_conv_kbar=press_conv_kbar,
    )
    write_divacancy_group_job(
        setting_dir / "group_job.sh",
        job_name=f"QDV{rel_setting.split('/')[-1].replace('_', '')[:8]}",
        partition=partition,
        ntasks=ntasks,
        time_limit=time_limit,
        mem=mem,
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare QE/PBE vc-relax scan for same-height divacancy pairs in conventional fcc Al."
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--pseudo", required=True)
    parser.add_argument("--a0", type=float, default=DEFAULT_A0_A)
    parser.add_argument("--repeat", type=parse_repeat, default=(3, 3, 3))
    parser.add_argument("--ecut", type=float, default=800.0)
    parser.add_argument("--kmesh", default="3x3x3")
    parser.add_argument("--force-conv", type=float, default=0.002)
    parser.add_argument("--press-conv-kbar", type=float, default=0.5)
    parser.add_argument("--partition", default="ct56")
    parser.add_argument("--ntasks", type=int, default=28)
    parser.add_argument("--time-limit", default="4-00:00:00")
    parser.add_argument("--mem", default="128G")
    parser.add_argument("--max-parallel", type=int, default=5)
    parser.add_argument("--pair-selection", choices=["fixed_direction", "shells"], default="fixed_direction",
                        help="fixed_direction radial scan, or independent 3D FCC shell/direction representatives")
    parser.add_argument("--direction", type=parse_direction, default=(1, 1, 0))
    parser.add_argument("--direction-tol", type=float, default=1e-6)
    parser.add_argument("--z-tol", type=float, default=1.0e-6)
    parser.add_argument("--distance-tol", type=float, default=1.0e-4)
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        validate_positive(a0=args.a0, ecut=args.ecut, force_conv=args.force_conv,
                          press_conv_kbar=args.press_conv_kbar, ntasks=args.ntasks,
                          max_parallel=args.max_parallel, z_tol=args.z_tol,
                          distance_tol=args.distance_tol, direction_tol=args.direction_tol)
        meshes = parse_kmesh_list(args.kmesh)
        if len(meshes) != 1 or any(v <= 0 for v in meshes[0]):
            raise ValueError("kmesh must contain exactly one positive mesh, e.g. 3x3x3")
        if args.max_pairs < 0:
            raise ValueError("max-pairs must be nonnegative")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    pseudo = Path(args.pseudo).expanduser().resolve()
    if not pseudo.exists():
        raise FileNotFoundError(f"Missing pseudo: {pseudo}")
    repeat = args.repeat
    kmesh = parse_kmesh_list(args.kmesh)[0]
    pristine, center_index, shift_scaled = build_centered_pristine(args.a0, repeat)
    pairs = enumerate_pairs(pristine, center_index, selection=args.pair_selection,
                            direction=args.direction, z_tol=args.z_tol,
                            distance_tol=args.distance_tol, direction_tol=args.direction_tol)
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    if not pairs:
        raise ValueError(f"No pairs for selection={args.pair_selection}, direction={args.direction}, repeat={repeat}")
    prepare_output_directory(outdir)
    psp_dir = outdir / "psp"
    psp_dir.mkdir()
    pseudo_name = pseudo.name
    shutil.copy2(pseudo, psp_dir / pseudo_name)
    write_structure_pair(outdir / "pristine_start", pristine)

    settings: list[str] = []
    rows: list[dict[str, object]] = []
    pair_root = outdir / "pair_scan"
    pair_root.mkdir()
    n_pristine = len(pristine)

    for case_idx, (distance, second_index, delta) in enumerate(pairs, start=1):
        setting_name = f"pair_{case_idx:02d}_{safe_distance_label(distance)}"
        rel_setting = f"pair_scan/{setting_name}"
        case_dir = outdir / rel_setting
        divacancy = remove_two_atoms(pristine, center_index, second_index)
        geometry = pair_geometry_metadata(pristine, center_index, second_index, delta, args.a0)
        prepare_case(
            outdir,
            rel_setting,
            pristine=pristine,
            divacancy=divacancy,
            ecut_ev=args.ecut,
            kmesh=kmesh,
            pseudo_name=pseudo_name,
            force_conv_eva=args.force_conv,
            press_conv_kbar=args.press_conv_kbar,
            partition=args.partition,
            ntasks=args.ntasks,
            time_limit=args.time_limit,
            mem=args.mem,
        )
        write_structure_pair(case_dir / "pristine_start", pristine)
        write_structure_pair(case_dir / "divacancy_start", divacancy)
        manifest = {
            "setting": setting_name,
            "scan_type": "pair",
            "pair_selection": args.pair_selection,
            **geometry,
            "scan_point_index": case_idx,
            "requested_direction_indices": list(args.direction) if args.pair_selection == "fixed_direction" else None,
            "direction_tolerance_A": args.direction_tol,
            "pseudo_sha256": sha256_file(pseudo),
            "pseudo_source_at_preparation": str(pseudo),
            "generator_sha256": sha256_file(Path(__file__)),
            "geometry_helper_sha256": sha256_file(SCRIPT_DIR / "divacancy_geometry.py"),
            "code": "Quantum ESPRESSO",
            "functional": "PBE",
            "pseudo": pseudo_name,
            "relaxation": "vc-relax for pristine and divacancy",
            "a0_start_A": args.a0,
            "conventional_repeat": list(repeat),
            "cell_summary": cell_summary(pristine),
            "all_cell_lengths_exceed_10_A": all(float(x) > 10.0 for x in pristine.cell.lengths()),
            "pristine_n_atoms": n_pristine,
            "vacancy_n_atoms": len(divacancy),
            "vacancy_count": n_pristine - len(divacancy),
            "vacancy_concentration_percent": 100.0 * (n_pristine - len(divacancy)) / n_pristine,
            "first_vacancy_index": center_index,
            "second_vacancy_index": second_index,
            "pair_distance_A": distance,
            "pair_delta_A": [float(x) for x in delta],
            "ecut_eV": args.ecut,
            "kmesh": list(kmesh),
            "force_conv_eV_A": args.force_conv,
            "press_conv_kbar": args.press_conv_kbar,
            "formation_energy_formula": "E_f^2vac(r) = E_divac^(N-2,r) - ((N-2)/N) E_pristine^N",
            "per_vacancy_formula": "E_f^2vac(r) / 2",
        }
        (case_dir / "pair_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        settings.append(rel_setting)
        rows.append(
            {
                "case": setting_name,
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
                "input_files": "pristine_vcrelax/vc-relax.in; divacancy_vcrelax/vc-relax.in; pair_manifest.json",
                "expected_output_files": "pristine_vcrelax/vc-relax.out; divacancy_vcrelax/vc-relax.out",
                "ecut_eV": args.ecut,
                "kmesh": "x".join(str(x) for x in kmesh),
            }
        )

    write_divacancy_array(
        outdir / "submit_qe_divacancy_pair_array.sh",
        settings=settings,
        max_parallel=args.max_parallel,
        partition=args.partition,
        ntasks=args.ntasks,
        time_limit=args.time_limit,
        mem=args.mem,
    )
    with (outdir / "qe_divacancy_pair_plan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    top_manifest = {
        "workflow": "qe_pbe_al_divacancy_" + args.pair_selection + "_rscan",
        "pair_selection": args.pair_selection,
        "pair_direction_family": direction_label(args.direction) if args.pair_selection == "fixed_direction" else None,
        "pair_direction_indices": list(args.direction) if args.pair_selection == "fixed_direction" else None,
        "distance_convention": "initial minimum-image distance under PBC",
        "shell_note": "scan_point_index is not the FCC neighbour-shell index; see fcc_shell_index in each case",
        "method": {"code": "Quantum ESPRESSO", "functional": "PBE", "a0_start_A": args.a0,
                   "ecut_eV": args.ecut, "kmesh": list(kmesh), "pseudo": pseudo_name,
                   "pseudo_sha256": sha256_file(pseudo), "force_conv_eV_A": args.force_conv,
                   "press_conv_kbar": args.press_conv_kbar, "relaxation": "vc-relax"},
        "generator_sha256": sha256_file(Path(__file__)),
        "geometry_helper_sha256": sha256_file(SCRIPT_DIR / "divacancy_geometry.py"),
        "root": str(outdir),
        "pair_count": len(settings),
        "cell_lengths_A": [float(x) for x in pristine.cell.lengths()],
        "all_cell_lengths_exceed_10_A": all(float(x) > 10.0 for x in pristine.cell.lengths()),
        "settings": settings,
        "submit_command": f"cd {outdir} && sbatch submit_qe_divacancy_pair_array.sh",
    }
    (outdir / "manifest.json").write_text(json.dumps(top_manifest, indent=2), encoding="utf-8")
    source_dir = outdir / "preparation_sources"
    source_dir.mkdir()
    for source in (Path(__file__), SCRIPT_DIR / "divacancy_geometry.py", SCRIPT_DIR / "prepare_qe_vacancy_vcrelax_3x3x3.py"):
        shutil.copy2(source, source_dir / source.name)
    print(json.dumps(top_manifest, indent=2))


if __name__ == "__main__":
    main()
