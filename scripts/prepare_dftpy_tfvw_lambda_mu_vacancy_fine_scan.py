from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_dftpy_vacancy_conventional import (  # noqa: E402
    _build_conventional_pair,
    _repeat_label,
    _write_case,
)


def float_list(text: str) -> list[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required")
    return values


def token(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a fine DFTpy TFvW lambda/mu single-vacancy scan.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--pp", required=True)
    parser.add_argument("--a0", type=float, default=4.039848)
    parser.add_argument("--repeat", default="3x3x3")
    parser.add_argument("--lambda-list", type=float_list, default=float_list("0.90,0.91,0.92,0.93,0.94,0.95"))
    parser.add_argument("--mu-list", type=float_list, default=float_list("0.04,0.05,0.06,0.07,0.08,0.09,0.10"))
    parser.add_argument("--spacing", type=float, default=0.20)
    parser.add_argument("--xc", default="LDA")
    parser.add_argument("--kedf", default="TFVW")
    parser.add_argument("--fmax", type=float, default=0.002)
    parser.add_argument("--relax-steps", type=int, default=5000)
    parser.add_argument("--account", default="MST114175")
    parser.add_argument("--partition", default="ctest")
    parser.add_argument("--time-limit", default="02:00:00")
    parser.add_argument("--mem", default="96G")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--qe-ef-reference", type=float, default=0.601167)
    parser.add_argument("--lattice-reference", type=float, default=4.039848)
    return parser.parse_args()


def parse_repeat(text: str) -> tuple[int, int, int]:
    values = tuple(int(value) for value in text.lower().replace(",", "x").split("x"))
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError("repeat must look like 3x3x3")
    return values


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    pp_path = Path(args.pp).expanduser().resolve()
    if not pp_path.exists():
        raise FileNotFoundError(pp_path)
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)

    repeat = parse_repeat(args.repeat)
    pristine, vacancy, removed = _build_conventional_pair(a0_A=args.a0, repeat=repeat)
    settings: list[str] = []

    for lambda_tf in args.lambda_list:
        for mu_vw in args.mu_list:
            setting = f"tfvw_lam{token(lambda_tf)}_mu{token(mu_vw)}"
            settings.append(setting)
            _write_case(
                outdir / "weight_scan" / setting,
                pristine=pristine,
                vacancy=vacancy,
                pp_path=pp_path,
                spacing_A=args.spacing,
                kedf=args.kedf,
                fmax=args.fmax,
                relax_steps=args.relax_steps,
                extra_manifest={
                    "setting": setting,
                    "scan_type": "weight",
                    "scan_purpose": "joint_lambda_mu_fine_calibration",
                    "conventional_repeat": list(repeat),
                    "conventional_repeat_label": _repeat_label(repeat),
                    "xc": args.xc.upper(),
                    "kedf_x": lambda_tf,
                    "kedf_y": mu_vw,
                    **removed,
                },
            )

    (outdir / "settings_weight_scan.txt").write_text("\n".join(settings) + "\n", encoding="utf-8")
    array_end = len(settings) - 1
    submit = f"""#!/bin/bash
#SBATCH --job-name=dftpyLMfine
#SBATCH --output=logs_ctest/%x_%A_%a.out
#SBATCH --error=logs_ctest/%x_%A_%a.err
#SBATCH --time={args.time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={args.mem}
#SBATCH --partition={args.partition}
#SBATCH --no-requeue
#SBATCH --account={args.account}
#SBATCH --array=0-{array_end}%{args.max_parallel}

set -euo pipefail

ROOT="${{ROOT:-/work/dawson666/dftpy_project/relax/dftpy45}}"
SERIES_NAME="${{SERIES_NAME:-{outdir.name}}}"
SETTING_FILE="${{ROOT}}/results/${{SERIES_NAME}}/settings_weight_scan.txt"

source /home/dawson666/miniconda3/etc/profile.d/conda.sh
conda activate dftpy-env

export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p "${{ROOT}}/logs_ctest"
cd "${{ROOT}}"
SETTING=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SETTING_FILE")
echo "[INFO] SERIES_NAME=$SERIES_NAME SETTING=$SETTING"

python scripts/run_dftpy_vcrelax_vacancy_one.py \\
  --rootdir "${{ROOT}}/results/${{SERIES_NAME}}" \\
  --setting "$SETTING" \\
  --scan weight \\
  --ase-optimizer BFGS
"""
    (outdir / "submit_dftpy_lambda_mu_fine_array.sh").write_text(submit, encoding="utf-8")

    manifest = {
        "workflow": "dftpy_tfvw_lambda_mu_single_vacancy_fine_scan",
        "status": "prepared",
        "cell": "conventional cubic fcc 3x3x3",
        "N_pristine": len(pristine),
        "N_vacancy": len(vacancy),
        "a0_start_A": args.a0,
        "lambda_tf_values": args.lambda_list,
        "mu_vw_values": args.mu_list,
        "point_count": len(settings),
        "spacing_A": args.spacing,
        "xc": args.xc.upper(),
        "kedf": args.kedf,
        "target_fmax_eV_A": args.fmax,
        "qe_vacancy_formation_reference_eV": args.qe_ef_reference,
        "lattice_reference_A": args.lattice_reference,
        "selection_rule": "Evaluate formation energy, pristine lattice constant, force/stress quality jointly.",
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (outdir / "README.md").write_text(
        "# DFTpy TFvW lambda/mu fine scan\n\n"
        "This is a single-vacancy calibration scan prepared after the complete 10x10 coarse matrix. "
        "It refines the boundary region where both the QE vacancy formation energy and the lattice "
        "constant may be matched. Lambda and mu are independent coefficients.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

