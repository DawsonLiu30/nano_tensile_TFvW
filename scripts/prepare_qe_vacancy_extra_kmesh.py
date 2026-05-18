from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Clone an existing QE vacancy k-mesh case and retarget it to a denser mesh."
    )
    ap.add_argument("--rootdir", required=True, help="QE vacancy workflow root, e.g. qe_vacancy_convergence_20260506")
    ap.add_argument("--source-tag", default="05x05x05", help="Source mesh tag without leading k_, e.g. 05x05x05")
    ap.add_argument("--target-tag", default="06x06x06", help="Target mesh tag without leading k_, e.g. 06x06x06")
    ap.add_argument("--partition", default="ct56")
    ap.add_argument("--ntasks", type=int, default=28)
    ap.add_argument("--time-limit", default="24:00:00")
    ap.add_argument("--mem", default="96G")
    ap.add_argument("--account", default="MST114175")
    ap.add_argument("--job-name", default="QV56K060")
    ap.add_argument("--force", action="store_true", help="Replace the target case if it already exists.")
    return ap.parse_args()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _target_k_triplet(tag: str) -> tuple[int, int, int]:
    nums = [int(x) for x in str(tag).split("x")]
    if len(nums) != 3:
        raise ValueError(f"Invalid mesh tag: {tag}")
    return nums[0], nums[1], nums[2]


def _retarget_qe_input(path: Path, *, tag: str, prefix: str) -> None:
    k1, k2, k3 = _target_k_triplet(tag)
    text = _read(path)
    text = re.sub(
        r"K_POINTS automatic\s*\n\s*\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+",
        f"K_POINTS automatic\n{k1} {k2} {k3}  0 0 0",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"prefix\s*=\s*'[^']+'", f"prefix = '{prefix}'", text)
    _write(path, text)


def _cleanup_case(case_dir: Path) -> None:
    for rel in [
        "pristine_scf/scf.out",
        "vacancy_relax/relax.out",
        "pristine_scf/CRASH",
        "vacancy_relax/CRASH",
        "slurm_ct56.out",
        "slurm_ct56.err",
        "slurm_%j.out",
        "slurm_%j.err",
        "group_job_ct56_long.sh",
    ]:
        path = case_dir / rel
        if path.exists():
            path.unlink()
    for pattern in ["slurm*.out", "slurm*.err", "group_job_fixed_ct56*.sh"]:
        for path in case_dir.glob(pattern):
            if path.is_file():
                path.unlink()
    for sub in [case_dir / "pristine_scf" / "tmp", case_dir / "vacancy_relax" / "tmp"]:
        shutil.rmtree(sub, ignore_errors=True)


def _write_group_job(
    path: Path,
    *,
    job_name: str,
    partition: str,
    ntasks: int,
    time_limit: str,
    mem: str,
    account: str,
) -> None:
    text = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --account={account}
#SBATCH --output=slurm_ct56_%j.out
#SBATCH --error=slurm_ct56_%j.err

set -euo pipefail

module purge
module load intel/2021 intelmpi/2021.11

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PWX="${{PWX:-/gpfs-home/dawson666/q-e-qe-7.3.1/bin/pw.x}}"

echo "[INFO] Host      : $(hostname)"
echo "[INFO] Workdir   : $(pwd)"
echo "[INFO] Job ID    : $SLURM_JOB_ID"
echo "[INFO] Job Name  : $SLURM_JOB_NAME"
echo "[INFO] Start     : $(date)"
echo "[INFO] PWX       : $PWX"

if [ ! -x "$PWX" ]; then
    echo "[ERROR] pw.x not found or not executable: $PWX"
    exit 1
fi

run_qe () {{
    local subdir="$1"
    local input="$2"
    local output="$3"
    echo "============================================================"
    echo "[TASK] $subdir / $input"
    if [ -f "$subdir/$output" ] && grep -q "JOB DONE" "$subdir/$output"; then
        echo "[SKIP] $subdir already completed."
        return 0
    fi
    (
        cd "$subdir"
        rm -f CRASH
        rm -rf tmp
        mkdir -p tmp
        mpirun -np ${{SLURM_NTASKS}} "$PWX" -in "$input" > "$output"
    )
}}

run_qe pristine_scf scf.in scf.out
run_qe vacancy_relax relax.in relax.out

echo "[INFO] End       : $(date)"
"""
    _write(path, text)


def main() -> None:
    args = parse_args()
    rootdir = Path(args.rootdir).expanduser().resolve()
    source = rootdir / "kmesh_scan" / f"k_{args.source_tag}"
    target = rootdir / "kmesh_scan" / f"k_{args.target_tag}"

    if not source.exists():
        raise FileNotFoundError(f"Source case not found: {source}")
    if target.exists():
        if not args.force:
            raise FileExistsError(f"Target already exists: {target}. Re-run with --force to replace it.")
        shutil.rmtree(target)

    shutil.copytree(source, target)
    _cleanup_case(target)

    _retarget_qe_input(
        target / "pristine_scf" / "scf.in",
        tag=str(args.target_tag),
        prefix=f"Al_pristine_prim4_k_{args.target_tag}",
    )
    _retarget_qe_input(
        target / "vacancy_relax" / "relax.in",
        tag=str(args.target_tag),
        prefix=f"Al_vac_prim4_k_{args.target_tag}",
    )
    _write_group_job(
        target / "group_job_ct56_long.sh",
        job_name=str(args.job_name),
        partition=str(args.partition),
        ntasks=int(args.ntasks),
        time_limit=str(args.time_limit),
        mem=str(args.mem),
        account=str(args.account),
    )

    print("============================================================")
    print("QE vacancy extra k-mesh case prepared")
    print("============================================================")
    print(f"Source : {source}")
    print(f"Target : {target}")
    print(f"Submit : cd {target} && sbatch group_job_ct56_long.sh")


if __name__ == "__main__":
    main()
