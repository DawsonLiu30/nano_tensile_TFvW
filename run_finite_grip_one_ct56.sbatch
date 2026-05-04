#!/bin/bash
#SBATCH -J fgTFvW1
#SBATCH -A MST114175
#SBATCH -p ct56
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=120G
#SBATCH -t 24:00:00
#SBATCH -o logs_ctest/%x_%j.out
#SBATCH -e logs_ctest/%x_%j.err

set -euo pipefail

ROOT="${ROOT:-/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"
DIAMETER="${DIAMETER:?DIAMETER is not set. Example: DIAMETER=3 sbatch run_finite_grip_one_ct56.sbatch}"
CYCLES="${CYCLES:-80}"
STEP="${STEP:-0.01}"
ECUT="${ECUT:-1000}"
SPACING="${SPACING:-}"
FMAX="${FMAX:-0.02}"
PREP_RELAX_STEPS="${PREP_RELAX_STEPS:-120}"
TENSILE_RELAX_STEPS="${TENSILE_RELAX_STEPS:-80}"
CYCLES_PER_LAUNCH="${CYCLES_PER_LAUNCH:-1}"
A0="${A0:-4.118877004246}"

mkdir -p "${ROOT}/logs_ctest" "${ROOT}/results/professor_review"

echo "============================================================"
echo "[INFO] Job start     : $(date)"
echo "[INFO] Host          : $(hostname)"
echo "[INFO] ROOT          : ${ROOT}"
echo "[INFO] DIAMETER      : ${DIAMETER}"
echo "[INFO] STEP/CYCLES   : ${STEP} / ${CYCLES}"
echo "[INFO] SPACING       : ${SPACING:-auto-from-ecut}"
echo "[INFO] CHUNK CYCLES  : ${CYCLES_PER_LAUNCH}"
echo "============================================================"

if [[ ! -d "${ROOT}" ]]; then
    echo "[ERROR] Missing ROOT: ${ROOT}"
    exit 1
fi

cd "${ROOT}"

if [[ ! -f "al.gga.recpot" ]]; then
    echo "[ERROR] Missing pseudopotential: ${ROOT}/al.gga.recpot"
    exit 1
fi

source /home/dawson666/miniconda3/etc/profile.d/conda.sh
conda activate dftpy-env

export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "[ENV] python   = $(which python)"
echo "[ENV] version  = $(python -V 2>&1)"
echo "[ENV] dftpy    = $(python -c 'import dftpy; print(dftpy.__file__)')"
echo "[ENV] pylibxc  = $(python -c 'import pylibxc; print(pylibxc.__file__)')"
echo "[ENV] OMP_NUM_THREADS = ${OMP_NUM_THREADS}"

CMD=(
    python scripts/run_grip_vacancy_hpc_series.py
    --diameters "${DIAMETER}"
    --a0 "${A0}"
    --orientation 111
    --wire-length 21.0
    --min-wire-span 10.0
    --xy-vacuum 10.0
    --z-vacuum 10.0
    --grip-thickness 3.0
    --vacancy-z-window-fraction 0.35
    --pp al.gga.recpot
    --kedf TFVW
    --fmax "${FMAX}"
    --prep-relax-steps "${PREP_RELAX_STEPS}"
    --step "${STEP}"
    --cycles "${CYCLES}"
    --cycles-per-launch "${CYCLES_PER_LAUNCH}"
    --tensile-relax-steps "${TENSILE_RELAX_STEPS}"
    --fracture-gap-factor 3.0
    --resume-latest
)

if [[ -n "${SPACING}" ]]; then
    CMD+=(--spacing "${SPACING}")
else
    CMD+=(--ecut "${ECUT}")
fi

"${CMD[@]}"

echo "============================================================"
echo "[INFO] Job end       : $(date)"
echo "[INFO] DIAMETER      : ${DIAMETER}"
echo "============================================================"
