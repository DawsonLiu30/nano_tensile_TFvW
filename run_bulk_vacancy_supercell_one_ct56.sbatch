#!/bin/bash
#SBATCH -J bulkVac
#SBATCH -A MST114175
#SBATCH -p ct56
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=120G
#SBATCH -t 4-00:00:00
#SBATCH -o logs_ctest/%x_%j.out
#SBATCH -e logs_ctest/%x_%j.err

set -euo pipefail

ROOT="${ROOT:-/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"
SERIES_NAME="${SERIES_NAME:?SERIES_NAME is not set. Example: SERIES_NAME=bulk_fcc_vacancy_supercell_tfvw_20260430_8to14}"
SUPERCELL="${SUPERCELL:?SUPERCELL is not set. Example: SUPERCELL=8}"
PP="${PP:-al.gga.recpot}"
KEDF="${KEDF:-TFVW}"
ECUT="${ECUT:-1000}"
SPACING="${SPACING:-}"
FMAX="${FMAX:-0.02}"
RELAX_STEPS="${RELAX_STEPS:-200}"
DELTA_THRESHOLD="${DELTA_THRESHOLD:-0.02}"
A0="${A0:-}"
CELL_BASIS="${CELL_BASIS:-conventional}"

OUTDIR="${ROOT}/results/${SERIES_NAME}"

mkdir -p "${ROOT}/logs_ctest"

echo "============================================================"
echo "[INFO] Job start      : $(date)"
echo "[INFO] Host           : $(hostname)"
echo "[INFO] ROOT           : ${ROOT}"
echo "[INFO] SERIES_NAME    : ${SERIES_NAME}"
echo "[INFO] SUPERCELL      : ${SUPERCELL}x${SUPERCELL}x${SUPERCELL}"
echo "[INFO] OUTDIR         : ${OUTDIR}"
echo "[INFO] PP/KEDF        : ${PP} / ${KEDF}"
echo "[INFO] ECUT/SPACING   : ${ECUT} / ${SPACING:-auto-from-ecut}"
echo "[INFO] FMAX/STEPS     : ${FMAX} / ${RELAX_STEPS}"
echo "[INFO] DELTA TARGET   : ${DELTA_THRESHOLD} eV/vacancy"
echo "[INFO] CELL_BASIS     : ${CELL_BASIS}"
echo "============================================================"

if [[ ! -d "${ROOT}" ]]; then
    echo "[ERROR] Missing ROOT: ${ROOT}"
    exit 1
fi

cd "${ROOT}"

if [[ ! -f "${PP}" ]]; then
    echo "[ERROR] Missing pseudopotential: ${ROOT}/${PP}"
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
    python scripts/run_bulk_vacancy_supercell_series.py
    --series "${SERIES_NAME}"
    --outdir "${OUTDIR}"
    --supercells "${SUPERCELL}"
    --pp "${PP}"
    --kedf "${KEDF}"
    --fmax "${FMAX}"
    --relax-steps "${RELAX_STEPS}"
    --delta-threshold "${DELTA_THRESHOLD}"
    --cell-basis "${CELL_BASIS}"
)

if [[ -n "${A0}" ]]; then
    CMD+=(--a0 "${A0}")
fi

if [[ -n "${SPACING}" ]]; then
    CMD+=(--spacing "${SPACING}")
else
    CMD+=(--ecut "${ECUT}")
fi

"${CMD[@]}"

echo "============================================================"
echo "[INFO] Job end        : $(date)"
echo "[INFO] SUPERCELL      : ${SUPERCELL}x${SUPERCELL}x${SUPERCELL}"
echo "============================================================"
