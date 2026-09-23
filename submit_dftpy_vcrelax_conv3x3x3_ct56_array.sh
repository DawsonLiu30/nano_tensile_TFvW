#!/bin/bash
#SBATCH -J dftpyVCR3
#SBATCH -A MST114175
#SBATCH -p ct56
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 4-00:00:00
#SBATCH --array=0-5%3
#SBATCH -o logs_ctest/%x_%A_%a.out
#SBATCH -e logs_ctest/%x_%A_%a.err

set -euo pipefail

ROOT="${ROOT:-/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"
SERIES_NAME="${SERIES_NAME:-dftpy_vacancy_vcrelax_conv3x3x3_qe_a0_20260528}"
SETTING_FILE="${ROOT}/results/${SERIES_NAME}/settings_spacing_scan.txt"

mkdir -p "${ROOT}/logs_ctest"
cd "${ROOT}"

source /home/dawson666/miniconda3/etc/profile.d/conda.sh
conda activate dftpy-env

export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

SETTING=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SETTING_FILE")
if [ -z "$SETTING" ]; then
  echo "[ERROR] Empty setting from $SETTING_FILE"
  exit 1
fi

echo "[INFO] SERIES_NAME=$SERIES_NAME"
echo "[INFO] SETTING=$SETTING"

python scripts/run_dftpy_vcrelax_vacancy_one.py \
  --rootdir "${ROOT}/results/${SERIES_NAME}" \
  --setting "$SETTING" \
  --scan spacing
