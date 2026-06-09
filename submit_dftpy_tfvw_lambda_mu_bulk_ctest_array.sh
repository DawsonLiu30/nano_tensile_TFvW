#!/bin/bash
#SBATCH -J dftpyLM
#SBATCH -A MST114175
#SBATCH -p ctest
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --array=0-1%2
#SBATCH -o logs_ctest/%x_%A_%a.out
#SBATCH -e logs_ctest/%x_%A_%a.err

set -euo pipefail

ROOT="${ROOT:-/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"
SERIES_NAME="${SERIES_NAME:-dftpy_tfvw_lambda_mu_bulk_10x10_20260609}"
SETTING_FILE="${ROOT}/results/${SERIES_NAME}/settings_lambda_mu_scan.txt"
GROUP_SIZE="${GROUP_SIZE:-10}"

mkdir -p "${ROOT}/logs_ctest"
cd "${ROOT}"

source /home/dawson666/miniconda3/etc/profile.d/conda.sh
conda activate dftpy-env

export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "[INFO] SERIES_NAME=${SERIES_NAME}"
echo "[INFO] GROUP_SIZE=${GROUP_SIZE}"

N_SETTINGS=$(wc -l < "${SETTING_FILE}")
START_INDEX=$((SLURM_ARRAY_TASK_ID * GROUP_SIZE))
END_INDEX=$((START_INDEX + GROUP_SIZE - 1))
if (( START_INDEX >= N_SETTINGS )); then
  echo "[ERROR] Group starts at ${START_INDEX}, but only ${N_SETTINGS} settings exist." >&2
  exit 1
fi
if (( END_INDEX >= N_SETTINGS )); then
  END_INDEX=$((N_SETTINGS - 1))
fi

FAILED=0
for ((INDEX=START_INDEX; INDEX<=END_INDEX; INDEX++)); do
  SETTING=$(sed -n "$((INDEX + 1))p" "${SETTING_FILE}")
  echo "============================================================"
  echo "[INFO] INDEX=${INDEX}"
  echo "[INFO] SETTING=${SETTING}"
  if ! python scripts/run_dftpy_tfvw_lambda_mu_bulk_one.py \
    --rootdir "${ROOT}/results/${SERIES_NAME}" \
    --setting "${SETTING}"; then
    echo "[ERROR] Failed setting: ${SETTING}" >&2
    FAILED=$((FAILED + 1))
  fi
done

echo "[INFO] Failed settings in this group: ${FAILED}"
exit "${FAILED}"
