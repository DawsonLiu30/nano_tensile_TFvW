#!/bin/bash
#SBATCH --job-name=dftpyLMfine
#SBATCH --output=logs_ctest/%x_%A_%a.out
#SBATCH --error=logs_ctest/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --partition=ctest
#SBATCH --no-requeue
#SBATCH --account=MST114175
#SBATCH --array=0-2%2

set -euo pipefail

ROOT="${ROOT:-/work/dawson666/dftpy_project/relax/dftpy45}"
SERIES_NAME="${SERIES_NAME:-dftpy_vacancy_tfvw_lambda_mu_fine_20260622}"
SETTING_FILE="${ROOT}/results/${SERIES_NAME}/settings_weight_scan.txt"

source /home/dawson666/miniconda3/etc/profile.d/conda.sh
conda activate dftpy-env

export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p "${ROOT}/logs_ctest"
cd "${ROOT}"
mapfile -t SETTINGS < "$SETTING_FILE"
WORKER_COUNT="${SLURM_ARRAY_TASK_COUNT:-3}"

for ((INDEX=SLURM_ARRAY_TASK_ID; INDEX<${#SETTINGS[@]}; INDEX+=WORKER_COUNT)); do
  SETTING="${SETTINGS[$INDEX]}"
  RESULT="${ROOT}/results/${SERIES_NAME}/weight_scan/${SETTING}/result.json"
  if [ -s "$RESULT" ]; then
    echo "[SKIP] $SETTING already completed"
    continue
  fi

  echo "[RUN] index=$INDEX setting=$SETTING"
  if ! python scripts/run_dftpy_vcrelax_vacancy_one.py \
    --rootdir "${ROOT}/results/${SERIES_NAME}" \
    --setting "$SETTING" \
    --scan weight \
    --ase-optimizer BFGS; then
    echo "[FAILED] $SETTING" >&2
    touch "${ROOT}/results/${SERIES_NAME}/weight_scan/${SETTING}/WORKER_FAILED.txt"
  fi
done

