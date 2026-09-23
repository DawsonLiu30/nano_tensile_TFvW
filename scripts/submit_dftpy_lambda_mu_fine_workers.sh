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
SERIES_DIR="${SERIES_DIR:-${ROOT}/results/Al_defects/01_calibration/single_vacancy/dftpy_tfvw_lambda_mu/fine_L0p90-0p95_M0p04-0p10}"
SETTING_FILE="${SERIES_DIR}/settings_weight_scan.txt"

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
  RESULT="${SERIES_DIR}/weight_scan/${SETTING}/result.json"
  if [ -s "$RESULT" ]; then
    echo "[SKIP] $SETTING already completed"
    continue
  fi

  echo "[RUN] index=$INDEX setting=$SETTING"
  if ! python scripts/run_dftpy_vcrelax_vacancy_one.py \
    --rootdir "${SERIES_DIR}" \
    --setting "$SETTING" \
    --scan weight \
    --ase-optimizer BFGS; then
    echo "[FAILED] $SETTING" >&2
    touch "${SERIES_DIR}/weight_scan/${SETTING}/WORKER_FAILED.txt"
  fi
done
