#!/bin/bash
#SBATCH --job-name=dftpyLM1
#SBATCH --output=logs_ct56/%x_%A_%a.out
#SBATCH --error=logs_ct56/%x_%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --partition=ct56
#SBATCH --no-requeue
#SBATCH --account=MST114175

set -euo pipefail

ROOT="${ROOT:-/work/dawson666/dftpy_project/relax/dftpy45}"
SERIES_DIR="${SERIES_DIR:-${ROOT}/results/Al_defects/01_calibration/single_vacancy/dftpy_tfvw_lambda_mu/fine_L0p90-0p95_M0p04-0p10}"
SETTING_FILE="${SETTING_FILE:-${SERIES_DIR}/01_settings/settings_weight_scan.txt}"

source /home/dawson666/miniconda3/etc/profile.d/conda.sh
conda activate dftpy-env

export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${ROOT}"
SETTING=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${SETTING_FILE}")
if [[ -z "${SETTING}" ]]; then
  echo "[ERROR] No setting for array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

CASE_DIR="${SERIES_DIR}/weight_scan/${SETTING}"
RESULT="${CASE_DIR}/result.json"
RUN_MARKER="${CASE_DIR}/RUNNING_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
FAIL_MARKER="${CASE_DIR}/FAILED_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
DONE_MARKER="${CASE_DIR}/COMPLETED_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"

for required in \
  point_manifest.json \
  pristine_raw.vasp \
  vacancy_start.vasp \
  dftpy_pristine_input.ini \
  dftpy_vacancy_input.ini \
  al.lda.recpot; do
  if [[ ! -s "${CASE_DIR}/${required}" ]]; then
    echo "[ERROR] Missing case input: ${CASE_DIR}/${required}" >&2
    exit 3
  fi
done

if [[ -s "${RESULT}" ]]; then
  echo "[SKIP] ${SETTING} already has result.json"
  exit 0
fi

cleanup() {
  rc=$?
  rm -f "${RUN_MARKER}"
  if [[ ${rc} -ne 0 ]]; then
    printf 'job_id=%s\narray_task_id=%s\nsetting=%s\nexit_code=%s\nend=%s\n' \
      "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${SETTING}" "${rc}" "$(date -Is)" \
      > "${FAIL_MARKER}"
  fi
}
trap cleanup EXIT

printf 'job_id=%s\narray_task_id=%s\nsetting=%s\nhost=%s\nstart=%s\n' \
  "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${SETTING}" "$(hostname)" "$(date -Is)" \
  > "${RUN_MARKER}"
rm -f "${FAIL_MARKER}" "${DONE_MARKER}" "${CASE_DIR}/WORKER_FAILED.txt"

echo "[INFO] SERIES_DIR=${SERIES_DIR}"
echo "[INFO] SETTING=${SETTING}"
echo "[INFO] CASE_DIR=${CASE_DIR}"
echo "[INFO] START=$(date -Is)"

python scripts/run_dftpy_vcrelax_vacancy_one.py \
  --rootdir "${SERIES_DIR}" \
  --setting "${SETTING}" \
  --scan weight \
  --ase-optimizer BFGS

if [[ ! -s "${RESULT}" ]]; then
  echo "[ERROR] Runner returned without a non-empty result.json" >&2
  exit 4
fi

printf 'job_id=%s\narray_task_id=%s\nsetting=%s\nend=%s\n' \
  "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${SETTING}" "$(date -Is)" \
  > "${DONE_MARKER}"
echo "[PASS] ${SETTING} completed"
