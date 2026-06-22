#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
REMOTE_HOST="${REMOTE_HOST:-iservice}"
REMOTE_ROOT="${REMOTE_ROOT:-/work/dawson666/dftpy_project/relax/dftpy45}"
SERIES_NAME="${SERIES_NAME:-dftpy_vacancy_tfvw_lambda_mu_fine_20260622}"

LAMBDA_LIST="${LAMBDA_LIST:-0.90,0.91,0.92,0.93,0.94,0.95}"
MU_LIST="${MU_LIST:-0.04,0.05,0.06,0.07,0.08,0.09,0.10}"
PP_LOCAL="${PP_LOCAL:-${LOCAL_ROOT}/al.lda.recpot}"

cd "${LOCAL_ROOT}"

echo "[1/3] Upload scripts and pseudopotential"
ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_ROOT}/scripts' '${REMOTE_ROOT}/app'"
rsync -avhP \
  app/dft_engine.py \
  "${REMOTE_HOST}:${REMOTE_ROOT}/app/dft_engine.py"
rsync -avhP \
  scripts/prepare_dftpy_vacancy_conventional.py \
  scripts/prepare_dftpy_tfvw_lambda_mu_vacancy_fine_scan.py \
  scripts/run_dftpy_vcrelax_vacancy_one.py \
  scripts/collect_dftpy_conventional_vacancy.py \
  "${REMOTE_HOST}:${REMOTE_ROOT}/scripts/"
rsync -avhP "${PP_LOCAL}" "${REMOTE_HOST}:${REMOTE_ROOT}/al.lda.recpot"

echo "[2/3] Prepare 42-point fine scan"
ssh "${REMOTE_HOST}" "
set -euo pipefail
cd '${REMOTE_ROOT}'
python scripts/prepare_dftpy_tfvw_lambda_mu_vacancy_fine_scan.py \\
  --outdir 'results/${SERIES_NAME}' \\
  --pp 'al.lda.recpot' \\
  --lambda-list '${LAMBDA_LIST}' \\
  --mu-list '${MU_LIST}'
"

echo "[3/3] Submit ctest array"
ssh "${REMOTE_HOST}" "
set -euo pipefail
cd '${REMOTE_ROOT}'
SERIES_NAME='${SERIES_NAME}' sbatch 'results/${SERIES_NAME}/submit_dftpy_lambda_mu_fine_array.sh'
"

echo "Submitted ${SERIES_NAME}"

