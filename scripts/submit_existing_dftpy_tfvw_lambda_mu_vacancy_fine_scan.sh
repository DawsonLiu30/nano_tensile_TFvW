#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
REMOTE_HOST="${REMOTE_HOST:-iservice}"
REMOTE_ROOT="${REMOTE_ROOT:-/work/dawson666/dftpy_project/relax/dftpy45}"
SERIES_NAME="${SERIES_NAME:-dftpy_vacancy_tfvw_lambda_mu_fine_20260622}"
WORKERS="${WORKERS:-3}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

cd "${LOCAL_ROOT}"

echo "Submitting existing fine scan as ${WORKERS} workers, max ${MAX_PARALLEL} concurrent"
rsync -avhP \
  scripts/submit_dftpy_lambda_mu_fine_workers.sh \
  "${REMOTE_HOST}:${REMOTE_ROOT}/results/${SERIES_NAME}/submit_dftpy_lambda_mu_fine_workers.sh"

ssh "${REMOTE_HOST}" "
set -euo pipefail
cd '${REMOTE_ROOT}'
SERIES='results/${SERIES_NAME}'
test -s \"\${SERIES}/settings_weight_scan.txt\"
SERIES_NAME='${SERIES_NAME}' sbatch \\
  --array=0-$((WORKERS - 1))%${MAX_PARALLEL} \\
  \"\${SERIES}/submit_dftpy_lambda_mu_fine_workers.sh\"
"
