#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
REMOTE_HOST="${REMOTE_HOST:-iservice}"
REMOTE_ROOT="${REMOTE_ROOT:-/work/dawson666/dftpy_project/relax/dftpy45}"
SERIES_RELATIVE_DIR="${SERIES_RELATIVE_DIR:-${SERIES_NAME:-Al_defects/01_calibration/single_vacancy/dftpy_tfvw_lambda_mu/fine_L0p90-0p95_M0p04-0p10}}"
REMOTE_SERIES_DIR="${REMOTE_ROOT}/results/${SERIES_RELATIVE_DIR}"
WORKERS="${WORKERS:-3}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

cd "${LOCAL_ROOT}"

echo "Submitting existing fine scan as ${WORKERS} workers, max ${MAX_PARALLEL} concurrent"
ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_SERIES_DIR}'"
rsync -avhP \
  scripts/submit_dftpy_lambda_mu_fine_workers.sh \
  "${REMOTE_HOST}:${REMOTE_SERIES_DIR}/submit_dftpy_lambda_mu_fine_workers.sh"

ssh "${REMOTE_HOST}" "
set -euo pipefail
cd '${REMOTE_ROOT}'
SERIES='${REMOTE_SERIES_DIR}'
test -s \"\${SERIES}/settings_weight_scan.txt\"
SERIES_DIR='${REMOTE_SERIES_DIR}' sbatch \\
  --array=0-$((WORKERS - 1))%${MAX_PARALLEL} \\
  \"\${SERIES}/submit_dftpy_lambda_mu_fine_workers.sh\"
"
