#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-iservice}"
REMOTE_ROOT="${REMOTE_ROOT:-/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"
SERIES_NAME="${SERIES_NAME:-dftpy_tfvw_lambda_mu_bulk_10x10_20260609}"
ACCOUNT="${ACCOUNT:-MST114175}"
PARTITION="${PARTITION:-ctest}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
GROUP_SIZE="${GROUP_SIZE:-10}"
ARRAY_START="${1:-0}"
ARRAY_END="${2:-1}"

cat <<EOF
============================================================
Submit existing DFTpy TF+vW lambda-mu bulk scan
============================================================
[REMOTE] ${REMOTE_HOST}:${REMOTE_ROOT}
[SERIES] ${SERIES_NAME}
[ARRAY ] ${ARRAY_START}-${ARRAY_END}%${MAX_PARALLEL}
[GROUP ] ${GROUP_SIZE} coefficient pairs per task
[PART  ] ${PARTITION}
EOF

ssh "${REMOTE_HOST}" "
set -euo pipefail
cd '${REMOTE_ROOT}'
SETTING_FILE='results/${SERIES_NAME}/settings_lambda_mu_scan.txt'
N=\$(wc -l < \"\${SETTING_FILE}\")
NGROUPS=\$(((N + ${GROUP_SIZE} - 1) / ${GROUP_SIZE}))
LAST=\$((NGROUPS - 1))
if (( ${ARRAY_START} < 0 || ${ARRAY_END} < ${ARRAY_START} || ${ARRAY_END} > LAST )); then
  echo '[ERROR] Invalid group range ${ARRAY_START}-${ARRAY_END}; valid range is 0-'\"\${LAST}\" >&2
  exit 2
fi
sbatch \
  -A '${ACCOUNT}' \
  -p '${PARTITION}' \
  -t '${TIME_LIMIT}' \
  --export=ALL,SERIES_NAME='${SERIES_NAME}',GROUP_SIZE='${GROUP_SIZE}' \
  --array='${ARRAY_START}-${ARRAY_END}%${MAX_PARALLEL}' \
  submit_dftpy_tfvw_lambda_mu_bulk_ctest_array.sh
"
