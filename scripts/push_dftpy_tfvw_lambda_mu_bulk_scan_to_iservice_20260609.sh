#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
REMOTE_HOST="${REMOTE_HOST:-iservice}"
REMOTE_ROOT="${REMOTE_ROOT:-/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"
SERIES_NAME="${SERIES_NAME:-dftpy_tfvw_lambda_mu_bulk_10x10_20260609}"
ACCOUNT="${ACCOUNT:-MST114175}"
PARTITION="${PARTITION:-ctest}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
GROUP_SIZE="${GROUP_SIZE:-10}"
ARRAY_START="${1:-${ARRAY_START:-0}}"
ARRAY_END="${2:-${ARRAY_END:-1}}"
PREPARE="${PREPARE:-1}"
SUBMIT="${SUBMIT:-1}"

LAMBDA_LIST="${LAMBDA_LIST:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
MU_LIST="${MU_LIST:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
A0_LIST="${A0_LIST:-2.20,2.30,2.40,2.50,2.60,2.70,2.80,2.90,3.00,3.10,3.20,3.30,3.40,3.50,3.60,3.70,3.80,3.90,4.00,4.10,4.20,4.30,4.40,4.50,4.60,4.70,4.80,4.90,5.00}"
SPACING="${SPACING:-0.20}"
OPT_METHOD="${OPT_METHOD:-CG-HS}"
OPT_MAXITER="${OPT_MAXITER:-500}"
OPT_MAXFUN="${OPT_MAXFUN:-500}"

cat <<EOF
============================================================
Push DFTpy TF+vW lambda-mu bulk scan
============================================================
[LOCAL ] ${LOCAL_ROOT}
[REMOTE] ${REMOTE_HOST}:${REMOTE_ROOT}
[SERIES] ${SERIES_NAME}
[LAMBDA] ${LAMBDA_LIST}
[MU    ] ${MU_LIST}
[A0    ] ${A0_LIST}
[GRID  ] ${SPACING} A
[OPT   ] ${OPT_METHOD}, maxiter=${OPT_MAXITER}, maxfun=${OPT_MAXFUN}
[ARRAY ] ${ARRAY_START}-${ARRAY_END}%${MAX_PARALLEL}
[GROUP ] ${GROUP_SIZE} coefficient pairs per array task
[PART  ] ${PARTITION}
EOF

cd "${LOCAL_ROOT}"

rsync -avhP \
  app/dft_engine.py \
  "${REMOTE_HOST}:${REMOTE_ROOT}/app/dft_engine.py"
rsync -avhP \
  scripts/prepare_dftpy_tfvw_lambda_mu_bulk_scan.py \
  scripts/run_dftpy_tfvw_lambda_mu_bulk_one.py \
  scripts/evaluate_dftpy_tfvw_bulk_point.py \
  scripts/collect_dftpy_tfvw_lambda_mu_bulk_scan.py \
  "${REMOTE_HOST}:${REMOTE_ROOT}/scripts/"
rsync -avhP \
  submit_dftpy_tfvw_lambda_mu_bulk_ctest_array.sh \
  "${REMOTE_HOST}:${REMOTE_ROOT}/"

if [[ "${PREPARE}" == "1" ]]; then
  ssh "${REMOTE_HOST}" "
set -euo pipefail
cd '${REMOTE_ROOT}'
python scripts/prepare_dftpy_tfvw_lambda_mu_bulk_scan.py \
  --outdir 'results/${SERIES_NAME}' \
  --lambda-list '${LAMBDA_LIST}' \
  --mu-list '${MU_LIST}' \
  --a0-list '${A0_LIST}' \
  --spacing '${SPACING}' \
  --repeat 1x1x1 \
  --pp al.lda.recpot \
  --xc LDA \
  --opt-method '${OPT_METHOD}' \
  --opt-maxiter '${OPT_MAXITER}' \
  --opt-maxfun '${OPT_MAXFUN}'
"
fi

if [[ "${SUBMIT}" == "1" ]]; then
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
fi

cat <<EOF

Prepared/submitted task groups ${ARRAY_START}-${ARRAY_END}.

Submit later chunks without rebuilding:
  bash scripts/submit_existing_dftpy_tfvw_lambda_mu_bulk_scan_to_iservice_20260609.sh 2 3
  bash scripts/submit_existing_dftpy_tfvw_lambda_mu_bulk_scan_to_iservice_20260609.sh 4 5
  bash scripts/submit_existing_dftpy_tfvw_lambda_mu_bulk_scan_to_iservice_20260609.sh 6 7
  bash scripts/submit_existing_dftpy_tfvw_lambda_mu_bulk_scan_to_iservice_20260609.sh 8 9
EOF
