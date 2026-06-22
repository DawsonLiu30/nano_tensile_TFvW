#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
REMOTE_HOST="${REMOTE_HOST:-iservice}"
REMOTE_ROOT="${REMOTE_ROOT:-/work/dawson666/dftpy_project/relax/dftpy45}"
SERIES_DIR="${SERIES_DIR:-${REMOTE_ROOT}/results/Al_defects/01_calibration/single_vacancy/dftpy_tfvw_lambda_mu/fine_L0p90-0p95_M0p04-0p10}"
ACTIVE_DIR="${SERIES_DIR}/02_submission_scripts/active"
ARRAY_START="${ARRAY_START:-0}"
ARRAY_END="${ARRAY_END:-20}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"

if (( ARRAY_START < 0 || ARRAY_END > 41 || ARRAY_START > ARRAY_END )); then
  echo "[ERROR] Invalid array range ${ARRAY_START}-${ARRAY_END}; expected indices 0-41" >&2
  exit 2
fi
if (( ARRAY_END - ARRAY_START + 1 > 21 )); then
  echo "[ERROR] Submit at most 21 tasks per chunk to stay below the user job-submit limit" >&2
  exit 3
fi

cd "${LOCAL_ROOT}"
echo "============================================================"
echo "DFTpy lambda/mu one-case ct56 submission"
echo "============================================================"
echo "[SERIES] ${REMOTE_HOST}:${SERIES_DIR}"
echo "[ARRAY ] ${ARRAY_START}-${ARRAY_END}%${MAX_PARALLEL}"
echo "[TIME  ] ${TIME_LIMIT} per point"

ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_ROOT}/scripts' '${ACTIVE_DIR}' '${REMOTE_ROOT}/logs_ct56'"
rsync -avhP \
  scripts/run_dftpy_vcrelax_vacancy_one.py \
  scripts/materialize_dftpy_case_reproducibility.py \
  scripts/audit_vacancy_submission_gate.py \
  "${REMOTE_HOST}:${REMOTE_ROOT}/scripts/"
rsync -avhP \
  scripts/submit_dftpy_lambda_mu_fine_one_case_ct56.sh \
  "${REMOTE_HOST}:${ACTIVE_DIR}/submit_dftpy_lambda_mu_fine_one_case_ct56.sh"

ssh "${REMOTE_HOST}" "
set -euo pipefail
cd '${REMOTE_ROOT}'
source /home/dawson666/miniconda3/etc/profile.d/conda.sh
conda activate dftpy-env

python scripts/materialize_dftpy_case_reproducibility.py \\
  --rootdir '${SERIES_DIR}' \\
  --pp '${REMOTE_ROOT}/al.lda.recpot'

python scripts/audit_vacancy_submission_gate.py \\
  --dftpy-root '${SERIES_DIR}' \\
  --json-out '${SERIES_DIR}/03_audit/pre_submit_gate_latest.json'

active_jobs=\$(squeue -h -u dawson666 -n dftpyLM1 | wc -l)
if [[ \"\${active_jobs}\" -gt 0 ]]; then
  echo \"[ERROR] Existing dftpyLM1 jobs are still active; refusing duplicate submission\" >&2
  squeue -u dawson666 -n dftpyLM1
  exit 5
fi

SERIES_DIR='${SERIES_DIR}' sbatch \\
  --array='${ARRAY_START}-${ARRAY_END}%${MAX_PARALLEL}' \\
  --time='${TIME_LIMIT}' \\
  '${ACTIVE_DIR}/submit_dftpy_lambda_mu_fine_one_case_ct56.sh'
"

echo "Submitted chunk ${ARRAY_START}-${ARRAY_END}."
echo "After it completes, submit the second chunk with ARRAY_START=21 ARRAY_END=41."
