#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-dawson666@twnia3.nchc.org.tw}"

# Current cleaned iService layout checked on 2026-06-29.
REMOTE_TFVW_ROOT="${REMOTE_TFVW_ROOT:-/work/dawson666/dftpy_project/relax/TFvW_test}"

# Code provenance for the production runner used by TFvW_test.
REMOTE_CODE_ROOT="${REMOTE_CODE_ROOT:-/work/dawson666/dftpy_project/relax/OLD_dftpy45_column_scan}"

LOCAL_DELIVERY="${LOCAL_DELIVERY:-/mnt/c/Users/dawso/Desktop/DFTPY_VACLM_PROF_DELIVERY_20260629_ISERVICE_ONLY}"

echo "============================================================"
echo "Pull iService VACLM full provenance"
echo "============================================================"
echo "[REMOTE_HOST] ${REMOTE_HOST}"
echo "[REMOTE_TFVW] ${REMOTE_TFVW_ROOT}"
echo "[REMOTE_CODE] ${REMOTE_CODE_ROOT}"
echo "[LOCAL     ] ${LOCAL_DELIVERY}"
echo
echo "This will merge missing iService provenance into the existing"
echo "DFTPY_VACLM_PROF_DELIVERY_20260629_ISERVICE_ONLY package."
echo "It is expected to ask for NCHC 2FA/password."
echo "============================================================"

mkdir -p "${LOCAL_DELIVERY}/03_RAW_CASES"
mkdir -p "${LOCAL_DELIVERY}/04_RUN_METADATA"
mkdir -p "${LOCAL_DELIVERY}/05_SCRIPTS_USED/scripts"
mkdir -p "${LOCAL_DELIVERY}/05_SCRIPTS_USED/app"

echo
echo "[0/6] Remote sanity check"
ssh "${REMOTE_HOST}" "set -e
  echo HOST=\$(hostname)
  test -d '${REMOTE_TFVW_ROOT}'
  test -d '${REMOTE_TFVW_ROOT}/03_runs'
  echo RUN_DIRS=\$(find '${REMOTE_TFVW_ROOT}/03_runs' -mindepth 1 -maxdepth 1 -type d | wc -l)
  echo RESULT_JSON=\$(find '${REMOTE_TFVW_ROOT}/03_runs' -name result.json | wc -l)
  echo TRAJ=\$(find '${REMOTE_TFVW_ROOT}/03_runs' -name '*.traj' | wc -l)
  echo RELAX_LOGS=\$(find '${REMOTE_TFVW_ROOT}/03_runs' \\( -name 'pristine_relax.log' -o -name 'vacancy_relax.log' \\) | wc -l)
  echo INPUTS=\$(find '${REMOTE_TFVW_ROOT}/03_runs' \\( -name 'dftpy_pristine_input.ini' -o -name 'dftpy_vacancy_input.ini' \\) | wc -l)
"

echo
echo "[1/6] Pull complete raw case folders, including .traj"
rsync -avhP \
  "${REMOTE_HOST}:${REMOTE_TFVW_ROOT}/03_runs/" \
  "${LOCAL_DELIVERY}/03_RAW_CASES/03_runs/"

echo
echo "[2/6] Pull professor raw tables"
if ssh "${REMOTE_HOST}" "test -d '${REMOTE_TFVW_ROOT}/10_professor_raw_log_table'"; then
  mkdir -p "${LOCAL_DELIVERY}/01_RAW_TABLE"
  rsync -avhP \
    "${REMOTE_HOST}:${REMOTE_TFVW_ROOT}/10_professor_raw_log_table/" \
    "${LOCAL_DELIVERY}/01_RAW_TABLE/"
else
  echo "[WARN] remote 10_professor_raw_log_table not found"
fi

echo
echo "[3/6] Pull analysis/professor table metadata if present"
for remote_subdir in \
  00_README \
  00_matrix_index \
  01_settings \
  05_tables \
  07_audit \
  08_analysis_gillan_style \
  09_professor_table
do
  if ssh "${REMOTE_HOST}" "test -d '${REMOTE_TFVW_ROOT}/${remote_subdir}'"; then
    mkdir -p "${LOCAL_DELIVERY}/04_RUN_METADATA/${remote_subdir}"
    rsync -avhP \
      "${REMOTE_HOST}:${REMOTE_TFVW_ROOT}/${remote_subdir}/" \
      "${LOCAL_DELIVERY}/04_RUN_METADATA/${remote_subdir}/"
  else
    echo "[INFO] skip missing ${remote_subdir}"
  fi
done

echo
echo "[4/6] Pull submission scripts"
if ssh "${REMOTE_HOST}" "test -d '${REMOTE_TFVW_ROOT}/06_submission_scripts'"; then
  mkdir -p "${LOCAL_DELIVERY}/05_SCRIPTS_USED/06_submission_scripts"
  rsync -avhP \
    "${REMOTE_HOST}:${REMOTE_TFVW_ROOT}/06_submission_scripts/" \
    "${LOCAL_DELIVERY}/05_SCRIPTS_USED/06_submission_scripts/"
else
  echo "[WARN] remote 06_submission_scripts not found"
fi

echo
echo "[5/6] Pull production Python runner and DFT engine"
if ssh "${REMOTE_HOST}" "test -f '${REMOTE_CODE_ROOT}/scripts/run_dftpy_vcrelax_vacancy_matrix_one.py'"; then
  rsync -avhP \
    "${REMOTE_HOST}:${REMOTE_CODE_ROOT}/scripts/run_dftpy_vcrelax_vacancy_matrix_one.py" \
    "${LOCAL_DELIVERY}/05_SCRIPTS_USED/scripts/"
else
  echo "[WARN] production runner not found under ${REMOTE_CODE_ROOT}/scripts"
fi

if ssh "${REMOTE_HOST}" "test -f '${REMOTE_CODE_ROOT}/app/dft_engine.py'"; then
  rsync -avhP \
    "${REMOTE_HOST}:${REMOTE_CODE_ROOT}/app/dft_engine.py" \
    "${LOCAL_DELIVERY}/05_SCRIPTS_USED/app/"
else
  echo "[WARN] dft_engine.py not found under ${REMOTE_CODE_ROOT}/app"
fi

echo
echo "[6/6] Local audit"
AUDIT_DIR="${LOCAL_DELIVERY}/04_RUN_METADATA"
AUDIT_FILE="${AUDIT_DIR}/PROVENANCE_PULL_AUDIT_20260629.txt"
mkdir -p "${AUDIT_DIR}"

{
  echo "iService VACLM full provenance pull audit"
  echo "Generated: $(date)"
  echo
  echo "REMOTE_HOST=${REMOTE_HOST}"
  echo "REMOTE_TFVW_ROOT=${REMOTE_TFVW_ROOT}"
  echo "REMOTE_CODE_ROOT=${REMOTE_CODE_ROOT}"
  echo "LOCAL_DELIVERY=${LOCAL_DELIVERY}"
  echo
  echo "Local raw case folders:"
  find "${LOCAL_DELIVERY}/03_RAW_CASES/03_runs" -mindepth 1 -maxdepth 1 -type d | wc -l
  echo
  echo "Local result.json:"
  find "${LOCAL_DELIVERY}/03_RAW_CASES/03_runs" -name result.json | wc -l
  echo
  echo "Local .traj files:"
  find "${LOCAL_DELIVERY}/03_RAW_CASES/03_runs" -name '*.traj' | wc -l
  echo
  echo "Local relax logs:"
  find "${LOCAL_DELIVERY}/03_RAW_CASES/03_runs" \( -name 'pristine_relax.log' -o -name 'vacancy_relax.log' \) | wc -l
  echo
  echo "Local DFTpy ini inputs:"
  find "${LOCAL_DELIVERY}/03_RAW_CASES/03_runs" \( -name 'dftpy_pristine_input.ini' -o -name 'dftpy_vacancy_input.ini' \) | wc -l
  echo
  echo "Local submit snapshots:"
  find "${LOCAL_DELIVERY}/03_RAW_CASES/03_runs" -name 'submit_script_used_*' | wc -l
  echo
  echo "Code provenance:"
  ls -lh "${LOCAL_DELIVERY}/05_SCRIPTS_USED/scripts/run_dftpy_vcrelax_vacancy_matrix_one.py" 2>/dev/null || true
  ls -lh "${LOCAL_DELIVERY}/05_SCRIPTS_USED/app/dft_engine.py" 2>/dev/null || true
  echo
  echo "Sample trajectory files:"
  find "${LOCAL_DELIVERY}/03_RAW_CASES/03_runs" -name '*.traj' | head -20
} | tee "${AUDIT_FILE}"

cat > "${LOCAL_DELIVERY}/README_PROVENANCE_UPDATE_20260629.md" <<EOF
# Provenance Update 2026-06-29

This delivery package was updated by pulling the cleaned iService TFvW lambda-mu
scan provenance from:

\`\`\`text
${REMOTE_TFVW_ROOT}
\`\`\`

The update is intended to add the missing trajectory and production provenance
requested during review:

- complete raw case folders under \`03_RAW_CASES/03_runs\`
- DFTpy input files
- DFTpy output summaries
- relaxation logs
- ASE trajectory files, if present on iService
- final relaxed VASP/XYZ structures
- result.json files
- submission scripts
- Python runner and \`app/dft_engine.py\`

Local audit:

\`\`\`text
04_RUN_METADATA/PROVENANCE_PULL_AUDIT_20260629.txt
\`\`\`
EOF

echo
echo "============================================================"
echo "Done."
echo "[AUDIT] ${AUDIT_FILE}"
echo "[README] ${LOCAL_DELIVERY}/README_PROVENANCE_UPDATE_20260629.md"
echo "============================================================"
