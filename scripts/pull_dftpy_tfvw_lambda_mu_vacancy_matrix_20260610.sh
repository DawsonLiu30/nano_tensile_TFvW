#!/usr/bin/env bash
set -euo pipefail

LOCAL_REPO="${LOCAL_REPO:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
REMOTE_HOST="${REMOTE_HOST:-iservice}"
REMOTE_ROOT="${REMOTE_ROOT:-/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"
SERIES_NAME="${SERIES_NAME:-dftpy_vacancy_tfvw_lambda_mu_matrix_10x10_conv3x3x3_lda_vcrelax_20260609}"
LOCAL_BASE="${LOCAL_BASE:-/mnt/c/Users/dawso/Desktop/DFTPY_TFVW_LAMBDA_MU_VACANCY_MATRIX_20260610}"
ALLOW_INCOMPLETE="${ALLOW_INCOMPLETE:-0}"

REMOTE_SERIES="${REMOTE_ROOT}/results/${SERIES_NAME}"
LOCAL_SERIES="${LOCAL_BASE}/${SERIES_NAME}"

echo "============================================================"
echo "Pull complete DFTpy TFvW lambda-mu vacancy matrix"
echo "============================================================"
echo "[REMOTE] ${REMOTE_HOST}:${REMOTE_SERIES}"
echo "[LOCAL ] ${LOCAL_SERIES}"
echo "[ALLOW_INCOMPLETE] ${ALLOW_INCOMPLETE}"
echo

read -r EXPECTED DONE TIMEOUT FAILED <<<"$(ssh "${REMOTE_HOST}" "
set -euo pipefail
cd '${REMOTE_ROOT}'
SETTING_FILE='results/${SERIES_NAME}/settings_weight_scan.txt'
SERIES_DIR='results/${SERIES_NAME}'
EXPECTED=\$(awk 'NF {count++} END {print count+0}' \"\${SETTING_FILE}\")
DONE=\$(find \"\${SERIES_DIR}/weight_scan\" -name result.json -type f -size +0c | wc -l)
TIMEOUT=\$(find \"\${SERIES_DIR}/weight_scan\" -name 'RESCUE_TIMEOUT_*.txt' -type f | wc -l)
FAILED=\$(find \"\${SERIES_DIR}/weight_scan\" -name 'RESCUE_FAILED.txt' -type f | wc -l)
printf '%s %s %s %s\n' \"\${EXPECTED}\" \"\${DONE}\" \"\${TIMEOUT}\" \"\${FAILED}\"
")"

echo "[AUDIT] expected=${EXPECTED}, result.json=${DONE}, timeout_markers=${TIMEOUT}, failed_markers=${FAILED}"

if [[ "${DONE}" -lt "${EXPECTED}" && "${ALLOW_INCOMPLETE}" != "1" ]]; then
  echo "[STOP] The remote series is not complete yet." >&2
  echo "       expected=${EXPECTED}, result.json=${DONE}" >&2
  echo "       Wait for rescue jobs, then rerun this script." >&2
  echo "       For a partial snapshot, run:" >&2
  echo "       ALLOW_INCOMPLETE=1 bash $0" >&2
  exit 3
fi

mkdir -p "${LOCAL_SERIES}"

echo
echo "============================================================"
echo "[1/5] Mirror full raw series"
echo "============================================================"

rsync -avhP --partial \
  "${REMOTE_HOST}:${REMOTE_SERIES}/" \
  "${LOCAL_SERIES}/"

echo
echo "============================================================"
echo "[2/5] Pull scheduler logs: ctest + ct56"
echo "============================================================"

mkdir -p "${LOCAL_SERIES}/_scheduler_logs/logs_ctest"
rsync -avhP --partial \
  --include='dftpyLM*' \
  --include='*dftpyLM*' \
  --include='auto_pump*' \
  --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_ROOT}/logs_ctest/" \
  "${LOCAL_SERIES}/_scheduler_logs/logs_ctest/" || true

mkdir -p "${LOCAL_SERIES}/_scheduler_logs/logs_ct56"
rsync -avhP --partial \
  --include='dftpyLM*' \
  --include='*dftpyLM*' \
  --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_ROOT}/logs_ct56/" \
  "${LOCAL_SERIES}/_scheduler_logs/logs_ct56/" || true

echo
echo "============================================================"
echo "[3/5] Pull reproducibility files"
echo "============================================================"

mkdir -p "${LOCAL_SERIES}/_reproducibility"
rsync -avhP --partial \
  --include='/app/' \
  --include='/app/dft_engine.py' \
  --include='/scripts/' \
  --include='/scripts/prepare_dftpy_vacancy_conventional.py' \
  --include='/scripts/run_dftpy_vcrelax_vacancy_one.py' \
  --include='/scripts/collect_dftpy_conventional_vacancy.py' \
  --include='/scripts/collect_dftpy_vcrelax_fmax.py' \
  --include='/submit_dftpy*vcrelax*' \
  --include='/auto_pump_lammu_rescue_ctest_12h.sh' \
  --include='/check_lammu_matrix_vcrelax_progress.sh' \
  --include='/al.lda.recpot' \
  --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_ROOT}/" \
  "${LOCAL_SERIES}/_reproducibility/" || true

echo
echo "============================================================"
echo "[4/5] Run local collector"
echo "============================================================"

cd "${LOCAL_REPO}"

if [[ ! -f "scripts/collect_dftpy_tfvw_lambda_mu_vacancy_matrix.py" ]]; then
  echo "[ERROR] Collector not found:" >&2
  echo "        ${LOCAL_REPO}/scripts/collect_dftpy_tfvw_lambda_mu_vacancy_matrix.py" >&2
  exit 127
fi

FALLBACK_PP="${LOCAL_REPO}/al.lda.recpot"
if [[ ! -s "${FALLBACK_PP}" ]]; then
  echo "[ERROR] Local LDA pseudopotential is missing or empty:" >&2
  echo "        ${FALLBACK_PP}" >&2
  exit 127
fi

if command -v python3 >/dev/null 2>&1 \
  && python3 -c 'import numpy, matplotlib, dftpy, ase' >/dev/null 2>&1; then
  python3 scripts/collect_dftpy_tfvw_lambda_mu_vacancy_matrix.py \
    --rootdir "${LOCAL_SERIES}" \
    --fallback-pp "${FALLBACK_PP}"
elif command -v python.exe >/dev/null 2>&1 \
  && python.exe -c 'import numpy, matplotlib, dftpy, ase' >/dev/null 2>&1; then
  COLLECTOR_WIN=$(wslpath -w \
    "${LOCAL_REPO}/scripts/collect_dftpy_tfvw_lambda_mu_vacancy_matrix.py")
  LOCAL_SERIES_WIN=$(wslpath -w "${LOCAL_SERIES}")
  FALLBACK_PP_WIN=$(wslpath -w "${FALLBACK_PP}")
  python.exe "${COLLECTOR_WIN}" \
    --rootdir "${LOCAL_SERIES_WIN}" \
    --fallback-pp "${FALLBACK_PP_WIN}"
else
  echo "[ERROR] Python with numpy, matplotlib, dftpy, and ase is required." >&2
  exit 127
fi

echo
echo "============================================================"
echo "[5/5] Local audit"
echo "============================================================"

LOCAL_DONE=$(find "${LOCAL_SERIES}/weight_scan" -name result.json -type f -size +0c | wc -l)
LOCAL_TIMEOUT=$(find "${LOCAL_SERIES}/weight_scan" -name 'RESCUE_TIMEOUT_*.txt' -type f | wc -l)
LOCAL_FAILED=$(find "${LOCAL_SERIES}/weight_scan" -name 'RESCUE_FAILED.txt' -type f | wc -l)

echo "[LOCAL AUDIT] result.json=${LOCAL_DONE}/${EXPECTED}"
echo "[LOCAL AUDIT] timeout_markers=${LOCAL_TIMEOUT}"
echo "[LOCAL AUDIT] failed_markers=${LOCAL_FAILED}"

echo
echo "[MISSING CASES]"
idx=0
SETTING_FILE="${LOCAL_SERIES}/settings_weight_scan.txt"
while read -r line; do
  [[ -z "${line}" ]] && continue
  setting=$(echo "${line}" | awk '{print $NF}')
  result="${LOCAL_SERIES}/weight_scan/${setting}/result.json"
  if [[ ! -s "${result}" ]]; then
    printf "MISSING idx=%02d  %s\n" "${idx}" "${setting}"
  fi
  idx=$((idx+1))
done < "${SETTING_FILE}"

echo
echo "[KEDF RAW CHECK]"
KEDF_HITS=$(grep -RIl \
  "KEDF-TF energy\|KEDF-VW energy\|kedfs energy" \
  "${LOCAL_SERIES}/weight_scan" \
  "${LOCAL_SERIES}/_scheduler_logs" 2>/dev/null | wc -l || true)

echo "[KEDF RAW CHECK] files containing KEDF text = ${KEDF_HITS}"

if [[ "${KEDF_HITS}" -eq 0 ]]; then
  echo "[WARN] No KEDF text found in pulled raw files/logs." >&2
  echo "       Total energy and lattice constant can still be collected from result.json." >&2
  echo "       KEDF table requires collector support for DFTpy raw output/log parsing." >&2
fi

echo
echo "[TABLE CHECK]"
if [[ -d "${LOCAL_SERIES}/analysis/tables" ]]; then
  ls -lh "${LOCAL_SERIES}/analysis/tables" || true
else
  echo "[WARN] analysis/tables was not created by the collector." >&2
fi

echo
echo "============================================================"
echo "Pull and local collection completed"
echo "============================================================"
echo "[RAW      ] ${LOCAL_SERIES}"
echo "[LOGS     ] ${LOCAL_SERIES}/_scheduler_logs"
echo "[REPRO    ] ${LOCAL_SERIES}/_reproducibility"
echo "[SUMMARY  ] ${LOCAL_SERIES}/analysis/tables/lambda_mu_vacancy_long_summary.csv"
echo "[AUDIT    ] ${LOCAL_SERIES}/analysis/COMPLETION_AUDIT.md"
echo "[FIGURES  ] ${LOCAL_SERIES}/analysis/figures"
echo "============================================================"
