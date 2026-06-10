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

read -r EXPECTED DONE <<<"$(ssh "${REMOTE_HOST}" "
set -euo pipefail
cd '${REMOTE_ROOT}'
SETTING_FILE='results/${SERIES_NAME}/settings_weight_scan.txt'
EXPECTED=\$(awk 'NF {count++} END {print count+0}' \"\${SETTING_FILE}\")
DONE=\$(find 'results/${SERIES_NAME}/weight_scan' -name result.json -type f -size +0c | wc -l)
printf '%s %s\n' \"\${EXPECTED}\" \"\${DONE}\"
")"

echo "[AUDIT ] expected=${EXPECTED}, result.json=${DONE}"
if [[ "${DONE}" -lt "${EXPECTED}" && "${ALLOW_INCOMPLETE}" != "1" ]]; then
  echo "[STOP] The remote series is not complete yet." >&2
  echo "       Wait for rescue jobs, then rerun this script." >&2
  echo "       For a partial snapshot, set ALLOW_INCOMPLETE=1." >&2
  exit 3
fi

mkdir -p "${LOCAL_SERIES}"

# Mirror the full series without excluding trajectories, raw DFTpy output,
# structures, manifests, timeout markers, or result files.
rsync -avhP --partial \
  "${REMOTE_HOST}:${REMOTE_SERIES}/" \
  "${LOCAL_SERIES}/"

mkdir -p "${LOCAL_SERIES}/_scheduler_logs"
rsync -avhP --partial \
  --include='dftpyLM*' \
  --include='*dftpyLM*' \
  --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_ROOT}/logs_ctest/" \
  "${LOCAL_SERIES}/_scheduler_logs/" || true

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
  --include='/al.lda.recpot' \
  --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_ROOT}/" \
  "${LOCAL_SERIES}/_reproducibility/" || true

cd "${LOCAL_REPO}"
if command -v python3 >/dev/null 2>&1 \
  && python3 -c 'import numpy, matplotlib' >/dev/null 2>&1; then
  python3 scripts/collect_dftpy_tfvw_lambda_mu_vacancy_matrix.py \
    --rootdir "${LOCAL_SERIES}"
elif command -v python.exe >/dev/null 2>&1 \
  && python.exe -c 'import numpy, matplotlib' >/dev/null 2>&1; then
  COLLECTOR_WIN=$(wslpath -w \
    "${LOCAL_REPO}/scripts/collect_dftpy_tfvw_lambda_mu_vacancy_matrix.py")
  LOCAL_SERIES_WIN=$(wslpath -w "${LOCAL_SERIES}")
  python.exe "${COLLECTOR_WIN}" --rootdir "${LOCAL_SERIES_WIN}"
else
  echo "[ERROR] Python with numpy and matplotlib is required." >&2
  exit 127
fi

echo
echo "============================================================"
echo "Pull and local collection completed"
echo "============================================================"
echo "[RAW     ] ${LOCAL_SERIES}"
echo "[SUMMARY ] ${LOCAL_SERIES}/analysis/tables/lambda_mu_vacancy_long_summary.csv"
echo "[AUDIT   ] ${LOCAL_SERIES}/analysis/COMPLETION_AUDIT.md"
echo "[FIGURES ] ${LOCAL_SERIES}/analysis/figures"
