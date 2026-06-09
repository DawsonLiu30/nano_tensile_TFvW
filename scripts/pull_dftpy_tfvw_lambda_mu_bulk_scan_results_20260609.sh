#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-iservice}"
REMOTE_ROOT="${REMOTE_ROOT:-/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"
SERIES_NAME="${SERIES_NAME:-dftpy_tfvw_lambda_mu_bulk_10x10_20260609}"
LOCAL_BASE="${LOCAL_BASE:-/mnt/c/Users/dawso/Desktop/DFTPY_TFVW_LAMBDA_MU_BULK_SCAN_20260609}"
LOCAL_SERIES="${LOCAL_BASE}/${SERIES_NAME}"

PYTHON_MODE=""
if command -v python3 >/dev/null 2>&1 \
  && python3 -c 'import numpy, matplotlib' >/dev/null 2>&1; then
  PYTHON=python3
  PYTHON_MODE=linux
elif command -v python >/dev/null 2>&1 \
  && python -c 'import numpy, matplotlib' >/dev/null 2>&1; then
  PYTHON=python
  PYTHON_MODE=linux
elif command -v python.exe >/dev/null 2>&1 \
  && python.exe -c 'import numpy, matplotlib' >/dev/null 2>&1; then
  PYTHON=python.exe
  PYTHON_MODE=windows
else
  echo "[ERROR] Python with numpy and matplotlib is required for local collection." >&2
  exit 127
fi

mkdir -p "${LOCAL_SERIES}"

echo "============================================================"
echo "Pull DFTpy TF+vW lambda-mu bulk scan"
echo "============================================================"
echo "[REMOTE] ${REMOTE_HOST}:${REMOTE_ROOT}/results/${SERIES_NAME}"
echo "[LOCAL ] ${LOCAL_SERIES}"

rsync -avhP \
  "${REMOTE_HOST}:${REMOTE_ROOT}/results/${SERIES_NAME}/" \
  "${LOCAL_SERIES}/"

if [[ "${PYTHON_MODE}" == "windows" ]]; then
  COLLECTOR_WIN=$(wslpath -w "$(pwd)/scripts/collect_dftpy_tfvw_lambda_mu_bulk_scan.py")
  LOCAL_SERIES_WIN=$(wslpath -w "${LOCAL_SERIES}")
  "${PYTHON}" "${COLLECTOR_WIN}" --rootdir "${LOCAL_SERIES_WIN}"
else
  "${PYTHON}" scripts/collect_dftpy_tfvw_lambda_mu_bulk_scan.py \
    --rootdir "${LOCAL_SERIES}"
fi

echo
echo "Main professor table:"
echo "  ${LOCAL_SERIES}/tables/professor_three_panel_lambda_mu_table.csv"
