#!/usr/bin/env bash
set -euo pipefail

LOCAL_REPO="${LOCAL_REPO:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
REMOTE_HOST="${REMOTE_HOST:-iservice}"
REMOTE_ROOT="${REMOTE_ROOT:-/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"
SERIES_NAME="${SERIES_NAME:-dftpy_vacancy_tfvw_weight_conv3x3x3_lda_20260605}"
LOCAL_BASE="${LOCAL_BASE:-/mnt/c/Users/dawso/Desktop/DFTPY_TFVW_WEIGHT_SCAN_20260605}"

REMOTE_SERIES="${REMOTE_ROOT}/results/${SERIES_NAME}/"
LOCAL_SERIES="${LOCAL_BASE}/${SERIES_NAME}/"

echo "============================================================"
echo "Pull DFTpy TF/vW weight-scan results"
echo "============================================================"
echo "[REMOTE] ${REMOTE_HOST}:${REMOTE_SERIES}"
echo "[LOCAL ] ${LOCAL_SERIES}"

mkdir -p "${LOCAL_SERIES}"
rsync -avhP "${REMOTE_HOST}:${REMOTE_SERIES}" "${LOCAL_SERIES}"

echo
echo "[LOCAL] Collect summaries"
cd "${LOCAL_REPO}"
python scripts/collect_dftpy_conventional_vacancy.py --rootdir "${LOCAL_SERIES}"
python scripts/collect_dftpy_vcrelax_fmax.py "${LOCAL_SERIES}" --out "${LOCAL_SERIES}/dftpy_weight_actual_fmax_summary.csv"

echo
echo "[SUMMARY]"
cat "${LOCAL_SERIES}/dftpy_conventional_weight_summary.csv"
echo
echo "[ACTUAL FMAX]"
cat "${LOCAL_SERIES}/dftpy_weight_actual_fmax_summary.csv"

echo
echo "============================================================"
echo "Done:"
echo "  ${LOCAL_SERIES}/dftpy_conventional_weight_summary.csv"
echo "  ${LOCAL_SERIES}/dftpy_weight_actual_fmax_summary.csv"
echo "  ${LOCAL_SERIES}/dftpy_conventional_weight_Ef.png"
echo "============================================================"
