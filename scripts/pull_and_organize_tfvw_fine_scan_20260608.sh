#!/usr/bin/env bash
set -euo pipefail

LOCAL_REPO="${LOCAL_REPO:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
REMOTE_HOST="${REMOTE_HOST:-iservice}"
REMOTE_ROOT="${REMOTE_ROOT:-/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"
SERIES_NAME="${SERIES_NAME:-dftpy_vacancy_tfvw_weight_fine_y0115_020_conv3x3x3_lda_20260605_v2}"
LOCAL_BASE="${LOCAL_BASE:-/mnt/c/Users/dawso/Desktop/DFTPY_TFVW_WEIGHT_SCAN_20260605}"
PACKAGE_DIR="${PACKAGE_DIR:-/mnt/c/Users/dawso/Desktop/TFVW_FINE_SCAN_ORGANIZED_20260608}"
TARGET_EF="${TARGET_EF:-0.601167}"
FORCE_TARGET="${FORCE_TARGET:-0.002}"

REMOTE_SERIES="${REMOTE_ROOT}/results/${SERIES_NAME}/"
LOCAL_SERIES="${LOCAL_BASE}/${SERIES_NAME}"

cat <<EOF
============================================================
Pull and organize DFTpy TFvW fine scan
============================================================
[REMOTE ] ${REMOTE_HOST}:${REMOTE_SERIES}
[LOCAL  ] ${LOCAL_SERIES}
[PACKAGE] ${PACKAGE_DIR}
[TARGET ] ${TARGET_EF} eV
[FMAX   ] ${FORCE_TARGET} eV/A
EOF

mkdir -p "${LOCAL_SERIES}"
rsync -avhP "${REMOTE_HOST}:${REMOTE_SERIES}" "${LOCAL_SERIES}/"

echo
echo "[1/3] Rebuild local collector summaries"
cd "${LOCAL_REPO}"
python scripts/collect_dftpy_conventional_vacancy.py --rootdir "${LOCAL_SERIES}"
python scripts/collect_dftpy_vcrelax_fmax.py "${LOCAL_SERIES}" --out "${LOCAL_SERIES}/dftpy_weight_actual_fmax_summary.csv"

echo
echo "[2/3] Organize final package"
python scripts/summarize_tfvw_fine_scan_20260608.py \
  --rootdir "${LOCAL_SERIES}" \
  --outdir "${PACKAGE_DIR}" \
  --target-ef "${TARGET_EF}" \
  --force-target "${FORCE_TARGET}"

echo
echo "[3/3] Package file list"
find "${PACKAGE_DIR}" -maxdepth 3 -type f | sort

cat <<EOF

============================================================
Done
============================================================
Raw local data:
  ${LOCAL_SERIES}

Organized package:
  ${PACKAGE_DIR}

Key outputs:
  ${PACKAGE_DIR}/tables/tfvw_fine_scan_clean_summary.csv
  ${PACKAGE_DIR}/tables/tfvw_fine_scan_recommendations.csv
  ${PACKAGE_DIR}/figures/tfvw_fine_scan_Ef_vs_y.png
  ${PACKAGE_DIR}/notes/TFVW_FINE_SCAN_ANALYSIS.md
============================================================
EOF

