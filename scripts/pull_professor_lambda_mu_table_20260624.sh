#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-dawson666@twnia3.nchc.org.tw}"
REMOTE_BASE="${REMOTE_BASE:-/work/dawson666/dftpy_project/relax/dftpy45/results/Al_defects/01_calibration/single_vacancy/dftpy_tfvw_lambda_mu/coarse_10x10_vacancy_formation}"
REMOTE_TAR="${REMOTE_TAR:-${REMOTE_BASE}/professor_table_gillan_style_20260624.tar.gz}"
REMOTE_TABLE_DIR="${REMOTE_TABLE_DIR:-${REMOTE_BASE}/09_professor_table}"
LOCAL_DIR="${LOCAL_DIR:-/mnt/c/Users/dawso/Desktop/PROFESSOR_LAMMU_TABLE_GILLAN_STYLE_20260624}"

echo "============================================================"
echo "Pull professor lambda-mu Gillan-style table"
echo "============================================================"
echo "[REMOTE] ${REMOTE_HOST}:${REMOTE_BASE}"
echo "[LOCAL ] ${LOCAL_DIR}"
echo

mkdir -p "${LOCAL_DIR}"

echo "============================================================"
echo "[1/3] Pull tarball"
echo "============================================================"
rsync -avhP --partial \
  "${REMOTE_HOST}:${REMOTE_TAR}" \
  "${LOCAL_DIR}/"

echo
echo "============================================================"
echo "[2/3] Pull loose table files as a backup"
echo "============================================================"
mkdir -p "${LOCAL_DIR}/09_professor_table"
rsync -avhP --partial \
  "${REMOTE_HOST}:${REMOTE_TABLE_DIR}/" \
  "${LOCAL_DIR}/09_professor_table/"

echo
echo "============================================================"
echo "[3/3] Local file check"
echo "============================================================"
ls -lh "${LOCAL_DIR}"
echo
ls -lh "${LOCAL_DIR}/09_professor_table"

XLSX="${LOCAL_DIR}/09_professor_table/professor_lambda_mu_table_gillan_style.xlsx"
if [[ -s "${XLSX}" ]]; then
  echo
  echo "[OK] XLSX pulled:"
  echo "     ${XLSX}"
else
  echo "[ERROR] XLSX missing or empty: ${XLSX}" >&2
  exit 2
fi

echo
echo "============================================================"
echo "Done"
echo "============================================================"
echo "${LOCAL_DIR}"
