#!/usr/bin/env bash
set -euo pipefail

# Run this from WSL after the r=3 ct56 job has produced results:
#   wsl bash /mnt/c/Users/dawso/nano_tensile_TFvW/manual_runs/fetch_r3_ct56_results_from_wsl.sh

LOCAL_ROOT="${LOCAL_ROOT:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
REMOTE="${REMOTE:-iservice}"
REMOTE_ROOT="${REMOTE_ROOT:-/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"
CASE="finite_grip_111_3.0nm_vacancy_tfvw"

mkdir -p "${LOCAL_ROOT}/cases/${CASE}" "${LOCAL_ROOT}/results/professor_review" "${LOCAL_ROOT}/results/hpc_status"

echo "============================================================"
echo "[fetch-r3] remote : ${REMOTE}:${REMOTE_ROOT}"
echo "[fetch-r3] local  : ${LOCAL_ROOT}"
echo "============================================================"

rsync -av --info=progress2 "${REMOTE}:${REMOTE_ROOT}/cases/${CASE}/results/" "${LOCAL_ROOT}/cases/${CASE}/results/"
rsync -av --info=progress2 "${REMOTE}:${REMOTE_ROOT}/results/professor_review/r3"* "${LOCAL_ROOT}/results/professor_review/" || true
rsync -av --info=progress2 "${REMOTE}:${REMOTE_ROOT}/results/hpc_status/r3"* "${LOCAL_ROOT}/results/hpc_status/" || true

cd "${LOCAL_ROOT}"
python scripts/analyze_tensile_events.py --case-dir "cases/${CASE}" || true
python scripts/plot_grip_tensile_curve.py \
  --case-dir "cases/${CASE}" \
  --cycles-target 80 \
  --out "results/professor_review/r3_finite_grip_vacancy_tensile.png" \
  --copy-csv

echo "[fetch-r3] Local plot: ${LOCAL_ROOT}/results/professor_review/r3_finite_grip_vacancy_tensile.png"

