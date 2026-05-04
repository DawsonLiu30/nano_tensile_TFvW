#!/usr/bin/env bash
set -euo pipefail

# Run this from WSL, or from PowerShell with:
#   wsl bash /mnt/c/Users/dawso/nano_tensile_TFvW/manual_runs/submit_r3_ct56_from_wsl.sh
#
# The iService login may ask for password/OTP. Codex cannot answer that prompt,
# but this script keeps the actual sync and sbatch commands in one place.

LOCAL_ROOT="${LOCAL_ROOT:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
REMOTE="${REMOTE:-iservice}"
REMOTE_ROOT="${REMOTE_ROOT:-/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"
CYCLES="${CYCLES:-80}"
DIAMETER="${DIAMETER:-3}"

cd "${LOCAL_ROOT}"

echo "============================================================"
echo "[submit-r3] local root  : ${LOCAL_ROOT}"
echo "[submit-r3] remote      : ${REMOTE}:${REMOTE_ROOT}"
echo "[submit-r3] diameter    : ${DIAMETER}"
echo "[submit-r3] cycles      : ${CYCLES}"
echo "============================================================"

rsync -av --info=progress2 \
  --exclude ".git/" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude "logs_manual/" \
  "${LOCAL_ROOT}/" "${REMOTE}:${REMOTE_ROOT}/"

ssh "${REMOTE}" "cd '${REMOTE_ROOT}' && mkdir -p logs_ctest results/professor_review results/hpc_status && DIAMETER='${DIAMETER}' CYCLES='${CYCLES}' sbatch run_finite_grip_one_ct56.sbatch && squeue -u dawson666"

