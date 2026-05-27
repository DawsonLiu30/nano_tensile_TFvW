#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
REMOTE_ROOT="${REMOTE_ROOT:-iservice:/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"

echo "============================================================"
echo "Push nanocrystal tensile pilot scripts to iservice"
echo "============================================================"
echo "[LOCAL ] $LOCAL_ROOT"
echo "[REMOTE] $REMOTE_ROOT"
echo

rsync -avhP \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  "$LOCAL_ROOT/app/" "$REMOTE_ROOT/app/"

rsync -avhP \
  "$LOCAL_ROOT/scripts/prepare_vacancy_periodic_wire.py" \
  "$LOCAL_ROOT/scripts/run_periodic_tensile.py" \
  "$LOCAL_ROOT/scripts/run_vacancy_periodic_series.py" \
  "$REMOTE_ROOT/scripts/"

rsync -avhP \
  "$LOCAL_ROOT/submit_nanocrystal_vacancy_prepare_pilot_20260526.sbatch" \
  "$LOCAL_ROOT/submit_nanocrystal_tensile_pilot_20260526.sbatch" \
  "$REMOTE_ROOT/"

echo
echo "============================================================"
echo "Next remote commands"
echo "============================================================"
echo "cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45 && sbatch submit_nanocrystal_vacancy_prepare_pilot_20260526.sbatch"
echo "# After pulling/checking the three VESTA structures:"
echo "cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45 && sbatch submit_nanocrystal_tensile_pilot_20260526.sbatch"
