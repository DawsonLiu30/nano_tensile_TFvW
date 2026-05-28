#!/bin/bash
#SBATCH --job-name=NCfix
#SBATCH --partition=ct56
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --mem=96G
#SBATCH --account=MST114175
#SBATCH --array=0-1%2
#SBATCH --output=logs_nanocrystal_pilot/NC_FIX_%A_%a.out
#SBATCH --error=logs_nanocrystal_pilot/NC_FIX_%A_%a.err

set -euo pipefail

ROOT="/gpfs-work/dawson666/dftpy_project/relax/dftpy45"
PYTHON="${PYTHON:-/home/dawson666/miniconda3/envs/dftpy-env/bin/python}"
POSITIONS=(middle outer)
POS="${POSITIONS[$SLURM_ARRAY_TASK_ID]}"
DIAMETER_NM="2.0"
ORIENTATION="111"
SHAPE="hexagon"
CASE="nanocrystal_${SHAPE}_periodic_${ORIENTATION}_${DIAMETER_NM}nm_vac_${POS}_tfvw"

cd "$ROOT"
mkdir -p logs_nanocrystal_pilot

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "============================================================"
echo "[INFO] Nanocrystal vacancy prepare repair"
echo "============================================================"
echo "[INFO] Host        : $(hostname)"
echo "[INFO] Start       : $(date)"
echo "[INFO] ROOT        : $ROOT"
echo "[INFO] Case        : $CASE"
echo "[INFO] Vacancy pos : $POS"
echo "[INFO] Reason      : rerun only middle/outer because initial fmax did not meet 0.002 eV/A"
echo "[INFO] relax steps : 3000"

if ! "$PYTHON" scripts/prepare_vacancy_periodic_wire.py --help | grep -q -- "--cross-section-shape"; then
  echo "[ERROR] Remote scripts/prepare_vacancy_periodic_wire.py is stale."
  echo "[ERROR] Re-run: bash scripts/push_nanocrystal_tensile_pilot_to_iservice.sh"
  exit 2
fi

"$PYTHON" scripts/prepare_vacancy_periodic_wire.py \
  --case "$CASE" \
  --diameter-nm "$DIAMETER_NM" \
  --cross-section-shape "$SHAPE" \
  --orientation "$ORIENTATION" \
  --vacancy-radial-position "$POS" \
  --min-short-lz 12.0 \
  --vacuum 10.0 \
  --target-long-lz 200.0 \
  --spacing 0.20 \
  --kedf TFVW \
  --fmax 0.002 \
  --relax-steps 3000

echo "[INFO] End         : $(date)"

