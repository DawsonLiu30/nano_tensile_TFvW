#!/bin/bash
#SBATCH --job-name=NCprep
#SBATCH --partition=ct56
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=96G
#SBATCH --account=MST114175
#SBATCH --array=0-2%3
#SBATCH --output=logs_nanocrystal_pilot/NC_PREP_%A_%a.out
#SBATCH --error=logs_nanocrystal_pilot/NC_PREP_%A_%a.err

set -euo pipefail

ROOT="/gpfs-work/dawson666/dftpy_project/relax/dftpy45"
PYTHON="${PYTHON:-/home/dawson666/miniconda3/envs/dftpy-env/bin/python}"
POSITIONS=(inner middle outer)
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
echo "[INFO] Nanocrystal vacancy prepare pilot"
echo "============================================================"
echo "[INFO] Host        : $(hostname)"
echo "[INFO] Start       : $(date)"
echo "[INFO] ROOT        : $ROOT"
echo "[INFO] PYTHON      : $PYTHON"
echo "[INFO] Case        : $CASE"
echo "[INFO] Shape       : $SHAPE"
echo "[INFO] Orientation : $ORIENTATION"
echo "[INFO] Diameter nm : $DIAMETER_NM"
echo "[INFO] Vacancy pos : $POS"
echo "[INFO] Boundary    : periodic infinite z"
echo "[INFO] min Lz      : 12.0 A (> 10 A)"

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
  --relax-steps 1000

echo "[INFO] End         : $(date)"
echo "[INFO] VESTA-check these files before tensile:"
echo "       $ROOT/cases/$CASE/inputs/pristine_short_equilibrium.vasp"
echo "       $ROOT/cases/$CASE/inputs/vacancy_short_start.vasp"
echo "       $ROOT/cases/$CASE/inputs/vacancy_equilibrium.vasp"
