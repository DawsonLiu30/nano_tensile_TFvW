#!/bin/bash
#SBATCH --job-name=NCtens
#SBATCH --partition=ct56
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --mem=96G
#SBATCH --account=MST114175
#SBATCH --array=0-2%3
#SBATCH --output=logs_nanocrystal_pilot/NC_TENS_%A_%a.out
#SBATCH --error=logs_nanocrystal_pilot/NC_TENS_%A_%a.err

set -euo pipefail

ROOT="/gpfs-work/dawson666/dftpy_project/relax/dftpy45"
PYTHON="${PYTHON:-/home/dawson666/miniconda3/envs/dftpy-env/bin/python}"
POSITIONS=(inner middle outer)
POS="${POSITIONS[$SLURM_ARRAY_TASK_ID]}"
DIAMETER_NM="2.0"
ORIENTATION="111"
SHAPE="hexagon"
CASE="nanocrystal_${SHAPE}_periodic_${ORIENTATION}_${DIAMETER_NM}nm_vac_${POS}_tfvw"
RUN_NAME="nanocrystal_${SHAPE}_r2_vac_${POS}_tfvw"
WORKDIR="cases/$CASE"
INIT="$WORKDIR/inputs/vacancy_equilibrium.vasp"

cd "$ROOT"
mkdir -p logs_nanocrystal_pilot

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "============================================================"
echo "[INFO] Nanocrystal tensile pilot"
echo "============================================================"
echo "[INFO] Host        : $(hostname)"
echo "[INFO] Start       : $(date)"
echo "[INFO] ROOT        : $ROOT"
echo "[INFO] PYTHON      : $PYTHON"
echo "[INFO] Case        : $CASE"
echo "[INFO] Run name    : $RUN_NAME"
echo "[INFO] Vacancy pos : $POS"
echo "[INFO] Boundary    : periodic infinite z"
echo "[INFO] Stress      : area-corrected axial Cauchy wire stress"
echo "[INFO] Step/cycles : 0.01 / 50"

if [ ! -f "$INIT" ]; then
  echo "[ERROR] Missing VESTA-checked input: $INIT"
  echo "[ERROR] Run submit_nanocrystal_vacancy_prepare_pilot_20260526.sbatch first."
  exit 1
fi

"$PYTHON" scripts/run_periodic_tensile.py \
  --case "$RUN_NAME" \
  --workdir "$WORKDIR" \
  --init "inputs/vacancy_equilibrium.vasp" \
  --spacing 0.20 \
  --kedf TFVW \
  --step 0.01 \
  --cycles 50 \
  --fmax 0.002 \
  --relax-steps 1000 \
  --plot-summary

echo "[INFO] End         : $(date)"

