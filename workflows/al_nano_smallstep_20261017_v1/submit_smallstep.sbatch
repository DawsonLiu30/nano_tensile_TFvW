#!/bin/bash
#SBATCH --job-name=al-of-R13-finish
#SBATCH --account=mst114175
#SBATCH --partition=ct56
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${RECOVERY_SNAPSHOT:?required}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
cd "$RECOVERY_SNAPSHOT"
parent_snapshot=/work/dawson666/src/campaign_snapshots/al_nano_five_radii_of_20261015_20261015T124450Z
parent_runroot=/work/dawson666/dftpy_project/relax/AL_NANO_FIVE_RADII_CORRECTED_20261015
exec /home/dawson666/miniconda3/envs/dftpy-env/bin/python -B recover_nano.py execute \
  --parent-snapshot "$parent_snapshot" --parent-runroot "$parent_runroot" \
  --parent-attempt "$parent_runroot/R13_0V/attempt_001" \
  --output-root "$parent_runroot" --attempt attempt_002_smallstep --max-runtime-s 79200
