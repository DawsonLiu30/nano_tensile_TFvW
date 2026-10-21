#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --no-requeue
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${SLURM_ARRAY_TASK_ID:?}" "${CAMPAIGN_SNAPSHOT:?}" "${DFTPY_RUN_ROOT:?}" "${NANO_SIZE:?}"
[[ "$NANO_SIZE" == R15 || "$NANO_SIZE" == R20 ]]
[[ "$SLURM_ARRAY_TASK_ID" == 0 || "$SLURM_ARRAY_TASK_ID" == 1 ]]
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
cd "$CAMPAIGN_SNAPSHOT"
exec /home/dawson666/miniconda3/envs/dftpy-env/bin/python -B scripts/run_nano_dftpy_pilot_20260930.py \
 --campaign "$CAMPAIGN_SNAPSHOT/campaigns/al_luder_validation_nano_20260930" \
 --case "${NANO_SIZE}_${SLURM_ARRAY_TASK_ID}V" --execute --output-root "$DFTPY_RUN_ROOT"
