#!/bin/bash
#SBATCH --job-name=al-of-fiveR
#SBATCH --account=mst114175
#SBATCH --partition=ct56
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-7%8
#SBATCH --no-requeue
set -euo pipefail
: "${NANO_SNAPSHOT:?required}"
: "${NANO_RUN_ROOT:?required}"
: "${SLURM_ARRAY_TASK_ID:?required}"
cases=(R11_0V R11_1V R13_0V R13_1V R17_0V R17_1V R20_0V R20_1V)
case_id="${cases[$SLURM_ARRAY_TASK_ID]}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
cd "$NANO_SNAPSHOT"
exec /home/dawson666/miniconda3/envs/dftpy-env/bin/python -B scripts/gated_nano.py run \
  --case "$case_id" --gate "$NANO_RUN_ROOT/gate_pass.json" \
  --output-root "$NANO_RUN_ROOT" --max-runtime-s 79200
