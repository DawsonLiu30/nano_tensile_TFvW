#!/bin/bash
#SBATCH --job-name=al-of-stressgate
#SBATCH --account=mst114175
#SBATCH --partition=ctest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${NANO_SNAPSHOT:?required}"
: "${NANO_RUN_ROOT:?required}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
cd "$NANO_SNAPSHOT"
exec /home/dawson666/miniconda3/envs/dftpy-env/bin/python -B scripts/gated_nano.py validate \
  --output-root "$NANO_RUN_ROOT" --max-runtime-s 6600
