#!/bin/bash
#SBATCH -J dftpySize
#SBATCH -A MST114175
#SBATCH -p ct56
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -o logs_ctest/%x_%j.out
#SBATCH -e logs_ctest/%x_%j.err

set -euo pipefail

ROOT="${ROOT:-/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"
SERIES_NAME="${SERIES_NAME:?SERIES_NAME is not set. Example: dftpy_vacancy_size_primitive_qe_a0_spacing020_20260511}"
SETTING="${SETTING:?SETTING is not set. Example: prim_04x04x04}"

mkdir -p "${ROOT}/logs_ctest"
cd "${ROOT}"

source /home/dawson666/miniconda3/etc/profile.d/conda.sh
conda activate dftpy-env

export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python scripts/run_dftpy_primitive_size_one.py \
  --rootdir "${ROOT}/results/${SERIES_NAME}" \
  --setting "${SETTING}"
