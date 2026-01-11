#!/bin/bash
#SBATCH -J Al300Vac_TFvW_break
#SBATCH -p ct56
#SBATCH -A mst114175
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 3:00:00
#SBATCH -o slurm/%x_%j.out
#SBATCH -e slurm/%x_%j.err

set -e

WORKDIR=/work/dawson666/dftpy_project/nano_tensile_Al300Vac/01_tensile

mkdir -p "$WORKDIR/slurm"
cd "$WORKDIR"

echo "=== Job start: $(date) ==="
echo "Host: $(hostname)"
echo "PWD : $(pwd)"
echo "JOB : $SLURM_JOB_ID"
echo "CPU : ${SLURM_CPUS_PER_TASK:-1}"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dftpy-env

python main.py \
  --workdir . \
  --init init.xyz \
  --pp /work/dawson666/dftpy_project/PSP/al.gga.recpot \
  --spacing 0.50 \
  --step 0.03 \
  --cycles 200 \
  --fmax 0.8 \
  --relax-steps 2

echo "=== Job end: $(date) ==="

