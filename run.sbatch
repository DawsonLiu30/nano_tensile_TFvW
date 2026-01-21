#!/bin/bash
#SBATCH -J TFvW_Tensile
#SBATCH -A MST114175
#SBATCH -p ctest
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 2:00:00
#SBATCH -o slurm/%x_%j.out
#SBATCH -e slurm/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm

echo "=== Job start: $(date) ==="
echo "Host: $(hostname)"
echo "PWD : $(pwd)"
echo "JOB : ${SLURM_JOB_ID}"
echo "CPU : ${SLURM_CPUS_PER_TASK:-1}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

source ~/miniconda3/etc/profile.d/conda.sh
# source ~/miniforge3/etc/profile.d/conda.sh

conda activate dftpy-env

echo "=== Sanity check files ==="
ls -lh vac_one.vasp al.gga.psp bottom_idx.npy top_idx.npy main.py strain_engine.py dft_engine.py

echo "=== Clean old results (fresh run) ==="
rm -rf results
mkdir -p results

echo "=== Run tensile (1% per cycle, pinned grips by default) ==="
python -u main.py \
  --workdir . \
  --init vac_one.vasp \
  --pp al.gga.psp \
  --ecut 1000 \
  --step 0.01 \
  --cycles 200 \
  --relax-steps 80 \
  --fmax 0.08

echo "=== Job end: $(date) ==="

