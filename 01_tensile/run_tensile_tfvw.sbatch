#!/bin/bash
#SBATCH -J Al300Vac_gapMid
#SBATCH -p ct56
#SBATCH -A mst114175
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 18:00:00
#SBATCH -o /work/dawson666/dftpy_project/nano_tensile_Al300Vac/02_results/slurm_Al300Vac_TFvW_gapMid_%j.out
#SBATCH -e /work/dawson666/dftpy_project/nano_tensile_Al300Vac/02_results/slurm_Al300Vac_TFvW_gapMid_%j.err

set -e
set +u

echo "=== Job start: $(date) ==="
echo "Host: $(hostname)"
echo "PWD : $(pwd)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"

# threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# activate env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dftpy-env

WORKDIR=/work/dawson666/dftpy_project/nano_tensile_Al300Vac/01_tensile
SCRIPT=$WORKDIR/main.py

INPUT=/work/dawson666/dftpy_project/nano_tensile_Al300Vac/00_gen/Al_supercell300_vac5_vacancy.xyz
PPPATH=/work/dawson666/dftpy_project/nano_tensile_Al300Vac/PSP
PPFILE=al.gga.recpot
OUTDIR=/work/dawson666/dftpy_project/nano_tensile_Al300Vac/02_results/run_Al300Vac_TFvW_gapMid

mkdir -p "$OUTDIR"

echo "SCRIPT=$SCRIPT"
echo "INPUT =$INPUT"
echo "PPPATH=$PPPATH/$PPFILE"
echo "OUTDIR=$OUTDIR"

python3 "$SCRIPT" \
  --input "$INPUT" \
  --outdir "$OUTDIR" \
  --pppath "$PPPATH" \
  --ppfile "$PPFILE" \
  --element Al \
  --kedf TFvW \
  --spacing 0.35 \
  --optimizer BFGS \
  --fmax 0.08 \
  --grip_thickness 3.0 \
  --stretch 1.01 \
  --relax_steps 80 \
  --max_cycles 200 \
  --fracture_mode gap_mid \
  --gap_window 2.0 \
  --gap_break 3.0

echo "=== Job end: $(date) ==="

