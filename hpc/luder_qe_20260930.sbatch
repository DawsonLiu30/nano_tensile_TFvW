#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=29
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${SLURM_ARRAY_TASK_ID:?}" "${CAMPAIGN_SNAPSHOT:?}" "${QE_RUN_ROOT:?}"
export HOSTNAME="$(hostname)"
if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
module purge
module load intel/2022
module load intelmpi/2021.11
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
binary=/home/u1871490/qe-7.5/bin/pw.x
linkage="$(ldd "$binary")"
[[ "$linkage" != *'not found'* && "$linkage" == *'/intel/'* ]]
[[ "${SLURM_NTASKS:-0}" == 29 ]]
cd "$CAMPAIGN_SNAPSHOT"
exec /home/dawson666/miniconda3/bin/python3 -B scripts/run_luder_qe_20260930.py run \
 --campaign "$CAMPAIGN_SNAPSHOT/campaigns/al_luder_validation_nano_20260930" \
 --index "$SLURM_ARRAY_TASK_ID" --root "$QE_RUN_ROOT"
