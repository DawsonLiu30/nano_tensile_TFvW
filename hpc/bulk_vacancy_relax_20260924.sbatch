#!/usr/bin/env bash
# Submit this new campaign through Slurm; never execute pw.x on the login node.
#SBATCH --nodes=1
#SBATCH --ntasks=29
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
: "${CAMPAIGN_SNAPSHOT:?Missing immutable source snapshot}"
: "${QE_RUN_ROOT:?Missing durable output root outside source snapshot}"
[[ "${SLURM_NTASKS:-0}" == 29 ]] || { echo 'Expected 29 MPI ranks'; exit 2; }
cases=(0V 1V 2V_1NN_D110 2V_2NN_D100 2V_3NN_D211 2V_4NN_D110 2V_5NN_D310 2V_6NN_D111)
index="${SLURM_ARRAY_TASK_ID:?Missing array index}"
[[ "$index" =~ ^[0-7]$ ]] || { echo 'Invalid case index'; exit 2; }
case_id="${cases[$index]}"
export HOSTNAME="$(hostname)"
if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
module purge
module load intel/2022
module load intelmpi/2021.11
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
qe_binary=/home/u1871490/qe-7.5/bin/pw.x
linkage="$(ldd "$qe_binary")"
[[ "$linkage" != *'not found'* ]] || { echo 'Missing QE library'; exit 2; }
[[ "$linkage" == *'/intel/'* ]] || { echo 'Expected Intel runtime'; exit 2; }
module list 2>&1
sha256sum "$qe_binary"
printf 'case=%s job=%s host=%s array=%s\n' "$case_id" "$SLURM_JOB_ID" "$HOSTNAME" "$index"
cd "$CAMPAIGN_SNAPSHOT"
exec /home/dawson666/miniconda3/bin/python3 scripts/run_bulk_vacancy_qe_20260924.py run \
  --campaign "$CAMPAIGN_SNAPSHOT/campaigns/al_bulk_vacancy_relax_20260924" \
  --case "$case_id" --run-root "$QE_RUN_ROOT" \
  --pw "srun --mpi=pmi2 -n 29 $qe_binary" --execute
