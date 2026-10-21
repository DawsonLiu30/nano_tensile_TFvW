#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=29
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
: "${CAMPAIGN_SNAPSHOT:?Missing immutable snapshot}"
: "${QE_RUN_ROOT:?Missing output root}"
[[ "${SLURM_NTASKS:-0}" == 29 ]]
cases=(0V 1V 2V_1NN_D110 2V_2NN_D100 2V_3NN_D211 2V_4NN_D110 2V_5NN_D310 2V_6NN_D111)
index="${SLURM_ARRAY_TASK_ID:?Missing array index}"
[[ "$index" =~ ^[0-7]$ ]]
export HOSTNAME="$(hostname)"
if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
module purge
module load intel/2022
module load intelmpi/2021.11
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
qe_binary=/home/u1871490/qe-7.5/bin/pw.x
[[ "$(sha256sum "$qe_binary" | cut -d' ' -f1)" == d8e498c583ef8bc69e71ddc89c83bc33b06d583d6bc22ab39cc84e2c02bd022b ]]
linkage="$(ldd "$qe_binary")"
[[ "$linkage" != *'not found'* && "$linkage" == *'/intel/'* ]]
module list 2>&1
cd "$CAMPAIGN_SNAPSHOT"
exec /home/dawson666/miniconda3/bin/python3 -B scripts/run_bulk_vacancy_qe_vcrelax_20260925.py run \
 --campaign "$CAMPAIGN_SNAPSHOT/campaigns/al_bulk_vacancy_vcrelax_20260925" \
 --case "${cases[$index]}" --run-root "$QE_RUN_ROOT" \
 --pw "srun --mpi=pmi2 -n 29 $qe_binary" --max-seconds 169200 --execute
