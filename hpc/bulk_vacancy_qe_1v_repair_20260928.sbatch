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
exec /home/dawson666/miniconda3/bin/python3 -B scripts/run_bulk_vacancy_qe_1v_repair_20260928.py run \
 --campaign "$CAMPAIGN_SNAPSHOT/campaigns/al_bulk_vacancy_1v_repair_20260928" \
 --case 1V --run-root "$QE_RUN_ROOT" \
 --pw "srun --mpi=pmi2 -n 29 $qe_binary" --max-seconds 169200 --execute
