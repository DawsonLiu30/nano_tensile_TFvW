#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
: "${CAMPAIGN_SNAPSHOT:?Missing immutable snapshot}"
: "${DFTPY_RUN_ROOT:?Missing output root}"
case "${SLURM_ARRAY_TASK_ID:?Missing group index}" in
 0) cases=(0V 2V_3NN_D211 2V_6NN_D111) ;;
 1) cases=(1V 2V_4NN_D110) ;;
 2) cases=(2V_1NN_D110 2V_2NN_D100 2V_5NN_D310) ;;
 *) exit 2 ;;
esac
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
runtime_python=/home/dawson666/miniconda3/envs/dftpy-env/bin/python
cd "$CAMPAIGN_SNAPSHOT"
start_seconds=$SECONDS
overall_status=0
for case_id in "${cases[@]}"; do
 remaining=$((6600 - (SECONDS - start_seconds)))
 if (( remaining < 120 )); then
  printf 'Not started: %s; remaining group wall budget %s seconds\n' "$case_id" "$remaining"
  overall_status=1
  break
 fi
 "$runtime_python" -B scripts/run_bulk_vacancy_dftpy_vcrelax_20260925.py \
   --campaign "$CAMPAIGN_SNAPSHOT/campaigns/al_bulk_vacancy_vcrelax_20260925" \
   --case "$case_id" --output-root "$DFTPY_RUN_ROOT" --attempt attempt_001 \
   --threads 1 --max-runtime-s "$remaining" --execute || overall_status=1
done
exit "$overall_status"
