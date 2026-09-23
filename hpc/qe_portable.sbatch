#!/usr/bin/env bash
# Site-neutral template. Inspect site resources and explicitly submit this file.
# No account, partition, module stack or QE binary is assumed here.
#SBATCH --job-name=al-divac-qe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

: "${PROJECT_ROOT:?Export an absolute checkout path as PROJECT_ROOT}"
: "${QE_WORK_ROOT:?Export an absolute durable output path as QE_WORK_ROOT}"
: "${QE_CASE:?Export a campaign case ID as QE_CASE}"
: "${PW_COMMAND:?Export the verified QE/MPI launch command as PW_COMMAND}"

[[ "$PROJECT_ROOT" = /* && "$QE_WORK_ROOT" = /* ]] || {
  echo '[ERROR] PROJECT_ROOT and QE_WORK_ROOT must be absolute Linux paths.' >&2
  exit 2
}
project_root="$(realpath -e -- "$PROJECT_ROOT")"
work_root="$(realpath -m -- "$QE_WORK_ROOT")"
[[ -f "$project_root/scripts/qe_portable.py" ]] || {
  echo '[ERROR] PROJECT_ROOT does not contain the portable QE runner.' >&2
  exit 2
}
case "$work_root" in
  /|"$project_root"|"$project_root"/*)
    echo '[ERROR] QE_WORK_ROOT must be a separate output directory outside the checkout.' >&2
    exit 2
    ;;
esac

# Optional, user-created, ignored file: activate the environment or site modules.
# Do not store passwords/tokens here. PW_COMMAND must match the loaded QE build.
if [[ -f "$project_root/env/local-qe.sh" ]]; then
  source "$project_root/env/local-qe.sh"
fi

python_command="${QE_PYTHON:-python}"
command -v "$python_command" >/dev/null || {
  echo '[ERROR] Python is unavailable; activate the environment in env/local-qe.sh.' >&2
  exit 2
}

# Keep at least 10 minutes for final output/cleanup before the scheduler limit.
# If sbatch --time is changed, explicitly update QE_JOB_SECONDS as well.
job_seconds="${QE_JOB_SECONDS:-43200}"
max_seconds="${QE_MAX_SECONDS:-39600}"
for value in "$job_seconds" "$max_seconds"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo '[ERROR] QE_JOB_SECONDS and QE_MAX_SECONDS must be positive integers.' >&2
    exit 2
  }
done
(( max_seconds <= job_seconds - 600 )) || {
  echo '[ERROR] QE_MAX_SECONDS must be at least 600 seconds below QE_JOB_SECONDS.' >&2
  exit 2
}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p -- "$work_root"
cd -- "$project_root"
printf '[PLAN] case=%s tasks=%s OMP=%s QE_max_seconds=%s\n' \
  "$QE_CASE" "${SLURM_NTASKS:-1}" "$OMP_NUM_THREADS" "$max_seconds"
printf '[PATHS] project=%s work=%s\n' "$project_root" "$work_root"

# The runner uses an argument list, not shell eval, for PW_COMMAND.
# This explicit --execute is only reached when the user runs/submits this script.
exec "$python_command" scripts/qe_portable.py run \
  --campaign campaigns/qe_divacancy_20260923 \
  --case "$QE_CASE" \
  --work-root "$work_root" \
  --pw "$PW_COMMAND" \
  --threads "$OMP_NUM_THREADS" \
  --max-seconds "$max_seconds" \
  --execute
