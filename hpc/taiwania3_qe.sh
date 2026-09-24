#!/usr/bin/env bash
# Taiwania3 profile reconstructed from the user's 2026-09-24 terminal output.
# check: inspect software + validate a Slurm request, without starting QE.
# submit CASE: repeat checks, then submit exactly one fixed-geometry SCF case.
set -euo pipefail

fail() { printf '[ERROR] %s\n' "$*" >&2; exit 2; }
action="${1:-check}"
[[ "$action" == check || "$action" == submit ]] || fail 'Usage: bash hpc/taiwania3_qe.sh check|submit [CASE]'
[[ $# -le 2 ]] || fail 'Too many arguments.'
export QE_CASE="${2:-2V_1NN_D110}"
case "$QE_CASE" in
  2V_1NN_D110|2V_2NN_D100|2V_D310_r1|2V_D110_r2) ;;
  *) fail "Unknown case: $QE_CASE" ;;
esac

export PROJECT_ROOT
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export QE_WORK_ROOT="${QE_WORK_ROOT:-/work/dawson666/qe_cases/qe_runs/AL_DIVACANCY_SCF_20260924}"
export QE_PYTHON="${QE_PYTHON:-/home/dawson666/miniconda3/bin/python3}"
qe_binary="${QE_BINARY:-/home/u1871490/qe-7.5/bin/pw.x}"
export QE_BINARY="$qe_binary"
account="${QE_ACCOUNT:-mst114175}"
partition="${QE_PARTITION:-ct56}"
[[ "$QE_WORK_ROOT" == /* && "$QE_PYTHON" == /* && "$qe_binary" == /* ]] || fail 'Work root, Python and QE binary must be absolute paths.'
QE_WORK_ROOT="$(realpath -m -- "$QE_WORK_ROOT")"
case "$QE_WORK_ROOT" in /|"$PROJECT_ROOT"|"$PROJECT_ROOT"/*) fail 'Outputs must be outside the Git checkout.' ;; esac
[[ ! -e "$PROJECT_ROOT/env/local-qe.sh" ]] || fail 'env/local-qe.sh exists; review it before using this explicit Intel profile.'
[[ -x "$QE_PYTHON" ]] || fail "Python is not executable: $QE_PYTHON"
"$QE_PYTHON" -c 'import sys; print("[PYTHON]", sys.version); assert sys.version_info >= (3,10), "Python 3.10+ required"'
cd -- "$PROJECT_ROOT"

if ! command -v module >/dev/null 2>&1; then
  [[ -r /etc/profile.d/modules.sh ]] || fail 'Module initialization was not found.'
  source /etc/profile.d/modules.sh
fi
export HOSTNAME="$(hostname)"
module purge
module load intel/2022
module load intelmpi/2021.11
module list 2>&1
[[ -x "$qe_binary" ]] || fail "Observed historical QE path is unavailable: $qe_binary"
export I_MPI_PMI_LIBRARY="${QE_PMI_LIBRARY:-/usr/lib64/libpmi2.so}"
[[ -r "$I_MPI_PMI_LIBRARY" ]] || fail "PMI2 library is unavailable: $I_MPI_PMI_LIBRARY"
for command_name in srun sbatch squeue ldd sha256sum; do
  command -v "$command_name" >/dev/null || fail "Missing command: $command_name"
done
linkage="$(ldd "$qe_binary")" || fail 'Could not inspect QE shared libraries.'
printf '%s\n' "$linkage"
[[ "$linkage" != *'not found'* ]] || fail 'QE has unresolved shared libraries.'
mpi_library="$(printf '%s\n' "$linkage" | awk '$1 ~ /^libmpi\.so/ {print $3}')"
[[ "$mpi_library" == /*/intel/* ]] || fail 'Expected dynamically linked Intel MPI QE; serial or other MPI builds are not accepted by this profile.'
"$QE_PYTHON" scripts/qe_portable.py verify
mpi_plugins="$(srun --mpi=list 2>&1)" || fail 'Cannot list Slurm MPI plugins.'
printf '%s\n' "$mpi_plugins"
[[ "$mpi_plugins" == *pmi2* ]] || fail 'Slurm does not list PMI2.'
printf '[QE BINARY] '
sha256sum -- "$qe_binary"
printf '[GIT] '
git rev-parse HEAD

# Keep the observed historical rank/memory combination for the first baseline.
# It is not a performance or memory-convergence result. No pool tuning is added.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export QE_JOB_SECONDS=43200 QE_MAX_SECONDS=39600
export PW_COMMAND
PW_COMMAND="$("$QE_PYTHON" -c 'import shlex,sys; print(shlex.join(["srun","--mpi=pmi2","-n","29",sys.argv[1]]))' "$qe_binary")"
job_name="qe-${QE_CASE}"
mkdir -p -- "$QE_WORK_ROOT/slurm-logs"
receipt="$QE_WORK_ROOT/slurm-logs/preflight-$(date -u +%Y%m%dT%H%M%SZ)-$$.txt"
{
  printf 'case=%s\naccount=%s\npartition=%s\npython=%s\nlauncher=%s\npmi_library=%s\n' \
    "$QE_CASE" "$account" "$partition" "$QE_PYTHON" "$PW_COMMAND" "$I_MPI_PMI_LIBRARY"
  printf 'git_commit='; git rev-parse HEAD
  sha256sum -- "$qe_binary" "$PROJECT_ROOT/hpc/taiwania3_qe.sh"
  module list 2>&1
  printf '%s\n' "$linkage" "$mpi_plugins"
} > "$receipt"
printf '[PREFLIGHT RECORD] %s\n' "$receipt"
existing="$(squeue -h -u "$(id -un)" -n "$job_name" -o '%i')"
if [[ -n "$existing" ]]; then
  [[ "$action" == check ]] || fail "A matching job is already queued/running: $existing"
  printf '[CHECK ONLY] Existing job %s is unchanged; no submission will occur.\n' "$existing"
fi
job_args=(--account="$account" --partition="$partition"
  --nodes=1 --ntasks=29 --cpus-per-task=1 --mem=64G --time=12:00:00 --no-requeue
  --job-name="$job_name" --chdir="$QE_WORK_ROOT/slurm-logs"
  --output="$QE_WORK_ROOT/slurm-logs/%x-%j.out"
  --error="$QE_WORK_ROOT/slurm-logs/%x-%j.err" --export=ALL)
printf '[REQUEST] account=%s partition=%s case=%s ranks=29 threads=1 mem=64G time=12h\n' "$account" "$partition" "$QE_CASE"
printf '[WORK ROOT] %s\n' "$QE_WORK_ROOT"
sbatch --test-only "${job_args[@]}" "$PROJECT_ROOT/hpc/taiwania3_qe.sbatch"
if [[ "$action" == check ]]; then
  printf '[CHECK COMPLETE] No job submitted and no QE calculation started. Runtime compatibility, account balance and personal disk quota are not certified by this check.\n'
  exit 0
fi
job_id="$(sbatch --parsable "${job_args[@]}" "$PROJECT_ROOT/hpc/taiwania3_qe.sbatch")"
printf '[SUBMITTED] job=%s case=%s\n' "$job_id" "$QE_CASE"
printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$job_id" "$QE_CASE" "$receipt" >> "$QE_WORK_ROOT/slurm-logs/submissions.tsv"
printf '[NEXT] squeue -j %s\n' "${job_id%%;*}"
