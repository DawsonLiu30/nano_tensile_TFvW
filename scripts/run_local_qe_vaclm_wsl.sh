#!/usr/bin/env bash
set -euo pipefail

runtime="$(realpath -m "${AL_DEFECTS_RUNTIME:-/var/tmp/al-defects-runtime-20260907}")"
env_prefix="${AL_DEFECTS_ENV_PREFIX:-$runtime/env}"
data_root="${AL_DEFECTS_DATA_ROOT:-/mnt/c/OFDFT/AL_DEFECTS_USB_HANDOFF_20260717}"
source_root="$data_root/03_ACTIVE_QE_VCRELAX_REFERENCE"
runroot="$(realpath -m "${QE_RUNROOT:-$runtime/runs/qe_vaclm_reference}")"
case "$runroot" in "$runtime"/runs/?*) ;; *) echo '[ERROR] Unsafe QE_RUNROOT' >&2; exit 2;; esac
pwx="$env_prefix/bin/pw.x"
mpirun="$env_prefix/bin/mpirun"
np=4
selected_case='both'
run_mode='prepare'

usage() {
  cat <<'EOF'
Usage:
  run_local_qe_vaclm_wsl.sh --prepare-only
  run_local_qe_vaclm_wsl.sh --run [--case pristine|vacancy|both] [--np N]

Environment overrides:
  AL_DEFECTS_RUNTIME
  AL_DEFECTS_ENV_PREFIX
  AL_DEFECTS_DATA_ROOT
  QE_RUNROOT
EOF
}

while (($#)); do
  case "$1" in
    --prepare-only)
      run_mode='prepare'
      shift
      ;;
    --run)
      run_mode='run'
      shift
      ;;
    --case)
      selected_case="${2:?--case requires a value}"
      shift 2
      ;;
    --np)
      np="${2:?--np requires a value}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$selected_case" in
  pristine|vacancy|both) ;;
  *)
    echo "[ERROR] --case must be pristine, vacancy, or both" >&2
    exit 3
    ;;
esac

if ! [[ "$np" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] --np must be a positive integer" >&2
  exit 4
fi

for required in \
  "$pwx" \
  "$mpirun" \
  "$source_root/pristine_vcrelax/pw.in" \
  "$source_root/vacancy_vcrelax/pw.in" \
  "$source_root/pseudo/Al_PAW_PBE.UPF"; do
  if [[ ! -e "$required" ]]; then
    echo "[ERROR] Missing required path: $required" >&2
    exit 5
  fi
done

mkdir -p "$runroot"
exec 9>"$runtime/.source.lock"
flock -s -n 9 || { echo '[ERROR] Source sync is active' >&2; exit 3; }
exec 8>"$runroot/.runner.lock"
flock -n 8 || { echo '[ERROR] QE run is already active' >&2; exit 3; }
for case_name in pristine_vcrelax vacancy_vcrelax; do
  case_dir="$runroot/$case_name"
  mkdir -p "$case_dir/tmp" "$case_dir/pseudo"
  if [[ ! -e "$case_dir/pw.in" ]]; then
    cp "$source_root/$case_name/pw.in" "$case_dir/pw.in"
  fi
  cp "$source_root/pseudo/Al_PAW_PBE.UPF" \
    "$case_dir/pseudo/Al_PAW_PBE.UPF"
done

cat >"$runroot/LOCAL_RUNTIME.txt" <<EOF
prepared_at=$(date --iso-8601=seconds)
source_root=$source_root
runroot=$runroot
environment=$env_prefix
pw=$pwx
np=$np
EOF

if [[ "$run_mode" == 'prepare' ]]; then
  echo "[PREPARED] $runroot"
  echo "Review both pw.in files, then rerun with --run."
  exit 0
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OMP_PROC_BIND=false
export OMPI_MCA_rmaps_base_oversubscribe=1

printf 'case\tstatus\tstarted\tfinished\texit_code\n' \
  >>"$runroot/LOCAL_QE_STATUS.tsv"

case_completed() {
  local output="$1"
  [[ -s "$output" ]] && grep -q 'JOB DONE' "$output" && \
    grep -q 'convergence has been achieved' "$output" && \
    grep -qi 'bfgs converged' "$output" && \
    ! grep -qi 'convergence NOT achieved\|Error in routine\|Maximum CPU time exceeded' "$output"
}

run_case() {
  local case_name="$1"
  local case_dir="$runroot/$case_name"
  local started finished rc

  if case_completed "$case_dir/pw.out"; then
    echo "[SKIP] $case_name: electronic and BFGS completion markers present (numerical convergence still needs study)"
    return 0
  fi

  if [[ -s "$case_dir/pw.out" || -s "$case_dir/pw.err" ]]; then
    local archive="$runroot/audit/${case_name}_$(date -u +%Y%m%dT%H%M%S.%NZ)_$$"
    mkdir -p "$runroot/audit"
    cp -a "$case_dir" "$archive"
    echo "[ARCHIVED] $archive"
  fi

  started="$(date --iso-8601=seconds)"
  printf '%s\tRUNNING\t%s\t\t\n' "$case_name" "$started" \
    >>"$runroot/LOCAL_QE_STATUS.tsv"

  set +e
  (
    cd "$case_dir"
    "$mpirun" -np "$np" "$pwx" -in pw.in >pw.out 2>pw.err
  )
  rc=$?
  set -e
  finished="$(date --iso-8601=seconds)"

  if [[ "$rc" -eq 0 ]] && case_completed "$case_dir/pw.out"; then
    printf '%s\tCOMPLETED\t%s\t%s\t%s\n' \
      "$case_name" "$started" "$finished" "$rc" \
      >>"$runroot/LOCAL_QE_STATUS.tsv"
  else
    printf '%s\tFAILED\t%s\t%s\t%s\n' \
      "$case_name" "$started" "$finished" "$rc" \
      >>"$runroot/LOCAL_QE_STATUS.tsv"
    if [[ "$rc" -eq 0 ]]; then rc=1; fi
    return "$rc"
  fi
}

case "$selected_case" in
  pristine)
    run_case pristine_vcrelax
    ;;
  vacancy)
    run_case vacancy_vcrelax
    ;;
  both)
    run_case pristine_vcrelax
    run_case vacancy_vcrelax
    ;;
esac

echo "[COMPLETED] $runroot"
