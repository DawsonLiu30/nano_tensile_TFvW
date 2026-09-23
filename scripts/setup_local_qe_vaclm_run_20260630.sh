#!/usr/bin/env bash
set -euo pipefail

RUNROOT="$HOME/qe_vaclm_local_20260630"
SRC="/mnt/c/Users/dawso/Desktop/QE_VACLM_REFERENCE_20260630"

if [ ! -d "$SRC" ]; then
  echo "[ERROR] missing source QE package: $SRC" >&2
  exit 2
fi

rm -rf "$RUNROOT"
mkdir -p "$RUNROOT"
cp -a "$SRC"/. "$RUNROOT"/

cat > "$RUNROOT/run_local_qe_reference.sh" <<'EOF'
#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
MAMBA="$HOME/.local/bin/micromamba"
MAMBA_ROOT="$HOME/micromamba"
NP="${NP:-4}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OMP_PROC_BIND=false
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_rmaps_base_oversubscribe=1

STATUS="$ROOT/LOCAL_QE_STATUS.tsv"
SUMMARY="$ROOT/LOCAL_QE_SUMMARY.txt"

{
  echo "started_iso=$(date -Iseconds)"
  echo "root=$ROOT"
  echo "np=$NP"
  echo "omp=$OMP_NUM_THREADS"
  echo "pw=$("$MAMBA" run -r "$MAMBA_ROOT" -n qe command -v pw.x 2>/dev/null || true)"
} > "$SUMMARY"

printf "case\tstatus\tstart\tend\texit_code\n" > "$STATUS"

run_case() {
  local case="$1"
  local cdir="$ROOT/$case"
  local start end rc

  start="$(date -Iseconds)"
  printf "%s\tRUNNING\t%s\t\t\n" "$case" "$start" >> "$STATUS"

  cd "$cdir" || return 91
  mkdir -p tmp pseudo
  cp -f "$ROOT/pseudo/Al_PAW_PBE.UPF" pseudo/Al_PAW_PBE.UPF

  {
    echo "[INFO] case=$case"
    echo "[INFO] start=$start"
    echo "[INFO] NP=$NP OMP_NUM_THREADS=$OMP_NUM_THREADS"
    echo "[INFO] host=$(hostname)"
  } > local_qe_run.log

  "$MAMBA" run -r "$MAMBA_ROOT" -n qe mpirun -np "$NP" pw.x -in pw.in > pw.out 2> pw.err
  rc=$?
  end="$(date -Iseconds)"
  echo "[INFO] end=$end rc=$rc" >> local_qe_run.log

  if grep -q "JOB DONE" pw.out; then
    printf "%s\tCOMPLETED\t%s\t%s\t%s\n" "$case" "$start" "$end" "$rc" >> "$STATUS"
  else
    printf "%s\tNO_JOB_DONE\t%s\t%s\t%s\n" "$case" "$start" "$end" "$rc" >> "$STATUS"
  fi

  return "$rc"
}

run_case pristine_vcrelax || exit $?
run_case vacancy_vcrelax || exit $?

echo "finished_iso=$(date -Iseconds)" >> "$SUMMARY"
EOF

chmod +x "$RUNROOT/run_local_qe_reference.sh"
echo "$RUNROOT"
