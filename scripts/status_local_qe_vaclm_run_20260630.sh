#!/usr/bin/env bash
set -euo pipefail

RUNROOT="$HOME/qe_vaclm_local_20260630"
echo "[ROOT] $RUNROOT"

if [ -s "$RUNROOT/BACKGROUND_PID.txt" ]; then
  pid="$(cat "$RUNROOT/BACKGROUND_PID.txt")"
  if ps -p "$pid" >/dev/null 2>&1; then
    echo "[PID] $pid RUNNING"
  else
    echo "[PID] $pid NOT_RUNNING"
  fi
else
  echo "[PID] none"
fi

echo
echo "[STATUS]"
if [ -f "$RUNROOT/LOCAL_QE_STATUS.tsv" ]; then
  cat "$RUNROOT/LOCAL_QE_STATUS.tsv"
else
  echo "no status yet"
fi

echo
for case in pristine_vcrelax vacancy_vcrelax; do
  cdir="$RUNROOT/$case"
  echo "============================================================"
  echo "[$case]"
  if [ -f "$cdir/pw.out" ]; then
    echo "pw.out size: $(du -h "$cdir/pw.out" | awk '{print $1}')"
    echo "JOB DONE count: $(grep -c 'JOB DONE' "$cdir/pw.out" || true)"
    grep -E "Program PWSCF|iteration #|total energy|Total force|Final enthalpy|JOB DONE|convergence" "$cdir/pw.out" | tail -30 || true
  else
    echo "pw.out not created yet"
  fi
  if [ -s "$cdir/pw.err" ]; then
    echo "--- pw.err tail ---"
    tail -20 "$cdir/pw.err"
  fi
done
