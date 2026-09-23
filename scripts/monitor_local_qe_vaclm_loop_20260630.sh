#!/usr/bin/env bash
set -euo pipefail

RUNROOT="$HOME/qe_vaclm_local_20260630"
LOGDIR="/mnt/c/Users/dawso/Desktop/QE_VACLM_LOCAL_MONITOR_20260630"
LOG="$LOGDIR/monitor.log"
DONE="$LOGDIR/MONITOR_DONE.txt"

mkdir -p "$LOGDIR"
echo "$(date -Iseconds) START WSL QE monitor loop" >> "$LOG"

while true; do
  {
    echo "============================================================"
    date -Iseconds
    bash /mnt/c/Users/dawso/nano_tensile_TFvW/scripts/status_local_qe_vaclm_run_20260630.sh || true
    echo "[MEMORY]"
    free -h || true
    echo "[PW_PROCESSES]"
    ps -eo pid,ppid,pcpu,pmem,etime,args | grep "pw.x -in pw.in" | grep -v grep || true
  } >> "$LOG" 2>&1

  cd "$RUNROOT"
  pristine_done="$(grep -c "JOB DONE" pristine_vcrelax/pw.out 2>/dev/null || true)"
  vacancy_done="$(grep -c "JOB DONE" vacancy_vcrelax/pw.out 2>/dev/null || true)"
  if [ "${pristine_done:-0}" -ge 1 ] && [ "${vacancy_done:-0}" -ge 1 ]; then
    echo "$(date -Iseconds) BOTH_DONE copying results" >> "$LOG"
    bash /mnt/c/Users/dawso/nano_tensile_TFvW/scripts/copy_local_qe_vaclm_results_20260630.sh >> "$LOG" 2>&1 || true
    echo "DONE $(date -Iseconds)" > "$DONE"
    exit 0
  fi

  sleep "${QE_MONITOR_INTERVAL_SECONDS:-180}"
done
