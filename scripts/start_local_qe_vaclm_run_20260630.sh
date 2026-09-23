#!/usr/bin/env bash
set -euo pipefail

RUNROOT="$HOME/qe_vaclm_local_20260630"
if [ ! -x "$RUNROOT/run_local_qe_reference.sh" ]; then
  echo "[ERROR] missing run script: $RUNROOT/run_local_qe_reference.sh" >&2
  exit 2
fi

cd "$RUNROOT"
if [ -s BACKGROUND_PID.txt ]; then
  old_pid="$(cat BACKGROUND_PID.txt)"
  if ps -p "$old_pid" >/dev/null 2>&1; then
    echo "[INFO] already running pid=$old_pid"
    exit 0
  fi
fi

nohup env NP="${NP:-4}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" bash ./run_local_qe_reference.sh \
  > LOCAL_QE_DRIVER.out 2> LOCAL_QE_DRIVER.err &
pid="$!"
echo "$pid" > BACKGROUND_PID.txt
echo "[STARTED] pid=$pid root=$RUNROOT"
