#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/qe_vaclm_local_20260630"
exec env NP="${NP:-4}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
  bash ./run_local_qe_reference.sh
