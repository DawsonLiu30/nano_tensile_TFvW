#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/divacancy_transfer_common.sh"
export LOCAL_ROOT PYTHON REMOTE_HOST
export SERIES_NAME="${SERIES_NAME:-qe_divacancy_$(date -u +%Y%m%dT%H%M%SZ)}"
export REMOTE_ROOT="${REMOTE_ROOT:-/work/dawson666/qe_cases/qe_runs}"
export REMOTE_OUTDIR="${REMOTE_OUTDIR:-${REMOTE_ROOT}/${SERIES_NAME}}"
export LOCAL_BUILD="${LOCAL_BUILD:-${LOCAL_ROOT}/work/${SERIES_NAME}}"
bash "${LOCAL_ROOT}/scripts/upload_qe_divacancy_package_to_work_20260616.sh"
submit_divacancy_package "$REMOTE_OUTDIR" submit_qe_divacancy_pair_array.sh \
  submit_qe_divacancy_pair_array.settings "${MAX_PARALLEL:-5}"
