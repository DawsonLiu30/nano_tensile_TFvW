#!/usr/bin/env bash
set -euo pipefail
SERIES_NAME="${SERIES_NAME:-qe_divacancy_$(date -u +%Y%m%dT%H%M%SZ)}"
export SERIES_NAME
export REMOTE_OUTDIR="${REMOTE_STAGE:-/home/dawson666/${SERIES_NAME}}"
bash "$(dirname -- "${BASH_SOURCE[0]}")/upload_qe_divacancy_package_to_work_20260616.sh"
echo '[STAGING] Copy to your writable work filesystem before submission; inspect the copy there.'
