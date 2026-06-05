#!/usr/bin/env bash
set -euo pipefail

LOCAL_REPO="${LOCAL_REPO:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
REMOTE_HOST="${REMOTE_HOST:-iservice}"
REMOTE_REPO="${REMOTE_REPO:-/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"
OUTDIR="${OUTDIR:-/gpfs-work/dawson666/profess_runs/profess_periodic_vacancy_relax_20260605}"
LOCAL_BASE="${LOCAL_BASE:-/mnt/c/Users/dawso/Desktop/PROFESS_PERIODIC_VACANCY_RELAX_20260605}"

SERIES_NAME="$(basename "${OUTDIR}")"
LOCAL_SERIES="${LOCAL_BASE}/${SERIES_NAME}"

cat <<EOF
============================================================
Pull PROFESS periodic vacancy relax results
============================================================
[REMOTE] ${REMOTE_HOST}:${OUTDIR}
[LOCAL ] ${LOCAL_SERIES}
EOF

echo
echo "[CHECK] Remote directory exists"
if ! ssh "${REMOTE_HOST}" "test -d '${OUTDIR}'"; then
  cat <<EOF

[ERROR] Remote OUTDIR does not exist:
  ${OUTDIR}

This usually means the PROFESS job was prepared under a different OUTDIR, or the
push step was not run for this exact default series.

Find candidate PROFESS run folders on iservice with:
  ssh ${REMOTE_HOST} "find /gpfs-work/dawson666 -maxdepth 4 -type d -iname '*profess*vacancy*' 2>/dev/null | sort"

Then rerun pull with the correct path, for example:
  OUTDIR='/gpfs-work/dawson666/profess_runs/<actual_series>' \\
  bash scripts/pull_profess_periodic_vacancy_relax_results_20260605.sh

EOF
  exit 2
fi

mkdir -p "${LOCAL_SERIES}"
rsync -avhP "${REMOTE_HOST}:${OUTDIR}/" "${LOCAL_SERIES}/"

echo
echo "[LOCAL] Collect summaries"
cd "${LOCAL_REPO}"
python scripts/collect_profess_periodic_vacancy_relax.py --rootdir "${LOCAL_SERIES}"

echo
echo "[SUMMARY]"
cat "${LOCAL_SERIES}/profess_periodic_vacancy_relax_completion.csv" 2>/dev/null || true

echo
echo "============================================================"
echo "Done:"
echo "  ${LOCAL_SERIES}/profess_periodic_vacancy_relax_summary.csv"
echo "  ${LOCAL_SERIES}/profess_periodic_vacancy_relax_completion.csv"
echo "  ${LOCAL_SERIES}/profess_periodic_vacancy_relax_Ef.png"
echo "============================================================"
