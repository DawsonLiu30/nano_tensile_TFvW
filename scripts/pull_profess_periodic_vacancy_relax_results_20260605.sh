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
