#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/divacancy_transfer_common.sh"

REMOTE_ROOT="${REMOTE_ROOT:-/work/dawson666/qe_cases/qe_runs}"
SERIES_NAME="${SERIES_NAME:-qe_divacancy_$(date -u +%Y%m%dT%H%M%SZ)}"
REMOTE_OUTDIR="${REMOTE_OUTDIR:-${REMOTE_ROOT}/${SERIES_NAME}}"
LOCAL_BUILD="${LOCAL_BUILD:-${LOCAL_ROOT}/work/${SERIES_NAME}}"
PSEUDO_LOCAL="${PSEUDO_LOCAL:-${DATA_ROOT}/03_ACTIVE_QE_VCRELAX_REFERENCE/pseudo/Al_PAW_PBE.UPF}"

"$PYTHON" -B "${LOCAL_ROOT}/scripts/prepare_qe_divacancy_vcrelax_rscan_20260616.py" \
  --outdir "$LOCAL_BUILD" --pseudo "$PSEUDO_LOCAL" \
  --a0 "${A0:-4.039848}" --repeat "${REPEAT:-3x3x3}" \
  --ecut "${ECUT:-800}" --kmesh "${KMESH:-3x3x3}" \
  --force-conv "${FORCE_CONV:-0.002}" --press-conv-kbar "${PRESS_CONV_KBAR:-0.5}" \
  --pair-selection "${PAIR_SELECTION:-fixed_direction}" --direction="${DIRECTION:-1,1,0}" \
  --max-pairs "${MAX_PAIRS:-0}" --partition "${PARTITION:-ct56}" \
  --ntasks "${NTASKS:-28}" --time-limit "${TIME_LIMIT:-4-00:00:00}" \
  --mem "${MEM:-128G}" --max-parallel "${MAX_PARALLEL:-5}"

mkdir -p "$LOCAL_BUILD/scripts"
cp "${LOCAL_ROOT}/scripts/collect_qe_vcrelax_vacancy.py" \
   "${LOCAL_ROOT}/scripts/divacancy_analysis_checks.py" \
   "${LOCAL_ROOT}/scripts/divacancy_geometry.py" "$LOCAL_BUILD/scripts/"
cat > "$LOCAL_BUILD/RUN_ON_ISERVICE.md" <<EOF
QE divacancy package

Remote working directory: $REMOTE_OUTDIR
Selection: ${PAIR_SELECTION:-fixed_direction}; direction: ${DIRECTION:-1,1,0}
Method: QE/PBE, a0=${A0:-4.039848} A, ecut=${ECUT:-800} eV, kmesh=${KMESH:-3x3x3}.

This script uploads inputs only. To submit from the working directory:
  mkdir -p logs_submit
  sbatch submit_qe_divacancy_pair_array.sh

The generated array matches submit_qe_divacancy_pair_array.settings.
Check BOTH SCF and ionic/cell convergence before treating JOB DONE as a usable result.
Collect with:
  python scripts/collect_qe_vcrelax_vacancy.py --rootdir .

The preparation_sources directory records the generator and shared geometry implementation.
Raw calculations and existing remote directories are never overwritten by this upload command.
EOF

upload_new_divacancy_package "$LOCAL_BUILD" "$REMOTE_OUTDIR"
echo "[UPLOADED; not submitted] ${REMOTE_HOST}:${REMOTE_OUTDIR}"
