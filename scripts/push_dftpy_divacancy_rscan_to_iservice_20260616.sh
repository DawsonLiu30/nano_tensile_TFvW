#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/divacancy_transfer_common.sh"
REMOTE_ROOT="${REMOTE_ROOT:-/work/dawson666/dftpy_project/relax/dftpy45}"
SERIES_RELATIVE_DIR="${SERIES_RELATIVE_DIR:-${SERIES_NAME:-Al_defects/03_defect_cases/divacancy/dftpy_tfvw/divacancy_$(date -u +%Y%m%dT%H%M%SZ)}}"
REMOTE_OUTDIR="${REMOTE_OUTDIR:-${REMOTE_ROOT}/results/${SERIES_RELATIVE_DIR}}"
LOCAL_BUILD="${LOCAL_BUILD:-${LOCAL_ROOT}/work/$(basename -- "$SERIES_RELATIVE_DIR")}"
PP_LOCAL="${PP_LOCAL:-${LOCAL_ROOT}/al.lda.recpot}"

"$PYTHON" -B "${LOCAL_ROOT}/scripts/prepare_dftpy_divacancy_rscan_20260616.py" \
  --outdir "$LOCAL_BUILD" --pp "$PP_LOCAL" \
  --a0 "${A0:-3.9545804060131293}" --repeat "${REPEAT:-3x3x3}" \
  --xc "${XC:-LDA}" --kedf "${KEDF:-TFVW}" --kedf-x "${KEDF_X:-0.9}" --kedf-y "${KEDF_Y:-0.1}" \
  --spacing "${SPACING:-0.20}" --fmax "${FMAX:-0.005}" --relax-steps "${RELAX_STEPS:-5000}" \
  --pair-selection "${PAIR_SELECTION:-fixed_direction}" --direction="${DIRECTION:-1,1,0}" \
  --max-pairs "${MAX_PAIRS:-0}" --ase-optimizer "${ASE_OPTIMIZER:-BFGS}" \
  --account "${ACCOUNT:-MST114175}" --partition "${PARTITION:-ctest}" \
  --time-limit "${TIME_LIMIT:-02:00:00}" --cpus "${CPUS:-1}" --mem "${MEM:-96G}" \
  --max-parallel "${MAX_PARALLEL:-2}"

mkdir -p "$LOCAL_BUILD/scripts" "$LOCAL_BUILD/app"
cp "${LOCAL_ROOT}/app/dft_engine.py" "$LOCAL_BUILD/app/"
cp "${LOCAL_ROOT}/scripts/run_dftpy_vcrelax_vacancy_one.py" \
   "${LOCAL_ROOT}/scripts/collect_dftpy_conventional_vacancy.py" \
   "${LOCAL_ROOT}/scripts/prepare_dftpy_divacancy_rscan_20260616.py" \
   "${LOCAL_ROOT}/scripts/divacancy_geometry.py" \
   "${LOCAL_ROOT}/scripts/divacancy_analysis_checks.py" "$LOCAL_BUILD/scripts/"
upload_new_divacancy_package "$LOCAL_BUILD" "$REMOTE_OUTDIR"
submit_divacancy_package "$REMOTE_OUTDIR" submit_dftpy_divacancy_pair_array.sh \
  settings_pair_scan.txt "${MAX_PARALLEL:-2}"
