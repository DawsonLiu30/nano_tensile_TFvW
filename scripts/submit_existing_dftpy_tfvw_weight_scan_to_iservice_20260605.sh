#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-iservice}"
REMOTE_ROOT="${REMOTE_ROOT:-/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"
SERIES_NAME="${SERIES_NAME:-dftpy_vacancy_tfvw_weight_fine_y0115_020_conv3x3x3_lda_20260605_v2}"
ACCOUNT="${ACCOUNT:-MST114175}"
PARTITION="${PARTITION:-ctest}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
DRY_RUN="${DRY_RUN:-0}"

cat <<EOF
============================================================
Submit existing DFTpy TF/vW weight scan
============================================================
[REMOTE] ${REMOTE_HOST}:${REMOTE_ROOT}
[SERIES] ${SERIES_NAME}
[ACCT  ] ${ACCOUNT}
[PART  ] ${PARTITION}
[TIME  ] ${TIME_LIMIT}
[ARRAY ] %${MAX_PARALLEL}
EOF

REMOTE_CMD=$(cat <<'EOF'
set -euo pipefail
cd "${REMOTE_ROOT}"
SETTING_FILE="results/${SERIES_NAME}/settings_weight_scan.txt"
if [[ ! -s "${SETTING_FILE}" ]]; then
  echo "[ERROR] Missing or empty settings file: ${SETTING_FILE}" >&2
  exit 2
fi
N=$(wc -l < "${SETTING_FILE}")
LAST=$((N - 1))
echo "[REMOTE] settings=${N}"
echo "[REMOTE] settings_weight_scan.txt:"
cat "${SETTING_FILE}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY_RUN] sbatch -A ${ACCOUNT} -p ${PARTITION} -t ${TIME_LIMIT} --array=0-${LAST}%${MAX_PARALLEL} submit_dftpy_vcrelax_weight_scan_ct56_array.sh"
  exit 0
fi
SERIES_NAME="${SERIES_NAME}" sbatch \
  -A "${ACCOUNT}" \
  -p "${PARTITION}" \
  -t "${TIME_LIMIT}" \
  --array=0-${LAST}%${MAX_PARALLEL} \
  submit_dftpy_vcrelax_weight_scan_ct56_array.sh
EOF
)

ssh "${REMOTE_HOST}" \
  "REMOTE_ROOT='${REMOTE_ROOT}' SERIES_NAME='${SERIES_NAME}' ACCOUNT='${ACCOUNT}' PARTITION='${PARTITION}' TIME_LIMIT='${TIME_LIMIT}' MAX_PARALLEL='${MAX_PARALLEL}' DRY_RUN='${DRY_RUN}' bash -s" \
  <<< "${REMOTE_CMD}"

cat <<EOF

============================================================
Submitted existing prepared series.
============================================================
Monitor:
  squeue -u dawson666

Pull after completion:
  SERIES_NAME=${SERIES_NAME} bash scripts/pull_dftpy_tfvw_weight_scan_results_20260605.sh
============================================================
EOF

