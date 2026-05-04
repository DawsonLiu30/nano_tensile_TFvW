#!/usr/bin/env bash
set -euo pipefail

# Run this from WSL, for example:
#   wsl bash /mnt/c/Users/dawso/nano_tensile_TFvW/manual_runs/fetch_ct56_results_from_wsl.sh
#
# Useful overrides:
#   MODE=light DIAMETERS=3,4,5,6,7,8
#   MODE=full  DIAMETERS=3,4,5
#   REMOTE=iservice
#   REMOTE_ROOT=/gpfs-work/dawson666/dftpy_project/relax/dftpy45

LOCAL_ROOT="${LOCAL_ROOT:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
REMOTE="${REMOTE:-iservice}"
REMOTE_ROOT="${REMOTE_ROOT:-/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"
DIAMETERS="${DIAMETERS:-3,4,5,6,7,8}"
MODE="${MODE:-light}"

run_rsync_optional() {
  local src="$1"
  local dst="$2"
  shift 2
  local log_file
  log_file="$(mktemp)"

  set +e
  rsync "$@" "${src}" "${dst}" 2>&1 | tee "${log_file}"
  local rc=${PIPESTATUS[0]}
  set -e

  if [[ ${rc} -eq 0 ]]; then
    rm -f "${log_file}"
    return 0
  fi

  if grep -q "No such file or directory" "${log_file}"; then
    echo "[fetch-ct56] warning: remote path missing, skipped: ${src}"
    rm -f "${log_file}"
    return 0
  fi

  echo "[fetch-ct56] rsync failed for: ${src}" >&2
  cat "${log_file}" >&2
  rm -f "${log_file}"
  return ${rc}
}

format_case() {
  local d="$1"
  if [[ "$d" == *.* ]]; then
    printf 'finite_grip_111_%snm_vacancy_tfvw' "$d"
  else
    printf 'finite_grip_111_%s.0nm_vacancy_tfvw' "$d"
  fi
}

echo "============================================================"
echo "[fetch-ct56] local root : ${LOCAL_ROOT}"
echo "[fetch-ct56] remote     : ${REMOTE}:${REMOTE_ROOT}"
echo "[fetch-ct56] diameters  : ${DIAMETERS}"
echo "[fetch-ct56] mode       : ${MODE}"
echo "============================================================"

mkdir -p "${LOCAL_ROOT}/results/professor_review" "${LOCAL_ROOT}/results/hpc_status"

IFS=',' read -r -a DLIST <<< "${DIAMETERS}"

for d in "${DLIST[@]}"; do
  d="$(echo "$d" | xargs)"
  [[ -z "${d}" ]] && continue
  case_name="$(format_case "$d")"
  mkdir -p "${LOCAL_ROOT}/cases/${case_name}" \
           "${LOCAL_ROOT}/cases/${case_name}/inputs" \
           "${LOCAL_ROOT}/cases/${case_name}/inputs_preview"

  echo "------------------------------------------------------------"
  echo "[fetch-ct56] case: ${case_name}"
  echo "------------------------------------------------------------"

  if [[ "${MODE}" == "full" ]]; then
    run_rsync_optional \
      "${REMOTE}:${REMOTE_ROOT}/cases/${case_name}/results/" \
      "${LOCAL_ROOT}/cases/${case_name}/results/" \
      -av --info=progress2 \
      --ignore-missing-args
  else
    run_rsync_optional \
      "${REMOTE}:${REMOTE_ROOT}/cases/${case_name}/results/" \
      "${LOCAL_ROOT}/cases/${case_name}/results/" \
      -av --info=progress2 \
      --ignore-missing-args \
      --exclude '*.traj' \
      --exclude '*.log' \
      --exclude '*.out' \
      --exclude '*.err' \
      --exclude '*.tmp'
  fi

  run_rsync_optional \
    "${REMOTE}:${REMOTE_ROOT}/cases/${case_name}/inputs/grip_metadata.json" \
    "${LOCAL_ROOT}/cases/${case_name}/inputs/" \
    -av --info=progress2 --ignore-missing-args
  run_rsync_optional \
    "${REMOTE}:${REMOTE_ROOT}/cases/${case_name}/inputs/grip_vacancy_manifest.json" \
    "${LOCAL_ROOT}/cases/${case_name}/inputs/" \
    -av --info=progress2 --ignore-missing-args
  run_rsync_optional \
    "${REMOTE}:${REMOTE_ROOT}/cases/${case_name}/inputs_preview/grip_metadata_preview.json" \
    "${LOCAL_ROOT}/cases/${case_name}/inputs_preview/" \
    -av --info=progress2 --ignore-missing-args
  run_rsync_optional \
    "${REMOTE}:${REMOTE_ROOT}/cases/${case_name}/inputs_preview/grip_vacancy_preview_manifest.json" \
    "${LOCAL_ROOT}/cases/${case_name}/inputs_preview/" \
    -av --info=progress2 --ignore-missing-args
done

echo "------------------------------------------------------------"
echo "[fetch-ct56] syncing shared result folders"
echo "------------------------------------------------------------"

if [[ "${MODE}" == "full" ]]; then
  run_rsync_optional \
    "${REMOTE}:${REMOTE_ROOT}/results/professor_review/" \
    "${LOCAL_ROOT}/results/professor_review/" \
    -av --info=progress2 --ignore-missing-args
  run_rsync_optional \
    "${REMOTE}:${REMOTE_ROOT}/results/hpc_status/" \
    "${LOCAL_ROOT}/results/hpc_status/" \
    -av --info=progress2 --ignore-missing-args
else
  run_rsync_optional \
    "${REMOTE}:${REMOTE_ROOT}/results/professor_review/" \
    "${LOCAL_ROOT}/results/professor_review/" \
    -av --info=progress2 \
    --ignore-missing-args \
    --exclude '*.traj' \
    --exclude '*.log' \
    --exclude '*.out' \
    --exclude '*.err'
  run_rsync_optional \
    "${REMOTE}:${REMOTE_ROOT}/results/hpc_status/" \
    "${LOCAL_ROOT}/results/hpc_status/" \
    -av --info=progress2 \
    --ignore-missing-args \
    --exclude '*.traj' \
    --exclude '*.log' \
    --exclude '*.out' \
    --exclude '*.err'
fi

cd "${LOCAL_ROOT}"

for d in "${DLIST[@]}"; do
  d="$(echo "$d" | xargs)"
  [[ -z "${d}" ]] && continue
  case_name="$(format_case "$d")"
  case_dir="${LOCAL_ROOT}/cases/${case_name}"
  if [[ ! -d "${case_dir}/results" ]]; then
    continue
  fi

  echo "------------------------------------------------------------"
  echo "[fetch-ct56] local postprocess: ${case_name}"
  echo "------------------------------------------------------------"
  python scripts/analyze_tensile_events.py --case-dir "${case_dir}" || true
done

echo "[fetch-ct56] done."
