#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/work/dawson666/dftpy_project/relax/dftpy45}"
BASE="${BASE:-${ROOT}/results/Al_defects}"
FINE="${FINE:-${BASE}/01_calibration/single_vacancy/dftpy_tfvw_lambda_mu/fine_L0p90-0p95_M0p04-0p10}"
LEGACY_FINE="${ROOT}/results/dftpy_vacancy_tfvw_lambda_mu_fine_20260622"
JOB_ID="${JOB_ID:-1569500}"

echo "============================================================"
echo "Al defects status audit"
echo "============================================================"
echo "ROOT        : ${ROOT}"
echo "BASE        : ${BASE}"
echo "FINE        : ${FINE}"
echo "LEGACY_FINE : ${LEGACY_FINE}"
echo

test -d "${FINE}"
test -s "${FINE}/settings_weight_scan.txt"

echo "[PATH CHECK]"
echo "canonical=$(readlink -f "${FINE}")"
echo "legacy=$(readlink -f "${LEGACY_FINE}")"
if [[ "$(readlink -f "${FINE}")" != "$(readlink -f "${LEGACY_FINE}")" ]]; then
  echo "[ERROR] Canonical and legacy paths do not resolve to the same directory" >&2
  exit 2
fi

settings=$(grep -cve '^[[:space:]]*$' "${FINE}/settings_weight_scan.txt")
cases=$(find "${FINE}/weight_scan" -mindepth 1 -maxdepth 1 -type d | wc -l)
results=$(find "${FINE}/weight_scan" -mindepth 2 -maxdepth 2 -type f -name result.json -size +0c | wc -l)
failed=$(find "${FINE}/weight_scan" -mindepth 2 -maxdepth 2 -type f -name WORKER_FAILED.txt | wc -l)

echo
echo "[COUNTS]"
echo "settings=${settings}"
echo "case_dirs=${cases}"
echo "result_json=${results}"
echo "worker_failed=${failed}"

echo
echo "[RECENT RESULTS]"
find "${FINE}/weight_scan" -mindepth 2 -maxdepth 2 -type f -name result.json \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' | sort | tail -10 || true

echo
echo "[WORKER PROGRESS]"
grep -hE '^\[(RUN|SKIP|FAILED)\]' "${ROOT}"/logs_ctest/dftpyLMfine_${JOB_ID}_*.out 2>/dev/null | tail -30 || true

echo
echo "[WORKER ERRORS]"
for path in "${ROOT}"/logs_ctest/dftpyLMfine_${JOB_ID}_*.err; do
  [[ -e "${path}" ]] || continue
  if [[ -s "${path}" ]]; then
    echo "--- ${path}"
    tail -30 "${path}"
  fi
done

echo
echo "[SLURM]"
squeue -j "${JOB_ID}" || true
sacct -j "${JOB_ID}" --format=JobID,JobName%20,Partition,State,Elapsed,Timelimit,ExitCode || true

if [[ "${settings}" -ne 42 || "${cases}" -ne 42 ]]; then
  echo "[ERROR] Prepared fine-scan structure is incomplete" >&2
  exit 3
fi
if [[ "${failed}" -gt 0 ]]; then
  echo "[WARN] One or more worker failures need inspection"
fi
if [[ "${results}" -eq 42 ]]; then
  echo "[PASS] Fine scan is complete"
else
  echo "[INFO] Fine scan is incomplete: ${results}/42 results"
fi
