#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"
DIAMETERS="${DIAMETERS:-2,3,4,5,6,7,8}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
CYCLES="${CYCLES:-80}"
STEP="${STEP:-0.01}"
ECUT="${ECUT:-1000}"
SPACING="${SPACING:-}"
FMAX="${FMAX:-0.02}"
PREP_RELAX_STEPS="${PREP_RELAX_STEPS:-120}"
TENSILE_RELAX_STEPS="${TENSILE_RELAX_STEPS:-80}"
CYCLES_PER_LAUNCH="${CYCLES_PER_LAUNCH:-1}"
A0="${A0:-4.118877004246}"
PARTITION="${PARTITION:-ct56}"
CPUS="${CPUS:-4}"
MEM="${MEM:-120G}"
TIME_LIMIT="${TIME_LIMIT:-4-00:00:00}"

IFS=',' read -r -a DIAMETER_LIST <<< "${DIAMETERS}"
if (( ${#DIAMETER_LIST[@]} == 0 )); then
    echo "[ERROR] DIAMETERS is empty."
    exit 1
fi

ARRAY_END=$(( ${#DIAMETER_LIST[@]} - 1 ))

if [[ ! -d "${ROOT}" ]]; then
    echo "[ERROR] Missing ROOT: ${ROOT}"
    exit 1
fi

cd "${ROOT}"
mkdir -p logs_ctest results/hpc_status results/professor_review

export ROOT DIAMETERS CYCLES STEP ECUT SPACING FMAX PREP_RELAX_STEPS TENSILE_RELAX_STEPS CYCLES_PER_LAUNCH A0

echo "============================================================"
echo "[SUBMIT] ROOT         : ${ROOT}"
echo "[SUBMIT] DIAMETERS    : ${DIAMETERS}"
echo "[SUBMIT] ARRAY        : 0-${ARRAY_END}%${MAX_PARALLEL}"
echo "[SUBMIT] STEP/CYCLES  : ${STEP} / ${CYCLES}"
echo "[SUBMIT] ECUT/FMAX    : ${ECUT} / ${FMAX}"
echo "[SUBMIT] SPACING      : ${SPACING:-auto-from-ecut}"
echo "[SUBMIT] CHUNK CYCLES : ${CYCLES_PER_LAUNCH}"
echo "[SUBMIT] PART/CPU/MEM : ${PARTITION} / ${CPUS} / ${MEM}"
echo "[SUBMIT] TIME_LIMIT   : ${TIME_LIMIT}"
echo "============================================================"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY_RUN] sbatch --export=ALL --partition=${PARTITION} --cpus-per-task=${CPUS} --mem=${MEM} --time=${TIME_LIMIT} --array=0-${ARRAY_END}%${MAX_PARALLEL} run_finite_grip_array_ct56.sbatch"
    exit 0
fi

sbatch \
  --export=ALL \
  --partition="${PARTITION}" \
  --cpus-per-task="${CPUS}" \
  --mem="${MEM}" \
  --time="${TIME_LIMIT}" \
  --array="0-${ARRAY_END}%${MAX_PARALLEL}" \
  run_finite_grip_array_ct56.sbatch
