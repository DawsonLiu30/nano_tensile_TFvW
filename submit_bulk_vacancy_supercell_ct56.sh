#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"
SERIES_NAME="${SERIES_NAME:-bulk_fcc_vacancy_supercell_tfvw_20260430_8to14}"
SUPERCELLS="${SUPERCELLS:-8,10,12,14}"
PARTITION="${PARTITION:-ct56}"
CPUS="${CPUS:-4}"
MEM="${MEM:-120G}"
TIME_LIMIT="${TIME_LIMIT:-4-00:00:00}"
PP="${PP:-al.gga.recpot}"
KEDF="${KEDF:-TFVW}"
ECUT="${ECUT:-1000}"
SPACING="${SPACING:-}"
FMAX="${FMAX:-0.02}"
RELAX_STEPS="${RELAX_STEPS:-200}"
DELTA_THRESHOLD="${DELTA_THRESHOLD:-0.02}"
A0="${A0:-}"
CELL_BASIS="${CELL_BASIS:-conventional}"

IFS=',' read -r -a SUPERCELL_LIST <<< "${SUPERCELLS}"
if (( ${#SUPERCELL_LIST[@]} == 0 )); then
    echo "[ERROR] SUPERCELLS is empty."
    exit 1
fi

if [[ ! -d "${ROOT}" ]]; then
    echo "[ERROR] Missing ROOT: ${ROOT}"
    exit 1
fi

cd "${ROOT}"
mkdir -p logs_ctest "results/${SERIES_NAME}"

echo "============================================================"
echo "[SUBMIT] ROOT         : ${ROOT}"
echo "[SUBMIT] SERIES_NAME  : ${SERIES_NAME}"
echo "[SUBMIT] SUPERCELLS   : ${SUPERCELLS}"
echo "[SUBMIT] PART/CPU/MEM : ${PARTITION} / ${CPUS} / ${MEM}"
echo "[SUBMIT] TIME_LIMIT   : ${TIME_LIMIT}"
echo "[SUBMIT] ECUT/FMAX    : ${ECUT} / ${FMAX}"
echo "[SUBMIT] SPACING      : ${SPACING:-auto-from-ecut}"
echo "[SUBMIT] DELTA TARGET : ${DELTA_THRESHOLD} eV/vacancy"
echo "[SUBMIT] CELL_BASIS   : ${CELL_BASIS}"
echo "============================================================"

for N in "${SUPERCELL_LIST[@]}"; do
    N_CLEAN="$(echo "${N}" | xargs)"
    if [[ -z "${N_CLEAN}" ]]; then
        continue
    fi

    echo "[SUBMIT] ${N_CLEAN}x${N_CLEAN}x${N_CLEAN}"
    sbatch \
      --partition="${PARTITION}" \
      --cpus-per-task="${CPUS}" \
      --mem="${MEM}" \
      --time="${TIME_LIMIT}" \
      --export=ALL,ROOT="${ROOT}",SERIES_NAME="${SERIES_NAME}",SUPERCELL="${N_CLEAN}",PP="${PP}",KEDF="${KEDF}",ECUT="${ECUT}",SPACING="${SPACING}",FMAX="${FMAX}",RELAX_STEPS="${RELAX_STEPS}",DELTA_THRESHOLD="${DELTA_THRESHOLD}",A0="${A0}",CELL_BASIS="${CELL_BASIS}" \
      run_bulk_vacancy_supercell_one_ct56.sbatch
done
