#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
REMOTE_HOST="${REMOTE_HOST:-iservice}"
REMOTE_ROOT="${REMOTE_ROOT:-/work/dawson666/qe_cases/qe_runs}"
SERIES_NAME="${SERIES_NAME:-qe_divacancy_vcrelax_conv3x3x3_rscan_20260616}"
REMOTE_OUTDIR="${REMOTE_ROOT}/${SERIES_NAME}"
PSEUDO_LOCAL="${PSEUDO_LOCAL:-/mnt/c/Users/dawso/Desktop/LATEST_VACANCY_BENCHMARK_20260601/raw/QE/psp/Al_PAW_PBE.UPF}"
PSEUDO="${PSEUDO:-${REMOTE_ROOT}/psp/Al_PAW_PBE.UPF}"

ACCOUNT="${ACCOUNT:-MST114175}"
PARTITION="${PARTITION:-ct56}"
TIME_LIMIT="${TIME_LIMIT:-4-00:00:00}"
NTASKS="${NTASKS:-28}"
MEM="${MEM:-128G}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"

A0="${A0:-4.039848}"
REPEAT="${REPEAT:-3x3x3}"
ECUT="${ECUT:-800}"
KMESH="${KMESH:-3x3x3}"
FORCE_CONV="${FORCE_CONV:-0.002}"
PRESS_CONV_KBAR="${PRESS_CONV_KBAR:-0.5}"

echo "============================================================"
echo "Push QE/PBE divacancy r-scan workflow"
echo "============================================================"
echo "[LOCAL ] ${LOCAL_ROOT}"
echo "[REMOTE] ${REMOTE_HOST}:${REMOTE_OUTDIR}"
echo "[PSEUDO] remote=${PSEUDO}; local=${PSEUDO_LOCAL}"
echo "[CELL  ] conventional fcc ${REPEAT}; a0=${A0} A"
echo "[CALC  ] vc-relax, ecut=${ECUT} eV, kmesh=${KMESH}"
echo "[ARRAY ] 0-4%${MAX_PARALLEL}"

cd "${LOCAL_ROOT}"

echo
echo "[1/3] Push QE preparation/collection scripts"
ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_ROOT}/scripts' '${REMOTE_ROOT}/psp'"
rsync -avhP \
  scripts/prepare_qe_vacancy_vcrelax_3x3x3.py \
  scripts/prepare_qe_divacancy_vcrelax_rscan_20260616.py \
  scripts/collect_qe_vcrelax_vacancy.py \
  "${REMOTE_HOST}:${REMOTE_ROOT}/scripts/"
if [[ -f "${PSEUDO_LOCAL}" ]]; then
  rsync -avhP "${PSEUDO_LOCAL}" "${REMOTE_HOST}:${PSEUDO}"
else
  echo "[WARN] Local pseudo not found at ${PSEUDO_LOCAL}; assuming remote ${PSEUDO} already exists."
fi

echo
echo "[2/3] Prepare remote QE divacancy r-scan package"
ssh "${REMOTE_HOST}" "
set -euo pipefail
cd '${REMOTE_ROOT}'
python scripts/prepare_qe_divacancy_vcrelax_rscan_20260616.py \
  --outdir '${REMOTE_OUTDIR}' \
  --pseudo '${PSEUDO}' \
  --a0 '${A0}' \
  --repeat '${REPEAT}' \
  --ecut '${ECUT}' \
  --kmesh '${KMESH}' \
  --force-conv '${FORCE_CONV}' \
  --press-conv-kbar '${PRESS_CONV_KBAR}' \
  --partition '${PARTITION}' \
  --ntasks '${NTASKS}' \
  --time-limit '${TIME_LIMIT}' \
  --mem '${MEM}' \
  --max-parallel '${MAX_PARALLEL}'
echo
echo '[REMOTE] QE pair plan:'
cat '${REMOTE_OUTDIR}/qe_divacancy_pair_plan.csv'
"

echo
echo "[3/3] Submit QE array"
ssh "${REMOTE_HOST}" "
set -euo pipefail
cd '${REMOTE_OUTDIR}'
sbatch \
  -A '${ACCOUNT}' \
  -p '${PARTITION}' \
  -t '${TIME_LIMIT}' \
  --array=0-4%${MAX_PARALLEL} \
  submit_qe_divacancy_pair_array.sh
"

echo
echo "============================================================"
echo "Submitted. Monitor on iservice:"
echo "  squeue -u dawson666"
echo "  cd ${REMOTE_OUTDIR} && sacct -j <JOBID> --format=JobID,JobName%20,Partition,State,Elapsed,Timelimit,AllocCPUS,ReqMem,ExitCode,Start,End"
echo "============================================================"
