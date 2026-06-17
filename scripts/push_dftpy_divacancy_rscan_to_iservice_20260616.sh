#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
REMOTE_HOST="${REMOTE_HOST:-iservice}"
REMOTE_ROOT="${REMOTE_ROOT:-/work/dawson666/dftpy_project/relax/dftpy45}"
SERIES_NAME="${SERIES_NAME:-dftpy_divacancy_vcrelax_conv3x3x3_rscan_20260616}"
REMOTE_OUTDIR="${REMOTE_ROOT}/results/${SERIES_NAME}"

ACCOUNT="${ACCOUNT:-MST114175}"
PARTITION="${PARTITION:-ctest}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
CPUS="${CPUS:-8}"
MEM="${MEM:-96G}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

A0="${A0:-4.039848}"
REPEAT="${REPEAT:-3x3x3}"
PP="${PP:-al.lda.recpot}"
PP_LOCAL="${PP_LOCAL:-${LOCAL_ROOT}/al.lda.recpot}"
XC="${XC:-LDA}"
KEDF="${KEDF:-TFVW}"
KEDF_X="${KEDF_X:-1.0}"
KEDF_Y="${KEDF_Y:-0.13}"
SPACING="${SPACING:-0.20}"
FMAX="${FMAX:-0.002}"
RELAX_STEPS="${RELAX_STEPS:-5000}"
ASE_OPTIMIZER="${ASE_OPTIMIZER:-BFGS}"

echo "============================================================"
echo "Push DFTpy divacancy r-scan workflow"
echo "============================================================"
echo "[LOCAL ] ${LOCAL_ROOT}"
echo "[REMOTE] ${REMOTE_HOST}:${REMOTE_OUTDIR}"
echo "[CELL  ] conventional fcc ${REPEAT}; a0=${A0} A"
echo "[CALC  ] ${XC}, ${KEDF}, lambda=${KEDF_X}, mu=${KEDF_Y}, spacing=${SPACING} A"
echo "[PP    ] remote=${PP}; local=${PP_LOCAL}"
echo "[PART  ] ${PARTITION}, ${TIME_LIMIT}, array 0-4%${MAX_PARALLEL}"
echo "[FMAX  ] ${FMAX} eV/A, relax_steps=${RELAX_STEPS}"
echo "[OPT   ] ${ASE_OPTIMIZER}"

cd "${LOCAL_ROOT}"

echo
echo "[1/3] Push DFTpy scripts and engine"
ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_ROOT}/scripts' '${REMOTE_ROOT}/app'"
rsync -avhP \
  app/dft_engine.py \
  "${REMOTE_HOST}:${REMOTE_ROOT}/app/dft_engine.py"
rsync -avhP \
  scripts/prepare_dftpy_divacancy_rscan_20260616.py \
  scripts/run_dftpy_vcrelax_vacancy_one.py \
  scripts/collect_dftpy_conventional_vacancy.py \
  "${REMOTE_HOST}:${REMOTE_ROOT}/scripts/"
if [[ -f "${PP_LOCAL}" ]]; then
  rsync -avhP "${PP_LOCAL}" "${REMOTE_HOST}:${REMOTE_ROOT}/$(basename "${PP}")"
else
  echo "[WARN] Local pseudo not found at ${PP_LOCAL}; assuming remote ${PP} already exists."
fi

echo
echo "[2/3] Prepare remote DFTpy divacancy r-scan package"
ssh "${REMOTE_HOST}" "
set -euo pipefail
cd '${REMOTE_ROOT}'
python scripts/prepare_dftpy_divacancy_rscan_20260616.py \
  --outdir '${REMOTE_OUTDIR}' \
  --a0 '${A0}' \
  --repeat '${REPEAT}' \
  --pp '${PP}' \
  --xc '${XC}' \
  --kedf '${KEDF}' \
  --kedf-x '${KEDF_X}' \
  --kedf-y '${KEDF_Y}' \
  --spacing '${SPACING}' \
  --fmax '${FMAX}' \
  --relax-steps '${RELAX_STEPS}' \
  --ase-optimizer '${ASE_OPTIMIZER}' \
  --account '${ACCOUNT}' \
  --partition '${PARTITION}' \
  --time-limit '${TIME_LIMIT}' \
  --cpus '${CPUS}' \
  --mem '${MEM}' \
  --max-parallel '${MAX_PARALLEL}'
echo
echo '[REMOTE] DFTpy pair plan:'
cat '${REMOTE_OUTDIR}/divacancy_pair_plan.csv'
"

echo
echo "[3/3] Submit DFTpy array"
ssh "${REMOTE_HOST}" "
set -euo pipefail
cd '${REMOTE_ROOT}'
mkdir -p logs_ctest
SERIES_NAME='${SERIES_NAME}' sbatch \
  -A '${ACCOUNT}' \
  -p '${PARTITION}' \
  -t '${TIME_LIMIT}' \
  --array=0-4%${MAX_PARALLEL} \
  'results/${SERIES_NAME}/submit_dftpy_divacancy_pair_array.sh'
"

echo
echo "============================================================"
echo "Submitted. Monitor on iservice:"
echo "  squeue -u dawson666"
echo "  cd ${REMOTE_ROOT} && sacct -j <JOBID> --format=JobID,JobName%16,Partition,State,Elapsed,Timelimit,AllocCPUS,ReqMem,ExitCode,Start,End"
echo "============================================================"
