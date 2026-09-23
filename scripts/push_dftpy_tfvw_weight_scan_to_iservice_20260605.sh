#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
REMOTE_HOST="${REMOTE_HOST:-iservice}"
REMOTE_ROOT="${REMOTE_ROOT:-/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"
SERIES_NAME="${SERIES_NAME:-dftpy_vacancy_tfvw_weight_conv3x3x3_lda_20260605}"
ACCOUNT="${ACCOUNT:-MST114175}"
PARTITION="${PARTITION:-ct56}"
TIME_LIMIT="${TIME_LIMIT:-4-00:00:00}"

Y_LIST="${Y_LIST:-0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
RELAX_STEPS="${RELAX_STEPS:-1500}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
Y_COUNT="$(python - <<PY
items = [x.strip() for x in "${Y_LIST}".split(",") if x.strip()]
print(len(items))
PY
)"
if [[ "${Y_COUNT}" -lt 1 ]]; then
  echo "[ERROR] Empty Y_LIST" >&2
  exit 1
fi
ARRAY_LAST="$((Y_COUNT - 1))"

echo "============================================================"
echo "Push DFTpy TF/vW weight-scan workflow"
echo "============================================================"
echo "[LOCAL ] ${LOCAL_ROOT}"
echo "[REMOTE] ${REMOTE_HOST}:${REMOTE_ROOT}"
echo "[SERIES] ${SERIES_NAME}"
echo "[ACCT  ] ${ACCOUNT}"
echo "[PART  ] ${PARTITION}"
echo "[TIME  ] ${TIME_LIMIT}"
echo "[YLIST ] ${Y_LIST}"
echo "[NSTEPS] ${RELAX_STEPS}"
echo "[ARRAY ] 0-${ARRAY_LAST}%${MAX_PARALLEL}"

cd "${LOCAL_ROOT}"

echo
echo "[1/3] Push updated DFTpy plumbing scripts"
rsync -avhP \
  app/dft_engine.py \
  "${REMOTE_HOST}:${REMOTE_ROOT}/app/dft_engine.py"
rsync -avhP \
  scripts/prepare_dftpy_vacancy_conventional.py \
  scripts/run_dftpy_conventional_vacancy_one.py \
  scripts/run_dftpy_vcrelax_vacancy_one.py \
  scripts/collect_dftpy_conventional_vacancy.py \
  "${REMOTE_HOST}:${REMOTE_ROOT}/scripts/"
rsync -avhP \
  submit_dftpy_vcrelax_weight_scan_ct56_array.sh \
  "${REMOTE_HOST}:${REMOTE_ROOT}/"

echo
echo "[2/3] Prepare remote conventional 3x3x3 LDA TF/vW y-scan"
ssh "${REMOTE_HOST}" "
set -euo pipefail
cd '${REMOTE_ROOT}'
python scripts/prepare_dftpy_vacancy_conventional.py \
  --outdir 'results/${SERIES_NAME}' \
  --a0 4.039848 \
  --spacing-repeat 3x3x3 \
  --spacing-list 0.20 \
  --tfvw-y-list '${Y_LIST}' \
  --pp al.lda.recpot \
  --xc LDA \
  --kedf TFVW \
  --kedf-x 1.0 \
  --fmax 0.002 \
  --relax-steps '${RELAX_STEPS}'
echo
echo '[REMOTE] settings_weight_scan.txt'
cat 'results/${SERIES_NAME}/settings_weight_scan.txt'
"

echo
echo "[3/3] Submit ${PARTITION} array"
ssh "${REMOTE_HOST}" "
set -euo pipefail
cd '${REMOTE_ROOT}'
SERIES_NAME='${SERIES_NAME}' sbatch \
  -A '${ACCOUNT}' \
  -p '${PARTITION}' \
  -t '${TIME_LIMIT}' \
  --array=0-${ARRAY_LAST}%${MAX_PARALLEL} \
  submit_dftpy_vcrelax_weight_scan_ct56_array.sh
"

echo
echo "============================================================"
echo "Submitted. Monitor on iservice with:"
echo "  squeue -u dawson666"
echo "  cd ${REMOTE_ROOT} && sacct -j <JOBID> --format=JobID,JobName%16,Partition,State,Elapsed,Timelimit,AllocCPUS,ReqMem,ExitCode,Start,End"
echo "============================================================"
