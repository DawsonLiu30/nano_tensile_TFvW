#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
REMOTE_HOST="${REMOTE_HOST:-iservice}"
REMOTE_REPO="${REMOTE_REPO:-/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"
OUTDIR="${OUTDIR:-/gpfs-work/dawson666/profess_runs/profess_periodic_vacancy_relax_20260605}"
PROFESS_BIN="${PROFESS_BIN:-/gpfs-work/dawson666/profess3_build_20260605/bin/PROFESS}"
ACCOUNT="${ACCOUNT:-MST114175}"
PARTITION="${PARTITION:-ctest}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

DIAMETERS="${DIAMETERS:-1.0,1.5,2.0}"
SHAPES="${SHAPES:-circle,hexagon}"
POSITIONS="${POSITIONS:-inner,middle,outer}"
KEDFS="${KEDFS:-TFPLUS_DEFAULT,CAT}"
ORIENTATION="${ORIENTATION:-111}"
A0="${A0:-4.039848}"
MIN_LZ="${MIN_LZ:-10.0}"
VACUUM="${VACUUM:-10.0}"
ECUT="${ECUT:-1600}"
ION_METHOD="${ION_METHOD:-cg2}"
ION_TOLF_EV_A="${ION_TOLF_EV_A:-0.002}"

cat <<EOF
============================================================
Push PROFESS periodic vacancy relax workflow
============================================================
[LOCAL  ] ${LOCAL_ROOT}
[REMOTE ] ${REMOTE_HOST}:${REMOTE_REPO}
[OUTDIR ] ${OUTDIR}
[PROFESS] ${PROFESS_BIN}
[PART   ] ${PARTITION}
[TIME   ] ${TIME_LIMIT}
[ARRAY  ] %${MAX_PARALLEL}
[SHAPES ] ${SHAPES}
[DIAM   ] ${DIAMETERS}
[POS    ] ${POSITIONS}
[KEDFS  ] ${KEDFS}
EOF

cd "${LOCAL_ROOT}"

echo
echo "[1/4] Push scripts"
rsync -avhP \
  scripts/prepare_profess_periodic_vacancy_relax.py \
  scripts/collect_profess_periodic_vacancy_relax.py \
  "${REMOTE_HOST}:${REMOTE_REPO}/scripts/"

echo
echo "[2/4] Prepare remote PROFESS cases"
ssh "${REMOTE_HOST}" "
set -euo pipefail
cd '${REMOTE_REPO}'
python scripts/prepare_profess_periodic_vacancy_relax.py \
  --outdir '${OUTDIR}' \
  --a0 '${A0}' \
  --diameters '${DIAMETERS}' \
  --shapes '${SHAPES}' \
  --orientation '${ORIENTATION}' \
  --positions '${POSITIONS}' \
  --kedfs '${KEDFS}' \
  --vacuum '${VACUUM}' \
  --min-lz '${MIN_LZ}' \
  --ecut '${ECUT}' \
  --ion-method '${ION_METHOD}' \
  --ion-tolf-ev-a '${ION_TOLF_EV_A}'
"

echo
echo "[3/4] Submit array"
JOB_SUBMIT_OUTPUT="$(ssh "${REMOTE_HOST}" "
set -euo pipefail
cd '${OUTDIR}'
N=\$(wc -l < settings.tsv)
LAST=\$((N - 1))
echo \"[REMOTE] settings count=\${N}\"
PROFESS_BIN='${PROFESS_BIN}' sbatch \
  -A '${ACCOUNT}' \
  -p '${PARTITION}' \
  -t '${TIME_LIMIT}' \
  --array=0-\${LAST}%${MAX_PARALLEL} \
  submit_profess_relax_array.sh
")"
echo "${JOB_SUBMIT_OUTPUT}"

echo
echo "[4/4] Save submission note"
ssh "${REMOTE_HOST}" "cat > '${OUTDIR}/SUBMISSION_NOTE.txt' <<EOF
Submitted by push_profess_periodic_vacancy_relax_to_iservice_20260605.sh
PROFESS_BIN=${PROFESS_BIN}
ACCOUNT=${ACCOUNT}
PARTITION=${PARTITION}
TIME_LIMIT=${TIME_LIMIT}
MAX_PARALLEL=${MAX_PARALLEL}
DIAMETERS=${DIAMETERS}
SHAPES=${SHAPES}
POSITIONS=${POSITIONS}
KEDFS=${KEDFS}
${JOB_SUBMIT_OUTPUT}
EOF"

cat <<EOF

============================================================
Submitted PROFESS relax workflow
============================================================
Monitor:
  squeue -u dawson666

Collect on remote after completion:
  cd ${REMOTE_REPO} && python scripts/collect_profess_periodic_vacancy_relax.py --rootdir ${OUTDIR}

Pull to local:
  OUTDIR='${OUTDIR}' bash scripts/pull_profess_periodic_vacancy_relax_results_20260605.sh
============================================================
EOF
