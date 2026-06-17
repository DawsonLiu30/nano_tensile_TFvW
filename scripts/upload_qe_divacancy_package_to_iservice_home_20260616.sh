#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
LOCAL_BUILD="${LOCAL_BUILD:-/mnt/c/Users/dawso/Desktop/QE_DIVACANCY_RSCAN_UPLOAD_20260616}"
REMOTE_HOST="${REMOTE_HOST:-iservice}"
REMOTE_STAGE="${REMOTE_STAGE:-/home/dawson666/qe_divacancy_vcrelax_conv3x3x3_rscan_20260616}"

PSEUDO_LOCAL="${PSEUDO_LOCAL:-/mnt/c/Users/dawso/Desktop/LATEST_VACANCY_BENCHMARK_20260601/raw/QE/psp/Al_PAW_PBE.UPF}"
A0="${A0:-4.039848}"
REPEAT="${REPEAT:-3x3x3}"
ECUT="${ECUT:-800}"
KMESH="${KMESH:-3x3x3}"
FORCE_CONV="${FORCE_CONV:-0.002}"
PRESS_CONV_KBAR="${PRESS_CONV_KBAR:-0.5}"
PARTITION="${PARTITION:-ct56}"
NTASKS="${NTASKS:-28}"
TIME_LIMIT="${TIME_LIMIT:-4-00:00:00}"
MEM="${MEM:-128G}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"

echo "============================================================"
echo "Build and upload QE divacancy package to iservice HOME"
echo "============================================================"
echo "[LOCAL ROOT ] ${LOCAL_ROOT}"
echo "[LOCAL BUILD] ${LOCAL_BUILD}"
echo "[REMOTE    ] ${REMOTE_HOST}:${REMOTE_STAGE}"
echo "[PSEUDO    ] ${PSEUDO_LOCAL}"

cd "${LOCAL_ROOT}"

if [[ ! -f "${PSEUDO_LOCAL}" ]]; then
  echo "[ERROR] Missing local pseudo: ${PSEUDO_LOCAL}" >&2
  exit 1
fi

echo
echo "[1/3] Build local QE package"
python scripts/prepare_qe_divacancy_vcrelax_rscan_20260616.py \
  --outdir "${LOCAL_BUILD}" \
  --pseudo "${PSEUDO_LOCAL}" \
  --a0 "${A0}" \
  --repeat "${REPEAT}" \
  --ecut "${ECUT}" \
  --kmesh "${KMESH}" \
  --force-conv "${FORCE_CONV}" \
  --press-conv-kbar "${PRESS_CONV_KBAR}" \
  --partition "${PARTITION}" \
  --ntasks "${NTASKS}" \
  --time-limit "${TIME_LIMIT}" \
  --mem "${MEM}" \
  --max-parallel "${MAX_PARALLEL}"

echo
echo "[2/3] Upload to iservice HOME staging"
ssh "${REMOTE_HOST}" "rm -rf '${REMOTE_STAGE}' && mkdir -p '${REMOTE_STAGE}'"
rsync -avhP "${LOCAL_BUILD}/" "${REMOTE_HOST}:${REMOTE_STAGE}/"

echo
echo "[3/3] Write remote run note"
ssh "${REMOTE_HOST}" "cat > '${REMOTE_STAGE}/RUN_ON_ISERVICE.md' <<'EOF'
# QE divacancy r-scan package

This package was uploaded to HOME as a staging copy.

If the work filesystem becomes visible after interactive login, copy this folder there before submitting, for example:

```bash
mkdir -p /work/dawson666/qe_cases/qe_runs
cp -a ~/qe_divacancy_vcrelax_conv3x3x3_rscan_20260616 /work/dawson666/qe_cases/qe_runs/
cd /work/dawson666/qe_cases/qe_runs/qe_divacancy_vcrelax_conv3x3x3_rscan_20260616
sbatch submit_qe_divacancy_pair_array.sh
```

If /work/dawson666 is not visible, do not submit yet. First find the correct writable work directory:

```bash
pwd
df -h .
find /work /gpfs-work /lustre /project /scratch -maxdepth 3 -type d -name dawson666 2>/dev/null
```

The package contains:
- pair_scan/pair_*/pristine_vcrelax/vc-relax.in
- pair_scan/pair_*/vacancy_vcrelax/vc-relax.in
- psp/Al_PAW_PBE.UPF
- submit_qe_divacancy_pair_array.sh
- qe_divacancy_pair_plan.csv
EOF"

echo
echo "============================================================"
echo "Upload complete"
echo "============================================================"
echo "On iservice:"
echo "  cd ${REMOTE_STAGE}"
echo "  cat RUN_ON_ISERVICE.md"
