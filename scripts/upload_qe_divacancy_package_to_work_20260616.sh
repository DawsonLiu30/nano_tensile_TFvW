#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
LOCAL_BUILD="${LOCAL_BUILD:-/mnt/c/Users/dawso/Desktop/QE_DIVACANCY_RSCAN_UPLOAD_20260616}"
REMOTE_HOST="${REMOTE_HOST:-iservice}"
REMOTE_ROOT="${REMOTE_ROOT:-/work/dawson666/qe_cases/qe_runs}"
SERIES_NAME="${SERIES_NAME:-qe_divacancy_vcrelax_conv3x3x3_rscan_20260616}"
REMOTE_OUTDIR="${REMOTE_ROOT}/${SERIES_NAME}"

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
echo "Build and upload QE divacancy package to /work"
echo "============================================================"
echo "[LOCAL ROOT ] ${LOCAL_ROOT}"
echo "[LOCAL BUILD] ${LOCAL_BUILD}"
echo "[REMOTE    ] ${REMOTE_HOST}:${REMOTE_OUTDIR}"
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
echo "[2/3] Upload package to /work"
ssh "${REMOTE_HOST}" "rm -rf '${REMOTE_OUTDIR}' && mkdir -p '${REMOTE_OUTDIR}'"
rsync -avhP "${LOCAL_BUILD}/" "${REMOTE_HOST}:${REMOTE_OUTDIR}/"

echo
echo "[3/3] Write remote run note"
ssh "${REMOTE_HOST}" "cat > '${REMOTE_OUTDIR}/RUN_ON_ISERVICE.md' <<'EOF'
# QE divacancy r-scan package

Working directory:
  /work/dawson666/qe_cases/qe_runs/qe_divacancy_vcrelax_conv3x3x3_rscan_20260616

Submit:

```bash
cd /work/dawson666/qe_cases/qe_runs/qe_divacancy_vcrelax_conv3x3x3_rscan_20260616
sbatch submit_qe_divacancy_pair_array.sh
```

Monitor:

```bash
squeue -u dawson666
sacct -j <JOBID> --format=JobID,JobName%20,Partition,State,Elapsed,Timelimit,AllocCPUS,ReqMem,ExitCode,Start,End
```

Check completion after all jobs leave queue:

```bash
cd /work/dawson666/qe_cases/qe_runs/qe_divacancy_vcrelax_conv3x3x3_rscan_20260616
for d in pair_scan/*; do
  [ -d \"\$d\" ] || continue
  echo \"===== \$d =====\"
  grep -R \"JOB DONE\" \"\$d\" --include='*.out' | wc -l
done
python scripts/collect_qe_vcrelax_vacancy.py --rootdir .
```
EOF"

echo
echo "============================================================"
echo "Upload complete. This script did NOT submit QE."
echo "============================================================"
echo "On iservice:"
echo "  cd ${REMOTE_OUTDIR}"
echo "  cat RUN_ON_ISERVICE.md"
