#!/usr/bin/env bash
set -euo pipefail

LOCAL_BASE="${LOCAL_BASE:-/mnt/c/Users/dawso/Desktop/LATEST_VACANCY_BENCHMARK_20260601}"
REPO_ROOT="${REPO_ROOT:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
QE_REMOTE="${QE_REMOTE:-iservice:/gpfs-work/dawson666/qe_cases/qe_runs/qe_vacancy_vcrelax_conv3x3x3_centered_20260528/}"
DFTPY_REMOTE_BASE="${DFTPY_REMOTE_BASE:-iservice:/gpfs-work/dawson666/dftpy_project/relax/dftpy45/results}"
DFTPY_WORK_REMOTE="${DFTPY_WORK_REMOTE:-iservice:/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"

TFVW_SERIES="dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_20260529"
SM_SERIES="dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_SM_spacing_20260529"
WT_SERIES="dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_WT_spacing_20260529"

RAW="$LOCAL_BASE/raw"
PROCESSED="$LOCAL_BASE/processed"
ORGANIZER="$LOCAL_BASE/organizer_scripts"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "[ERROR] Python is required for local collection, but neither python3 nor python was found." >&2
  exit 127
fi

echo "============================================================"
echo "Pull latest QE + DFTpy vacancy benchmark"
echo "============================================================"
echo "[LOCAL ] $LOCAL_BASE"
echo "[QE    ] $QE_REMOTE"
echo "[DFTpy ] $DFTPY_REMOTE_BASE"
echo

mkdir -p "$RAW/QE" "$RAW/DFTpy_TFVW/$TFVW_SERIES" "$RAW/DFTpy_SM/$SM_SERIES" "$RAW/DFTpy_WT/$WT_SERIES" "$RAW/DFTpy_support" "$PROCESSED" "$ORGANIZER"

echo "[STEP 1/5] Pull QE/PBE vc-relax raw data"
rsync -avhP --exclude '*/tmp/***' "$QE_REMOTE" "$RAW/QE/"

echo
echo "[STEP 2/5] Pull DFTpy/LDA + TFVW raw data"
rsync -avhP "$DFTPY_REMOTE_BASE/$TFVW_SERIES/" "$RAW/DFTpy_TFVW/$TFVW_SERIES/"

echo
echo "[STEP 3/5] Pull DFTpy/LDA + SM raw data"
rsync -avhP "$DFTPY_REMOTE_BASE/$SM_SERIES/" "$RAW/DFTpy_SM/$SM_SERIES/"

echo
echo "[STEP 4/5] Pull DFTpy/LDA + WT raw data"
rsync -avhP "$DFTPY_REMOTE_BASE/$WT_SERIES/" "$RAW/DFTpy_WT/$WT_SERIES/"

echo
echo "[STEP 5/5] Pull DFTpy pseudo and runner scripts used for reproducibility"
rsync -avhP --relative \
  "$DFTPY_WORK_REMOTE/./al.lda.recpot" \
  "$DFTPY_WORK_REMOTE/./app/dft_engine.py" \
  "$DFTPY_WORK_REMOTE/./scripts/prepare_dftpy_vacancy_conventional.py" \
  "$DFTPY_WORK_REMOTE/./scripts/run_dftpy_vcrelax_vacancy_one.py" \
  "$DFTPY_WORK_REMOTE/./submit_dftpy_vcrelax_conv3x3x3_ct56_array.sh" \
  "$RAW/DFTpy_support/"

echo
echo "============================================================"
echo "Collect local summaries"
echo "============================================================"
"$PYTHON" "$REPO_ROOT/scripts/collect_qe_vcrelax_vacancy.py" --rootdir "$RAW/QE"

for entry in \
  "DFTpy_TFVW:$TFVW_SERIES" \
  "DFTpy_SM:$SM_SERIES" \
  "DFTpy_WT:$WT_SERIES"
do
  bucket="${entry%%:*}"
  series="${entry#*:}"
  root="$RAW/$bucket/$series"
  "$PYTHON" "$REPO_ROOT/scripts/collect_dftpy_conventional_vacancy.py" --rootdir "$root"
  "$PYTHON" "$REPO_ROOT/scripts/collect_dftpy_vcrelax_fmax.py" "$root" --out "$root/dftpy_vcrelax_fmax_summary.csv"
done

"$PYTHON" "$REPO_ROOT/scripts/summarize_latest_vacancy_benchmark.py" --rootdir "$LOCAL_BASE"

cp "$REPO_ROOT/scripts/pull_latest_vacancy_benchmark_20260601.sh" "$ORGANIZER/"
cp "$REPO_ROOT/scripts/summarize_latest_vacancy_benchmark.py" "$ORGANIZER/"
cp "$REPO_ROOT/scripts/collect_qe_vcrelax_vacancy.py" "$ORGANIZER/"
cp "$REPO_ROOT/scripts/collect_dftpy_conventional_vacancy.py" "$ORGANIZER/"
cp "$REPO_ROOT/scripts/collect_dftpy_vcrelax_fmax.py" "$ORGANIZER/"

echo
echo "============================================================"
echo "Create compact review zip"
echo "============================================================"
ZIP_PATH="$LOCAL_BASE/LATEST_VACANCY_BENCHMARK_20260601.zip"
rm -f "$ZIP_PATH"
"$PYTHON" - "$LOCAL_BASE" "$ZIP_PATH" <<'PY'
from pathlib import Path
import sys
import zipfile

source = Path(sys.argv[1]).resolve()
zip_path = Path(sys.argv[2]).resolve()

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in source.rglob("*"):
        if path.is_dir() or path == zip_path:
            continue
        if "tmp" in path.relative_to(source).parts:
            continue
        zf.write(path, path.relative_to(source.parent))
PY

echo "[REPORT] $LOCAL_BASE/README_LATEST_RESULTS.md"
echo "[TABLES] $PROCESSED"
echo "[ZIP   ] $ZIP_PATH"
