#!/usr/bin/env bash
set -euo pipefail

LOCAL_BASE="${LOCAL_BASE:-/mnt/c/Users/dawso/Desktop/vacancy_vcrelax_3x3x3_pull_20260528}"
REPO_ROOT="${REPO_ROOT:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
QE_REMOTE="${QE_REMOTE:-iservice:/gpfs-work/dawson666/qe_cases/qe_runs/qe_vacancy_vcrelax_conv3x3x3_20260528/}"
DFTPY_REMOTE="${DFTPY_REMOTE:-iservice:/gpfs-work/dawson666/dftpy_project/relax/dftpy45/results/dftpy_vacancy_vcrelax_conv3x3x3_qe_a0_20260528/}"

QE_LOCAL="$LOCAL_BASE/qe_vacancy_vcrelax_conv3x3x3_20260528"
DFTPY_LOCAL="$LOCAL_BASE/dftpy_vacancy_vcrelax_conv3x3x3_qe_a0_20260528"

echo "============================================================"
echo "Pull vc-relax 3x3x3 vacancy results"
echo "============================================================"
echo "[LOCAL] $LOCAL_BASE"
echo "[QE   ] $QE_REMOTE"
echo "[DFTpy] $DFTPY_REMOTE"
echo

mkdir -p "$QE_LOCAL" "$DFTPY_LOCAL"

rsync -avhP --exclude '*/tmp/***' "$QE_REMOTE" "$QE_LOCAL/"
rsync -avhP "$DFTPY_REMOTE" "$DFTPY_LOCAL/"

echo
echo "============================================================"
echo "Collect summaries"
echo "============================================================"
python "$REPO_ROOT/scripts/collect_qe_vcrelax_vacancy.py" --rootdir "$QE_LOCAL"
python "$REPO_ROOT/scripts/collect_dftpy_conventional_vacancy.py" --rootdir "$DFTPY_LOCAL"
python "$REPO_ROOT/scripts/collect_dftpy_vcrelax_fmax.py" "$DFTPY_LOCAL" --out "$LOCAL_BASE/dftpy_vcrelax_fmax_summary.csv"

echo
echo "============================================================"
echo "Create zip"
echo "============================================================"
ZIP_PATH="$LOCAL_BASE/vacancy_vcrelax_3x3x3_20260528.zip"
rm -f "$ZIP_PATH"
python - "$LOCAL_BASE" "$ZIP_PATH" <<'PY'
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
echo "[ZIP] $ZIP_PATH"
