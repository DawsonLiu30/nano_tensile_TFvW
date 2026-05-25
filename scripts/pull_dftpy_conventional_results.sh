#!/usr/bin/env bash
set -euo pipefail

REMOTE_BASE="${REMOTE_BASE:-iservice:/gpfs-work/dawson666/dftpy_project/relax/dftpy45/results}"
LOCAL_BASE="${LOCAL_BASE:-/mnt/c/Users/dawso/Desktop/dftpy_conventional_pull_20260525}"
REPO_ROOT="${REPO_ROOT:-/mnt/c/Users/dawso/nano_tensile_TFvW}"

SAME_CELL="dftpy_vacancy_conventional_2x2x4_qe_a0_20260525"
SIZE_SCAN="dftpy_vacancy_conventional_size_qe_a0_20260525"

echo "============================================================"
echo "Pull DFTpy conventional vacancy results"
echo "============================================================"
echo "[REMOTE] $REMOTE_BASE"
echo "[LOCAL ] $LOCAL_BASE"
echo "[REPO  ] $REPO_ROOT"
echo

mkdir -p "$LOCAL_BASE"

rsync -avhP "$REMOTE_BASE/$SAME_CELL/" "$LOCAL_BASE/$SAME_CELL/"
rsync -avhP "$REMOTE_BASE/$SIZE_SCAN/" "$LOCAL_BASE/$SIZE_SCAN/"

echo
echo "============================================================"
echo "Collect local summaries"
echo "============================================================"
python "$REPO_ROOT/scripts/collect_dftpy_conventional_vacancy.py" --rootdir "$LOCAL_BASE/$SAME_CELL"
python "$REPO_ROOT/scripts/collect_dftpy_conventional_vacancy.py" --rootdir "$LOCAL_BASE/$SIZE_SCAN"
python "$REPO_ROOT/scripts/collect_dftpy_final_fmax.py" \
  "$LOCAL_BASE/$SAME_CELL" \
  "$LOCAL_BASE/$SIZE_SCAN" \
  --out "$LOCAL_BASE/dftpy_conventional_actual_final_fmax_summary.csv"

echo
echo "============================================================"
echo "Create upload/review zip"
echo "============================================================"
ZIP_PATH="$LOCAL_BASE/dftpy_conventional_vacancy_20260525.zip"
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
        zf.write(path, path.relative_to(source.parent))
PY
echo "[ZIP] $ZIP_PATH"
