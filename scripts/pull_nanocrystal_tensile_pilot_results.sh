#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-iservice:/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"
LOCAL_ROOT="${LOCAL_ROOT:-/mnt/c/Users/dawso/Desktop/nanocrystal_tensile_pilot_pull_20260526}"

echo "============================================================"
echo "Pull nanocrystal tensile pilot results"
echo "============================================================"
echo "[REMOTE] $REMOTE_ROOT"
echo "[LOCAL ] $LOCAL_ROOT"
echo

mkdir -p "$LOCAL_ROOT"

for pos in inner middle outer; do
  case_name="nanocrystal_hexagon_periodic_111_2.0nm_vac_${pos}_tfvw"
  rsync -avhP "$REMOTE_ROOT/cases/$case_name/" "$LOCAL_ROOT/cases/$case_name/"
done

rsync -avhP "$REMOTE_ROOT/logs_nanocrystal_pilot/" "$LOCAL_ROOT/logs_nanocrystal_pilot/" || true

ZIP_PATH="$LOCAL_ROOT/nanocrystal_tensile_pilot_20260526.zip"
rm -f "$ZIP_PATH"
python - "$LOCAL_ROOT" "$ZIP_PATH" <<'PY'
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

