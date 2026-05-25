#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-iservice:/gpfs-work/dawson666/qe_cases/qe_runs/qe_vacancy_conventional_2x2x4_20260522/}"
LOCAL_ROOT="${LOCAL_ROOT:-/mnt/c/Users/dawso/Desktop/qe_conventional_pull_20260522/qe_vacancy_conventional_2x2x4_20260522/}"
REPO_ROOT="${REPO_ROOT:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
ARCHIVE_DIR="${ARCHIVE_DIR:-/mnt/c/Users/dawso/Desktop/qe_conventional_pull_20260522}"

echo "============================================================"
echo "Pull QE conventional vacancy rerun"
echo "============================================================"
echo "[REMOTE] $REMOTE_ROOT"
echo "[LOCAL ] $LOCAL_ROOT"
echo "[REPO  ] $REPO_ROOT"
echo

mkdir -p "$LOCAL_ROOT"

rsync -avhP --exclude '*/tmp/***' "$REMOTE_ROOT" "$LOCAL_ROOT"

echo
echo "============================================================"
echo "Collect local QE vacancy summary"
echo "============================================================"
python "$REPO_ROOT/scripts/collect_all_qe_vacancy_recursive.py" --rootdir "$LOCAL_ROOT"

SUMMARY="$LOCAL_ROOT/processed_vacancy_convergence/qe_vacancy_all_recursive_summary.csv"
echo
echo "[SUMMARY] $SUMMARY"

echo
echo "============================================================"
echo "Completion check"
echo "============================================================"
python - "$SUMMARY" <<'PY'
from pathlib import Path
import csv
import sys

summary = Path(sys.argv[1])
if not summary.exists():
    raise SystemExit(f"Summary CSV not found: {summary}")

rows = list(csv.DictReader(summary.open(newline="", encoding="utf-8")))
done = [r for r in rows if r["pristine_done"] == "True" and r["vacancy_done"] == "True"]
pending = [r for r in rows if not (r["pristine_done"] == "True" and r["vacancy_done"] == "True")]

print(f"Total cases     : {len(rows)}")
print(f"Completed cases : {len(done)}")
print(f"Pending cases   : {len(pending)}")
if pending:
    print()
    print("Pending / incomplete:")
    for r in pending:
        print(f"- {r['path']}  P={r['pristine_done']} V={r['vacancy_done']}")
PY

echo
echo "============================================================"
echo "Create upload/review zip without QE tmp folders"
echo "============================================================"
mkdir -p "$ARCHIVE_DIR"
ZIP_PATH="$ARCHIVE_DIR/qe_vacancy_conventional_2x2x4_20260522_no_tmp.zip"
rm -f "$ZIP_PATH"
if command -v zip >/dev/null 2>&1; then
  (
    cd "$(dirname "$LOCAL_ROOT")"
    zip -r "$ZIP_PATH" "$(basename "$LOCAL_ROOT")" -x '*/tmp/*'
  )
else
  python - "$LOCAL_ROOT" "$ZIP_PATH" <<'PY'
from pathlib import Path
import sys
import zipfile

source = Path(sys.argv[1]).resolve()
zip_path = Path(sys.argv[2]).resolve()
base = source.parent

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in source.rglob("*"):
        if path.is_dir():
            continue
        if "tmp" in path.relative_to(source).parts:
            continue
        zf.write(path, path.relative_to(base))
PY
fi
echo "[ZIP] $ZIP_PATH"
