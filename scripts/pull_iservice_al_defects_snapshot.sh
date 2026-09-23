#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-iservice}"
REMOTE_ROOT="${REMOTE_ROOT:-/work/dawson666/dftpy_project/relax/dftpy45}"
LOCAL_REPO="${LOCAL_REPO:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
LOCAL_BASE="${LOCAL_BASE:-/mnt/c/Users/dawso/Desktop/ISERVICE_AL_DEFECTS_SNAPSHOT_$(date +%Y%m%d_%H%M%S)}"

REMOTE_AL_DEFECTS="${REMOTE_ROOT}/results/Al_defects"
LOCAL_ROOT="${LOCAL_BASE}/dftpy45_snapshot"

echo "============================================================"
echo "Pull focused iservice Al_defects snapshot"
echo "============================================================"
echo "[REMOTE] ${REMOTE_HOST}:${REMOTE_ROOT}"
echo "[LOCAL ] ${LOCAL_ROOT}"
echo "[REPO  ] ${LOCAL_REPO}"
echo
echo "This intentionally does NOT mirror the full dftpy45 tree."
echo "It pulls Al_defects results, Al_defects logs, and reproducibility files."
echo

mkdir -p "${LOCAL_ROOT}"

echo "============================================================"
echo "[1/4] Remote preflight summary"
echo "============================================================"
ssh "${REMOTE_HOST}" "
set -euo pipefail
echo '[REMOTE_ROOT]' '${REMOTE_ROOT}'
test -d '${REMOTE_AL_DEFECTS}'
echo
echo '[Al_defects size]'
du -sh '${REMOTE_AL_DEFECTS}' 2>/dev/null || true
echo
echo '[Result counts]'
printf 'result.json: '
find '${REMOTE_AL_DEFECTS}' -name result.json -type f -size +0c 2>/dev/null | wc -l
printf 'run_status*.json: '
find '${REMOTE_AL_DEFECTS}' -name 'run_status*.json' -type f 2>/dev/null | wc -l
printf 'FAILED markers: '
find '${REMOTE_AL_DEFECTS}' \( -name 'FAILED_*' -o -name 'WORKER_FAILED.txt' \) 2>/dev/null | wc -l
printf 'RUNNING markers: '
find '${REMOTE_AL_DEFECTS}' \( -name 'RUNNING_*' -o -name 'RUNNING_LOCK' \) 2>/dev/null | wc -l
echo
echo '[Recent Al_defects logs]'
find '${REMOTE_ROOT}/logs/Al_defects' -type f 2>/dev/null \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' | sort | tail -20 || true
"

echo
echo "============================================================"
echo "[2/4] Pull results/Al_defects"
echo "============================================================"
mkdir -p "${LOCAL_ROOT}/results"
rsync -avhP --partial \
  "${REMOTE_HOST}:${REMOTE_AL_DEFECTS}/" \
  "${LOCAL_ROOT}/results/Al_defects/"

echo
echo "============================================================"
echo "[3/4] Pull logs and reproducibility files"
echo "============================================================"
mkdir -p "${LOCAL_ROOT}/logs/Al_defects"
rsync -avhP --partial \
  "${REMOTE_HOST}:${REMOTE_ROOT}/logs/Al_defects/" \
  "${LOCAL_ROOT}/logs/Al_defects/" || true

mkdir -p "${LOCAL_ROOT}/_reproducibility/scripts"
rsync -avhP --partial \
  --include='*/' \
  --include='prepare_dftpy*.py' \
  --include='run_dftpy*.py' \
  --include='collect_dftpy*.py' \
  --include='audit_vacancy_submission_gate.py' \
  --include='materialize_dftpy_case_reproducibility.py' \
  --include='submit_dftpy*.sh' \
  --include='check_iservice_al_defects_status.sh' \
  --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_ROOT}/scripts/" \
  "${LOCAL_ROOT}/_reproducibility/scripts/" || true

rsync -avhP --partial \
  "${REMOTE_HOST}:${REMOTE_ROOT}/al.lda.recpot" \
  "${LOCAL_ROOT}/_reproducibility/" || true

echo
echo "============================================================"
echo "[4/4] Local audit"
echo "============================================================"
python_bin=""
if command -v python3 >/dev/null 2>&1; then
  python_bin="python3"
elif command -v python.exe >/dev/null 2>&1; then
  python_bin="python.exe"
elif command -v python >/dev/null 2>&1; then
  python_bin="python"
fi

if [[ -z "${python_bin}" ]]; then
  echo "[WARN] No Python found; skipping structured local audit." >&2
else
  SNAPSHOT_ROOT="${LOCAL_ROOT}" "${python_bin}" - <<'PY'
import csv
import json
import os
from collections import Counter
from pathlib import Path

root = Path(os.environ["SNAPSHOT_ROOT"])
al = root / "results" / "Al_defects"
logs = root / "logs" / "Al_defects"
audit_dir = root / "_audit"
audit_dir.mkdir(parents=True, exist_ok=True)

result_paths = sorted(al.rglob("result.json"))
status_paths = sorted(al.rglob("run_status*.json"))
failed_markers = sorted(
    [p for p in al.rglob("FAILED_*") if p.is_file()]
    + [p for p in al.rglob("WORKER_FAILED.txt") if p.is_file()]
)
running_markers = sorted(
    [p for p in al.rglob("RUNNING_*") if p.is_file()]
    + [p for p in al.rglob("RUNNING_LOCK") if p.exists()]
)

status_rows = []
status_counter = Counter()
rc_counter = Counter()
for path in status_paths:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        status_rows.append({
            "path": str(path.relative_to(root)),
            "setting": "",
            "task_id": "",
            "status": "JSON_READ_ERROR",
            "python_or_timeout_rc": "",
            "has_result_json": "",
            "note": str(exc),
        })
        status_counter["JSON_READ_ERROR"] += 1
        continue
    status = str(data.get("status", ""))
    rc = str(data.get("python_or_timeout_rc", ""))
    status_counter[status] += 1
    rc_counter[rc] += 1
    status_rows.append({
        "path": str(path.relative_to(root)),
        "setting": data.get("setting", ""),
        "task_id": data.get("task_id", ""),
        "status": status,
        "python_or_timeout_rc": rc,
        "has_result_json": data.get("has_result_json", ""),
        "note": "",
    })

with (audit_dir / "run_status_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "path",
            "setting",
            "task_id",
            "status",
            "python_or_timeout_rc",
            "has_result_json",
            "note",
        ],
    )
    writer.writeheader()
    writer.writerows(status_rows)

timeout_error_hits = []
for path in sorted(logs.rglob("*.out")) + sorted(logs.rglob("*.err")):
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        continue
    if "timeout: invalid time interval" in text:
        timeout_error_hits.append(str(path.relative_to(root)))

lines = [
    "# Local Al_defects Snapshot Audit",
    "",
    f"snapshot_root: `{root}`",
    "",
    "## Counts",
    "",
    f"- result.json files: {len(result_paths)}",
    f"- run_status*.json files: {len(status_paths)}",
    f"- failed markers: {len(failed_markers)}",
    f"- running markers/locks: {len(running_markers)}",
    f"- logs with invalid timeout interval: {len(timeout_error_hits)}",
    "",
    "## run_status counts",
    "",
]
if status_counter:
    lines.extend(f"- {k or '(blank)'}: {v}" for k, v in sorted(status_counter.items()))
else:
    lines.append("- none")
lines.extend(["", "## python_or_timeout_rc counts", ""])
if rc_counter:
    lines.extend(f"- {k or '(blank)'}: {v}" for k, v in sorted(rc_counter.items()))
else:
    lines.append("- none")

lines.extend(["", "## Important Interpretation", ""])
if timeout_error_hits:
    lines.extend([
        "At least one log contains `timeout: invalid time interval`.",
        "Those tasks failed before entering the DFTpy Python runner; this is a submission-script timeout-format problem, not a physics/structure convergence failure.",
        "",
        "Affected logs:",
    ])
    lines.extend(f"- `{p}`" for p in timeout_error_hits[:40])
    if len(timeout_error_hits) > 40:
        lines.append(f"- ... plus {len(timeout_error_hits) - 40} more")
else:
    lines.append("No invalid timeout interval message was found in pulled logs.")

(audit_dir / "LOCAL_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[AUDIT] wrote {audit_dir / 'LOCAL_AUDIT.md'}")
print(f"[AUDIT] wrote {audit_dir / 'run_status_summary.csv'}")
print(f"[AUDIT] result_json={len(result_paths)} run_status={len(status_paths)} invalid_timeout_logs={len(timeout_error_hits)}")
PY

  analyzer="${LOCAL_REPO}/scripts/analyze_al_defects_snapshot_gillan_style.py"
  if [[ -s "${analyzer}" ]]; then
    echo
    echo "[Gillan-style vacancy analysis]"
    "${python_bin}" "${analyzer}" --snapshot-root "${LOCAL_ROOT}" || {
      echo "[WARN] Gillan-style analyzer failed; snapshot pull itself is still complete." >&2
    }
  else
    echo "[WARN] Analyzer not found: ${analyzer}" >&2
  fi
fi

echo
echo "============================================================"
echo "Snapshot complete"
echo "============================================================"
echo "[LOCAL_ROOT] ${LOCAL_ROOT}"
echo "[AUDIT     ] ${LOCAL_ROOT}/_audit/LOCAL_AUDIT.md"
echo "[ANALYSIS  ] ${LOCAL_ROOT}/_analysis_gillan_style"
echo "============================================================"
