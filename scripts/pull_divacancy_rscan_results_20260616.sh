#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
LOCAL_BASE="${LOCAL_BASE:-/mnt/c/Users/dawso/Desktop/DIVACANCY_RSCAN_RESULTS_20260616}"
REMOTE_HOST="${REMOTE_HOST:-iservice}"

QE_SERIES="${QE_SERIES:-qe_divacancy_vcrelax_conv3x3x3_rscan_20260616}"
QE_REMOTE_ROOT="${QE_REMOTE_ROOT:-/work/dawson666/qe_cases/qe_runs}"
QE_REMOTE="${REMOTE_HOST}:${QE_REMOTE_ROOT}/${QE_SERIES}/"

DFTPY_SERIES="${DFTPY_SERIES:-dftpy_divacancy_vcrelax_conv3x3x3_rscan_20260616}"
DFTPY_REMOTE_ROOT="${DFTPY_REMOTE_ROOT:-/work/dawson666/dftpy_project/relax/dftpy45}"
DFTPY_REMOTE="${REMOTE_HOST}:${DFTPY_REMOTE_ROOT}/results/${DFTPY_SERIES}/"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "[ERROR] Python is required for local collection." >&2
  exit 127
fi

RAW="${LOCAL_BASE}/raw"
PROCESSED="${LOCAL_BASE}/processed"
REPRO="${LOCAL_BASE}/reproducibility"

echo "============================================================"
echo "Pull divacancy r-scan results"
echo "============================================================"
echo "[LOCAL ] ${LOCAL_BASE}"
echo "[QE    ] ${QE_REMOTE}"
echo "[DFTpy ] ${DFTPY_REMOTE}"

mkdir -p "${RAW}/QE/${QE_SERIES}" "${RAW}/DFTpy/${DFTPY_SERIES}" "${PROCESSED}" "${REPRO}"

echo
echo "[1/5] Pull QE raw input/output"
rsync -avhP --exclude '*/tmp/***' "${QE_REMOTE}" "${RAW}/QE/${QE_SERIES}/"

echo
echo "[2/5] Pull DFTpy raw input/output"
rsync -avhP "${DFTPY_REMOTE}" "${RAW}/DFTpy/${DFTPY_SERIES}/"

echo
echo "[3/5] Pull scheduler logs and reproducibility scripts"
rsync -avhP --ignore-missing-args \
  "${REMOTE_HOST}:${QE_REMOTE_ROOT}/${QE_SERIES}/logs_submit/" \
  "${RAW}/QE/${QE_SERIES}/logs_submit/" || true
rsync -avhP --ignore-missing-args \
  "${REMOTE_HOST}:${DFTPY_REMOTE_ROOT}/logs_ctest/" \
  "${RAW}/DFTpy/_logs_ctest/" || true
rsync -avhP --relative --ignore-missing-args \
  "${REMOTE_HOST}:${DFTPY_REMOTE_ROOT}/./al.lda.recpot" \
  "${REMOTE_HOST}:${DFTPY_REMOTE_ROOT}/./app/dft_engine.py" \
  "${REMOTE_HOST}:${DFTPY_REMOTE_ROOT}/./scripts/prepare_dftpy_divacancy_rscan_20260616.py" \
  "${REMOTE_HOST}:${DFTPY_REMOTE_ROOT}/./scripts/run_dftpy_vcrelax_vacancy_one.py" \
  "${REMOTE_HOST}:${DFTPY_REMOTE_ROOT}/./scripts/collect_dftpy_conventional_vacancy.py" \
  "${REPRO}/DFTpy/" || true
rsync -avhP --relative --ignore-missing-args \
  "${REMOTE_HOST}:${QE_REMOTE_ROOT}/./scripts/prepare_qe_divacancy_vcrelax_rscan_20260616.py" \
  "${REMOTE_HOST}:${QE_REMOTE_ROOT}/./scripts/collect_qe_vcrelax_vacancy.py" \
  "${REPRO}/QE/" || true

echo
echo "[4/5] Collect local summaries"
"${PYTHON}" "${LOCAL_ROOT}/scripts/collect_qe_vcrelax_vacancy.py" \
  --rootdir "${RAW}/QE/${QE_SERIES}" \
  --out "${PROCESSED}/qe_divacancy_pair_summary.csv"

"${PYTHON}" "${LOCAL_ROOT}/scripts/collect_dftpy_conventional_vacancy.py" \
  --rootdir "${RAW}/DFTpy/${DFTPY_SERIES}"

cp "${RAW}/DFTpy/${DFTPY_SERIES}/dftpy_conventional_pair_summary.csv" \
  "${PROCESSED}/dftpy_divacancy_pair_summary.csv" 2>/dev/null || true

"${PYTHON}" - "${PROCESSED}" <<'PY'
from pathlib import Path
import csv
import math
import sys

processed = Path(sys.argv[1])
rows = []

def as_float(value):
    try:
        return float(value)
    except Exception:
        return math.nan

qe = processed / "qe_divacancy_pair_summary.csv"
if qe.exists():
    with qe.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ef = as_float(row.get("Ef_vac_eV"))
            rows.append({
                "method": "QE/PBE",
                "case": Path(row.get("path", "")).name,
                "r_A": row.get("pair_distance_A", ""),
                "N_pristine": row.get("N_pristine", ""),
                "N_defect": row.get("N_vacancy", ""),
                "vacancy_count": row.get("vacancy_count", ""),
                "Ef_2vac_eV": row.get("Ef_vac_eV", ""),
                "Ef_2vac_per_vacancy_eV": "" if math.isnan(ef) else f"{ef/2:.10f}",
                "pristine_done": row.get("pristine_done", ""),
                "defect_done": row.get("vacancy_done", ""),
                "source_path": row.get("path", ""),
            })

dft = processed / "dftpy_divacancy_pair_summary.csv"
if dft.exists():
    with dft.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ef = as_float(row.get("Ef_vac_eV"))
            rows.append({
                "method": "DFTpy/LDA/TFvW",
                "case": row.get("setting", ""),
                "r_A": row.get("pair_distance_A", ""),
                "N_pristine": row.get("N_pristine", ""),
                "N_defect": row.get("N_vacancy", ""),
                "vacancy_count": row.get("vacancy_count", ""),
                "Ef_2vac_eV": row.get("Ef_vac_eV", ""),
                "Ef_2vac_per_vacancy_eV": "" if math.isnan(ef) else f"{ef/2:.10f}",
                "pristine_done": row.get("done", ""),
                "defect_done": row.get("done", ""),
                "source_path": row.get("case_dir", ""),
            })

out = processed / "combined_divacancy_pair_summary.csv"
if rows:
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(out)
PY

echo
echo "[5/5] Build compact zip without QE tmp folders"
ZIP_PATH="${LOCAL_BASE}/DIVACANCY_RSCAN_RESULTS_20260616.zip"
rm -f "${ZIP_PATH}"
"${PYTHON}" - "${LOCAL_BASE}" "${ZIP_PATH}" <<'PY'
from pathlib import Path
import sys
import zipfile

source = Path(sys.argv[1]).resolve()
zip_path = Path(sys.argv[2]).resolve()
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in source.rglob("*"):
        if path.is_dir() or path == zip_path:
            continue
        rel = path.relative_to(source)
        if "tmp" in rel.parts:
            continue
        zf.write(path, rel)
PY

echo
echo "============================================================"
echo "Done"
echo "============================================================"
echo "[RAW      ] ${RAW}"
echo "[PROCESSED] ${PROCESSED}"
echo "[ZIP      ] ${ZIP_PATH}"
