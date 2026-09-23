#!/usr/bin/env bash
set -euo pipefail

RUNROOT="$HOME/qe_vaclm_local_20260630"
DEST="/mnt/c/Users/dawso/Desktop/QE_VACLM_REFERENCE_20260630_LOCAL_RESULTS"

mkdir -p "$DEST"

cp -f "$RUNROOT"/LOCAL_QE_STATUS.tsv "$DEST"/ 2>/dev/null || true
cp -f "$RUNROOT"/LOCAL_QE_SUMMARY.txt "$DEST"/ 2>/dev/null || true
cp -f "$RUNROOT"/LOCAL_QE_DRIVER.out "$DEST"/ 2>/dev/null || true
cp -f "$RUNROOT"/LOCAL_QE_DRIVER.err "$DEST"/ 2>/dev/null || true
cp -f "$RUNROOT"/README_RUN_ON_ISERVICE.md "$DEST"/ 2>/dev/null || true
cp -f "$RUNROOT"/submit_qe_vacancy_reference_array.sh "$DEST"/ 2>/dev/null || true

for case in pristine_vcrelax vacancy_vcrelax; do
  mkdir -p "$DEST/$case"
  for f in pw.in pw.out pw.err local_qe_run.log; do
    cp -f "$RUNROOT/$case/$f" "$DEST/$case/" 2>/dev/null || true
  done
done

mkdir -p "$DEST/structures" "$DEST/pseudo"
cp -f "$RUNROOT"/structures/* "$DEST/structures/" 2>/dev/null || true
cp -f "$RUNROOT"/pseudo/Al_PAW_PBE.UPF "$DEST/pseudo/" 2>/dev/null || true

cat > "$DEST/README_LOCAL_QE_RESULTS.md" <<EOF
# Local QE VACLM Reference Results

Source WSL run directory:

\`\`\`text
$RUNROOT
\`\`\`

This folder intentionally copies only input/output/provenance files, not the
large QE scratch directories under \`tmp/\`.

Check:

\`\`\`powershell
wsl.exe -d Ubuntu -- bash /mnt/c/Users/dawso/nano_tensile_TFvW/scripts/status_local_qe_vaclm_run_20260630.sh
\`\`\`
EOF

echo "$DEST"
