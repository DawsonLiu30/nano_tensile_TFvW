$ErrorActionPreference = "Stop"

$OutDir = "C:\Users\dawso\Desktop\LOCAL_PROFESS_VACANCY_RADIUS_SWEEP_20260604"

Write-Output "============================================================"
Write-Output "Local PROFESS vacancy radius sweep status"
Write-Output "============================================================"
Write-Output "[OUTDIR] $OutDir"
Write-Output ""

Write-Output "[Processes]"
Get-Process |
    Where-Object { $_.ProcessName -match "python|wsl|PROFESS" } |
    Select-Object ProcessName,Id,CPU,StartTime |
    Format-Table -AutoSize

Write-Output ""
Write-Output "[Completed output count by KEDF]"
@'
from pathlib import Path
import re

outdir = Path(r"C:\Users\dawso\Desktop\LOCAL_PROFESS_VACANCY_RADIUS_SWEEP_20260604")
for kedf_dir in sorted([p for p in outdir.iterdir() if p.is_dir() and not p.name.startswith("_")]):
    ok = 0
    total = 0
    latest = None
    for case_dir in sorted([p for p in kedf_dir.iterdir() if p.is_dir()]):
        outs = list(case_dir.glob("*.out"))
        if not outs:
            continue
        total += 1
        text = outs[0].read_text(errors="ignore")
        if "TOTAL ENERGY" in text and "END OF PROFESS" in text:
            ok += 1
        latest = max([latest, outs[0].stat().st_mtime] if latest else [outs[0].stat().st_mtime])
    print(f"{kedf_dir.name:16s} {ok:2d}/{total:2d} completed")
'@ | python -

Write-Output ""
Write-Output "[Result files]"
Get-ChildItem -LiteralPath $OutDir -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "profess_vacancy_radius|run_manifest" } |
    Select-Object Name,Length,LastWriteTime |
    Format-Table -AutoSize

Write-Output ""
Write-Output "[Recent outputs]"
Get-ChildItem -LiteralPath $OutDir -Recurse -File -Filter "*.out" |
    Sort-Object LastWriteTime |
    Select-Object -Last 12 LastWriteTime,Length,FullName |
    Format-Table -AutoSize
