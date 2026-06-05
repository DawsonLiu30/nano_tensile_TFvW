$ErrorActionPreference = "Stop"

$Repo = "C:\Users\dawso\nano_tensile_TFvW"
$SourceRoot = "C:\Users\dawso\Desktop\NUS_upload\2026-04-21_full_sync\qe_runs\vacancy_nanocrystal_relax"
$OutDir = "C:\Users\dawso\Desktop\LOCAL_PROFESS_VACANCY_RADIUS_SWEEP_20260604"
$LogDir = Join-Path $OutDir "_logs"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

$ArgList = @(
    "scripts\run_local_profess_vacancy_radius_sweep.py",
    "--source-root", $SourceRoot,
    "--outdir", $OutDir,
    "--kedf", "WT",
    "--kedf", "HC",
    "--kedf", "TFPLUS_DEFAULT",
    "--kedf", "TFVW_L1_M1",
    "--timeout-s", "7200",
    "--resume"
)

$Process = Start-Process `
    -FilePath "python" `
    -ArgumentList $ArgList `
    -WorkingDirectory $Repo `
    -RedirectStandardOutput (Join-Path $LogDir "profess_radius_sweep.stdout.log") `
    -RedirectStandardError (Join-Path $LogDir "profess_radius_sweep.stderr.log") `
    -WindowStyle Hidden `
    -PassThru

Write-Output "Started local PROFESS vacancy radius KEDF sweep."
Write-Output "PID: $($Process.Id)"
Write-Output "Output directory: $OutDir"
Write-Output "Stdout log: $(Join-Path $LogDir "profess_radius_sweep.stdout.log")"
Write-Output "Stderr log: $(Join-Path $LogDir "profess_radius_sweep.stderr.log")"
