param(
    [Parameter(Mandatory = $true)]
    [int]$WaitPid,

    [double]$Step = 0.05,
    [int]$Cycles = 12,
    [double]$Ecut = 1000.0,
    [double]$A0 = 4.118877004246,
    [double]$FractureGapFactor = 3.0,
    [double]$Fmax = 0.02,
    [int]$TensileRelaxSteps = 80
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$case = "finite_grip_111_1.0nm_vacancy_tfvw"
$runName = "grip_r1_vacancy_tfvw_step5pct"
$caseDir = Join-Path $Root "cases\$case"

Write-Host "========================================"
Write-Host "[step5] Waiting for PID $WaitPid before starting r=1 5% scout run"
Write-Host "[step5] case      : $case"
Write-Host "[step5] run name  : $runName"
Write-Host "[step5] step      : $Step"
Write-Host "[step5] cycles    : $Cycles"
Write-Host "========================================"

$proc = Get-Process -Id $WaitPid -ErrorAction SilentlyContinue
if ($proc) {
    Wait-Process -Id $WaitPid
    Write-Host "[step5] PID $WaitPid finished."
} else {
    Write-Host "[step5] PID $WaitPid is not running. Starting immediately."
}

python scripts/run_grip_tensile.py `
    --case $runName `
    --workdir "cases\$case" `
    --init "inputs/vacancy_equilibrium.vasp" `
    --metadata "inputs/grip_metadata.json" `
    --pp "al.gga.recpot" `
    --kedf "TFVW" `
    --ecut "$Ecut" `
    --a0 "$A0" `
    --step "$Step" `
    --cycles "$Cycles" `
    --fmax "$Fmax" `
    --relax-steps "$TensileRelaxSteps" `
    --fracture-gap-factor "$FractureGapFactor" `
    --plot-summary
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$latest = Get-ChildItem -Path (Join-Path $caseDir "results") -Directory |
    Where-Object { $_.Name -like "$runName*" -and (Test-Path -LiteralPath (Join-Path $_.FullName "summary.csv")) } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $latest) {
    throw "[step5] Could not locate completed step5 result folder."
}

Write-Host "[step5] Running three-layer event analysis..."
python scripts/analyze_tensile_events.py `
    --results-dir $latest.FullName
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$out = Join-Path $Root "results\professor_review\r1_finite_grip_vacancy_tensile_step5pct.png"
Write-Host "[step5] Plotting: $($latest.FullName)"
python scripts/plot_grip_tensile_curve.py `
    --results-dir $latest.FullName `
    --cycles-target $Cycles `
    --out $out `
    --copy-csv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[step5] Done. Results in: $($latest.FullName)"
