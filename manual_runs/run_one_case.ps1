param(
    [Parameter(Mandatory = $true)]
    [double]$Diameter,

    [int]$Cycles = 20,
    [double]$Step = 0.01,
    [double]$Ecut = 1000.0,
    [double]$A0 = 4.118877004246,
    [double]$FractureGapFactor = 3.0,
    [double]$Fmax = 0.02,
    [int]$PrepRelaxSteps = 120,
    [int]$TensileRelaxSteps = 80,

    [switch]$ForcePrepare,
    [switch]$ResumeLatest,
    [switch]$PlotOnly
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Get-RTag([double]$Value) {
    $rounded = [math]::Round($Value)
    if ([math]::Abs($Value - $rounded) -lt 1.0e-9) {
        return [string][int]$rounded
    }
    return ((("{0:F3}" -f $Value).TrimEnd("0")).TrimEnd(".")).Replace(".", "p")
}

$diameterText = "{0:F1}" -f $Diameter
$tag = Get-RTag $Diameter
$case = "finite_grip_111_${diameterText}nm_vacancy_tfvw"
$runName = "grip_r${tag}_vacancy_tfvw"
$caseDir = Join-Path $Root "cases\$case"
$inputVasp = Join-Path $caseDir "inputs\vacancy_equilibrium.vasp"
$metadata = Join-Path $caseDir "inputs\grip_metadata.json"

Write-Host "========================================"
Write-Host "[manual] case      : $case"
Write-Host "[manual] run name  : $runName"
Write-Host "[manual] case dir  : $caseDir"
Write-Host "========================================"

if (-not $PlotOnly) {
    if ($ForcePrepare -or -not (Test-Path -LiteralPath $inputVasp) -or -not (Test-Path -LiteralPath $metadata)) {
        Write-Host "[manual] Preparing relaxed finite-grip vacancy inputs..."
        python scripts/prepare_grip_vacancy_wire.py `
            --case $case `
            --diameter-nm $diameterText `
            --orientation 111 `
            --a0 $A0 `
            --wire-length 21.0 `
            --min-wire-span 10.0 `
            --xy-vacuum 10.0 `
            --z-vacuum 10.0 `
            --grip-thickness 3.0 `
            --vacancy-z-window-fraction 0.35 `
            --pp al.gga.recpot `
            --kedf TFVW `
            --ecut $Ecut `
            --fmax $Fmax `
            --relax-steps $PrepRelaxSteps
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } else {
        Write-Host "[manual] Existing relaxed inputs found; skipping prepare. Use -ForcePrepare to rebuild."
    }

    $tensileArgs = @(
        "scripts/run_grip_tensile.py",
        "--case", $runName,
        "--workdir", "cases\$case",
        "--init", "inputs/vacancy_equilibrium.vasp",
        "--metadata", "inputs/grip_metadata.json",
        "--pp", "al.gga.recpot",
        "--kedf", "TFVW",
        "--ecut", "$Ecut",
        "--a0", "$A0",
        "--step", "$Step",
        "--cycles", "$Cycles",
        "--fmax", "$Fmax",
        "--relax-steps", "$TensileRelaxSteps",
        "--fracture-gap-factor", "$FractureGapFactor",
        "--plot-summary"
    )

    if ($ResumeLatest) {
        $latest = Get-ChildItem -Path (Join-Path $caseDir "results") -Directory -ErrorAction SilentlyContinue |
            Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "summary.csv") } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($latest) {
            Write-Host "[manual] Resuming latest result folder: $($latest.FullName)"
            $tensileArgs += @("--resume-results", $latest.FullName)
        } else {
            Write-Host "[manual] -ResumeLatest requested, but no previous summary.csv was found. Starting a new tensile run."
        }
    }

    Write-Host "[manual] Running tensile..."
    python @tensileArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$latest = Get-ChildItem -Path (Join-Path $caseDir "results") -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "$runName*" -and (Test-Path -LiteralPath (Join-Path $_.FullName "summary.csv")) } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $latest) {
    throw "[manual] Could not locate result folder for $runName"
}

Write-Host "[manual] Running three-layer event analysis..."
python scripts/analyze_tensile_events.py `
    --results-dir $latest.FullName
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$out = Join-Path $Root "results\professor_review\r${tag}_finite_grip_vacancy_tensile.png"
Write-Host "[manual] Plotting latest stress-strain curve..."
python scripts/plot_grip_tensile_curve.py `
    --results-dir $latest.FullName `
    --cycles-target $Cycles `
    --out $out `
    --copy-csv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[manual] Done."
