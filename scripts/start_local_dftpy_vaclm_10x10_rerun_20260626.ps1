param(
    [ValidateSet("pristine", "full")]
    [string]$Mode = "full",
    [ValidateSet("BFGS", "LBFGS", "BFGSLineSearch", "SciPyFminBFGS", "SciPyFminCG", "MDMin")]
    [string]$AseOptimizer = "MDMin",
    [int]$MaxCases = 0,
    [double]$AbortFmax = 100000.0,
    [int]$AbortAfterSteps = 80,
    [double]$MinPristineA0 = 3.5,
    [double]$MaxPristineA0 = 4.5,
    [string]$OutRoot = (Join-Path $env:USERPROFILE "Desktop\LOCAL_DFTPY_VACLM_10X10_FULL_RERUN_20260626"),
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\dawso\nano_tensile_TFvW"
$SourceRoot = Join-Path $Repo "iservice_packages\results\Al_defects\01_calibration\single_vacancy\dftpy_tfvw_lambda_mu\coarse_10x10_vacancy_formation"
$Runner = Join-Path $Repo "scripts\run_local_dftpy_vaclm_10x10_rerun.py"

if (-not (Test-Path $SourceRoot)) {
    throw "Source root not found: $SourceRoot"
}
if (-not (Test-Path $Runner)) {
    throw "Runner not found: $Runner"
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

$Python = (Get-Command python).Source
$Stdout = Join-Path $OutRoot "local_rerun.stdout.log"
$Stderr = Join-Path $OutRoot "local_rerun.stderr.log"

$ArgsList = @(
    $Runner,
    "--source-root", $SourceRoot,
    "--outroot", $OutRoot,
    "--mode", $Mode,
    "--ase-optimizer", $AseOptimizer,
    "--abort-fmax", [string]$AbortFmax,
    "--abort-after-steps", [string]$AbortAfterSteps,
    "--min-pristine-a0", [string]$MinPristineA0,
    "--max-pristine-a0", [string]$MaxPristineA0,
    "--collect-every", "5"
)

if ($MaxCases -gt 0) {
    $ArgsList += @("--max-cases", [string]$MaxCases)
}
if ($Force) {
    $ArgsList += "--force"
}

$Process = Start-Process `
    -FilePath $Python `
    -ArgumentList $ArgsList `
    -WorkingDirectory $Repo `
    -WindowStyle Hidden `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -PassThru

$Info = [ordered]@{
    pid = $Process.Id
    started_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    mode = $Mode
    ase_optimizer = $AseOptimizer
    abort_fmax = $AbortFmax
    abort_after_steps = $AbortAfterSteps
    min_pristine_a0 = $MinPristineA0
    max_pristine_a0 = $MaxPristineA0
    python = $Python
    repo = $Repo
    source_root = $SourceRoot
    outroot = $OutRoot
    stdout = $Stdout
    stderr = $Stderr
}

$InfoPath = Join-Path $OutRoot "BACKGROUND_PROCESS.json"
$Info | ConvertTo-Json -Depth 4 | Set-Content -Path $InfoPath -Encoding UTF8

Write-Host "Started local DFTpy VACLM rerun"
Write-Host "PID     : $($Process.Id)"
Write-Host "Mode    : $Mode"
Write-Host "OutRoot : $OutRoot"
Write-Host "Stdout  : $Stdout"
Write-Host "Stderr  : $Stderr"
Write-Host ""
Write-Host "Monitor:"
Write-Host "  Get-Content `"$Stdout`" -Tail 40 -Wait"
Write-Host "  Get-Content `"$OutRoot\local_rerun_progress.json`" -Raw"
