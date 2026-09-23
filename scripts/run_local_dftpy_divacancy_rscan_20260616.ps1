[CmdletBinding()]
param(
    [string]$Repo = (Split-Path -Parent $PSScriptRoot),
    [string]$OutDir = '',
    [double]$A0 = 3.9545804060131293,
    [string]$Repeat = '3x3x3',
    [string]$PP = 'al.lda.recpot',
    [double]$KedfX = 0.9,
    [double]$KedfY = 0.1,
    [double]$Spacing = 0.20,
    [double]$Fmax = 0.005,
    [int]$RelaxSteps = 5000,
    [ValidateSet('BFGS', 'LBFGS', 'BFGSLineSearch', 'SciPyFminBFGS', 'SciPyFminCG', 'MDMin')]
    [string]$AseOptimizer = 'BFGS',
    [ValidateSet('fixed_direction', 'shells')]
    [string]$PairSelection = 'fixed_direction',
    [string]$Direction = '1,1,0',
    [int]$MaxPairs = 0,
    [string]$Distro = 'Ubuntu-24.04',
    [string]$PythonPath = '/var/tmp/al-defects-runtime-20260907/env/bin/python',
    [switch]$PrepareOnly,
    [switch]$ReusePrepared
)

$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path -LiteralPath $Repo).Path
if (-not $OutDir) {
    $OutDir = Join-Path (Split-Path -Parent $Repo) ('LOCAL_RUNS\divacancy_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ'))
}
$OutDir = [IO.Path]::GetFullPath($OutDir)
if (-not [IO.Path]::IsPathRooted($PP)) { $PP = Join-Path $Repo $PP }
$PP = [IO.Path]::GetFullPath($PP)

function ConvertTo-WslPath([string]$Value) {
    if ($Value -notmatch '^([A-Za-z]):[\\/]') { throw "Expected a Windows drive path: $Value" }
    return '/mnt/' + $Matches[1].ToLowerInvariant() + '/' + $Value.Substring(3).Replace('\', '/')
}

$WslRepo = ConvertTo-WslPath $Repo
$WslOut = ConvertTo-WslPath $OutDir
function Invoke-ProjectPython([string[]]$PythonArgs) {
    & wsl.exe -d $Distro --cd $WslRepo -- $PythonPath -B @PythonArgs
    if ($LASTEXITCODE -ne 0) { throw "Project Python failed (exit $LASTEXITCODE): $($PythonArgs -join ' ')" }
}
function InvariantNumber($Value) { return $Value.ToString([Globalization.CultureInfo]::InvariantCulture) }

Write-Host "[REPO] $Repo"
Write-Host "[OUT] $OutDir"
Write-Host "[PROTOCOL] $PairSelection; direction=$Direction; LDA/TFVW lambda=$KedfX mu=$KedfY"

if (-not $ReusePrepared) {
    Invoke-ProjectPython @(
        "$WslRepo/scripts/prepare_dftpy_divacancy_rscan_20260616.py",
        '--outdir', $WslOut, '--a0', (InvariantNumber $A0), '--repeat', $Repeat,
        '--pp', (ConvertTo-WslPath $PP), '--xc', 'LDA', '--kedf', 'TFVW',
        '--kedf-x', (InvariantNumber $KedfX), '--kedf-y', (InvariantNumber $KedfY),
        '--spacing', (InvariantNumber $Spacing), '--fmax', (InvariantNumber $Fmax),
        '--relax-steps', "$RelaxSteps", '--ase-optimizer', $AseOptimizer,
        '--pair-selection', $PairSelection, "--direction=$Direction",
        '--partition', 'local', '--time-limit', '00:00:00', '--max-pairs', "$MaxPairs"
    )
} else {
    if (-not (Test-Path -LiteralPath (Join-Path $OutDir 'manifest.json'))) {
        throw "Prepared manifest not found in $OutDir"
    }
    Write-Host '[REUSE] Existing manifests determine the scientific parameters.'
}

$SettingsPath = Join-Path $OutDir 'settings_pair_scan.txt'
if (-not (Test-Path -LiteralPath $SettingsPath)) { throw "Missing settings: $SettingsPath" }
$Settings = @(Get-Content -LiteralPath $SettingsPath | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($Settings.Count -eq 0) { throw 'Prepared settings are empty.' }
if ($PrepareOnly) {
    Write-Host "[PREPARED] $($Settings.Count) cases: $OutDir"
    return
}

foreach ($Setting in $Settings) {
    if ($Setting -notmatch '^pair_[A-Za-z0-9_.-]+$') { throw "Invalid case setting: $Setting" }
    $CaseDir = Join-Path (Join-Path $OutDir 'pair_scan') $Setting
    $WslCase = ConvertTo-WslPath $CaseDir
    & wsl.exe -d $Distro --cd $WslRepo -- $PythonPath -B "$WslRepo/scripts/divacancy_analysis_checks.py" $WslCase
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[SKIP QUALIFIED] $Setting"
        continue
    }
    $RunArgs = @("$WslRepo/scripts/run_dftpy_vcrelax_vacancy_one.py", '--rootdir', $WslOut,
                 '--setting', $Setting, '--scan', 'pair', '--ase-optimizer', $AseOptimizer)
    if ($ReusePrepared) { $RunArgs += '--restart' }
    Write-Host "[RUN] $Setting"
    Invoke-ProjectPython $RunArgs
}
Invoke-ProjectPython @("$WslRepo/scripts/collect_dftpy_conventional_vacancy.py", '--rootdir', $WslOut)
Write-Host "[SUMMARY] $(Join-Path $OutDir 'analysis_current/dftpy_conventional_pair_summary.csv')"
