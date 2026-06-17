param(
    [string]$OutDir = "C:\Users\dawso\Desktop\LOCAL_DFTPY_DIVACANCY_RSCAN_20260616",
    [string]$A0 = "4.039848",
    [string]$Repeat = "3x3x3",
    [string]$PP = "al.lda.recpot",
    [string]$KedfX = "1.0",
    [string]$KedfY = "0.13",
    [string]$Spacing = "0.20",
    [string]$Fmax = "0.002",
    [string]$RelaxSteps = "5000",
    [string]$AseOptimizer = "SciPyFminBFGS",
    [int]$MaxPairs = 0,
    [switch]$PrepareOnly,
    [switch]$ReusePrepared
)

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\dawso\nano_tensile_TFvW"
Set-Location $Repo

Write-Host "============================================================"
Write-Host "Local DFTpy divacancy r-scan"
Write-Host "============================================================"
Write-Host "[REPO  ] $Repo"
Write-Host "[OUT   ] $OutDir"
Write-Host "[CALC  ] LDA, TFVW, lambda=$KedfX, mu=$KedfY, spacing=$Spacing A"
Write-Host "[OPT   ] ASE $AseOptimizer"
Write-Host "[CELL  ] conventional fcc $Repeat; a0=$A0 A"

if (-not $ReusePrepared) {
    python scripts\prepare_dftpy_divacancy_rscan_20260616.py `
      --outdir "$OutDir" `
      --a0 "$A0" `
      --repeat "$Repeat" `
      --pp "$PP" `
      --xc LDA `
      --kedf TFVW `
      --kedf-x "$KedfX" `
      --kedf-y "$KedfY" `
      --spacing "$Spacing" `
      --fmax "$Fmax" `
      --relax-steps "$RelaxSteps" `
      --partition local `
      --time-limit 00:00:00 `
      --max-pairs "$MaxPairs"
} else {
    Write-Host "[INFO] Reusing prepared directory. No files will be regenerated."
}

if ($PrepareOnly) {
    Write-Host "[DONE] Prepared only."
    exit 0
}

$settings = Get-Content (Join-Path $OutDir "settings_pair_scan.txt") | Where-Object { $_.Trim() -ne "" }
foreach ($setting in $settings) {
    $caseDir = Join-Path (Join-Path $OutDir "pair_scan") $setting
    $resultPath = Join-Path $caseDir "result.json"
    if (Test-Path $resultPath) {
        Write-Host "============================================================"
        Write-Host "[SKIP] $setting already has result.json"
        continue
    }
    Write-Host "============================================================"
    Write-Host "[RUN] $setting"
    python scripts\run_dftpy_vcrelax_vacancy_one.py `
      --rootdir "$OutDir" `
      --setting "$setting" `
      --scan pair `
      --ase-optimizer "$AseOptimizer"
}

Write-Host "============================================================"
Write-Host "[COLLECT]"
python scripts\collect_dftpy_conventional_vacancy.py --rootdir "$OutDir"

Write-Host "============================================================"
Write-Host "Done"
Write-Host "============================================================"
Write-Host "[SUMMARY] $(Join-Path $OutDir 'dftpy_conventional_pair_summary.csv')"
