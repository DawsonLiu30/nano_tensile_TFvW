param(
    [string]$PackageRoot = "$env:USERPROFILE\Desktop\FINAL_NUS_UPLOAD_20260526",
    [string]$RepoRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$PackageRoot = [System.IO.Path]::GetFullPath($PackageRoot)
$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)

function New-CleanGeneratedDir {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Root
    )

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $fullRoot = [System.IO.Path]::GetFullPath($Root)
    if (-not $fullPath.StartsWith($fullRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clean path outside package root: $fullPath"
    }

    if (Test-Path -LiteralPath $fullPath) {
        Remove-Item -LiteralPath $fullPath -Recurse -Force
    }
    New-Item -ItemType Directory -Path $fullPath | Out-Null
}

function Copy-DirectoryIfExists {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string]$PackageRoot
    )

    New-CleanGeneratedDir -Path $Destination -Root $PackageRoot
    if (Test-Path -LiteralPath $Source) {
        Copy-Item -Path (Join-Path $Source "*") -Destination $Destination -Recurse -Force
        return $true
    }
    return $false
}

function Copy-FileIfExists {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    if (Test-Path -LiteralPath $Source) {
        $parent = Split-Path -Parent $Destination
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
        Copy-Item -LiteralPath $Source -Destination $Destination -Force
        return $true
    }
    return $false
}

New-Item -ItemType Directory -Path $PackageRoot -Force | Out-Null

$reportDir = Join-Path $PackageRoot "00_FINAL_REPORT"
$qeDir = Join-Path $PackageRoot "01_QE_conventional_vacancy_2x2x4"
$dftpySameDir = Join-Path $PackageRoot "02_DFTpy_conventional_same_cell_2x2x4"
$dftpySizeDir = Join-Path $PackageRoot "03_DFTpy_conventional_size_concentration"
$scriptsDir = Join-Path $PackageRoot "04_scripts_and_reproducibility"
$pendingDir = Join-Path $PackageRoot "05_pending_or_not_final"

New-CleanGeneratedDir -Path $reportDir -Root $PackageRoot
New-CleanGeneratedDir -Path $scriptsDir -Root $PackageRoot
New-CleanGeneratedDir -Path $pendingDir -Root $PackageRoot

$qeSource = "$env:USERPROFILE\Desktop\qe_conventional_pull_20260522\qe_vacancy_conventional_2x2x4_20260522"
$dftpyPullRoot = "$env:USERPROFILE\Desktop\dftpy_conventional_pull_20260525"
$dftpySameSource = Join-Path $dftpyPullRoot "dftpy_vacancy_conventional_2x2x4_qe_a0_20260525"
$dftpySizeSource = Join-Path $dftpyPullRoot "dftpy_vacancy_conventional_size_qe_a0_20260525"

$copiedQe = Copy-DirectoryIfExists -Source $qeSource -Destination $qeDir -PackageRoot $PackageRoot
$copiedDftpySame = Copy-DirectoryIfExists -Source $dftpySameSource -Destination $dftpySameDir -PackageRoot $PackageRoot
$copiedDftpySize = Copy-DirectoryIfExists -Source $dftpySizeSource -Destination $dftpySizeDir -PackageRoot $PackageRoot
$copiedDftpyFmax = Copy-FileIfExists -Source (Join-Path $dftpyPullRoot "dftpy_conventional_actual_final_fmax_summary.csv") -Destination (Join-Path $dftpySizeDir "dftpy_conventional_actual_final_fmax_summary.csv")

Copy-FileIfExists -Source (Join-Path $RepoRoot "FINAL_REPORT_20260526.md") -Destination (Join-Path $reportDir "FINAL_REPORT_20260526.md") | Out-Null
Copy-FileIfExists -Source (Join-Path $RepoRoot "FINAL_PPT_CHINESE_SPEAKER_NOTES_20260527.md") -Destination (Join-Path $reportDir "FINAL_PPT_CHINESE_SPEAKER_NOTES_20260527.md") | Out-Null
Copy-FileIfExists -Source "$env:USERPROFILE\OneDrive\Documents\qe_bulk_vacancy_updated_20260525.pptx" -Destination (Join-Path $reportDir "FINAL_qe_bulk_vacancy_20260526.pptx") | Out-Null

$scriptNames = @(
    "pull_qe_vacancy_conventional_results.sh",
    "pull_dftpy_conventional_results.sh",
    "prepare_qe_vacancy_conventional_orthorhombic.py",
    "prepare_dftpy_vacancy_conventional.py",
    "run_dftpy_conventional_vacancy_one.py",
    "collect_dftpy_conventional_vacancy.py",
    "collect_dftpy_final_fmax.py",
    "build_final_nus_upload_package_20260526.ps1",
    "push_nanocrystal_tensile_pilot_to_iservice.sh",
    "pull_nanocrystal_tensile_pilot_results.sh"
)

foreach ($name in $scriptNames) {
    Copy-FileIfExists -Source (Join-Path $RepoRoot "scripts\$name") -Destination (Join-Path $scriptsDir $name) | Out-Null
}

Copy-FileIfExists -Source (Join-Path $RepoRoot "submit_nanocrystal_vacancy_prepare_pilot_20260526.sbatch") -Destination (Join-Path $scriptsDir "submit_nanocrystal_vacancy_prepare_pilot_20260526.sbatch") | Out-Null
Copy-FileIfExists -Source (Join-Path $RepoRoot "submit_nanocrystal_tensile_pilot_20260526.sbatch") -Destination (Join-Path $scriptsDir "submit_nanocrystal_tensile_pilot_20260526.sbatch") | Out-Null

$readme = @"
# FINAL_NUS_UPLOAD_20260526

This folder is the clean local package for professor/NUS upload.

## Folder map

- 00_FINAL_REPORT: one final report and one final slide deck copy.
- 01_QE_conventional_vacancy_2x2x4: complete QE conventional input/output/log/structure data pulled from iservice.
- 02_DFTpy_conventional_same_cell_2x2x4: complete DFTpy same-cell input/output/log/structure data.
- 03_DFTpy_conventional_size_concentration: complete DFTpy size/concentration data.
- 04_scripts_and_reproducibility: scripts needed to pull, regenerate, and collect the results.
- 05_pending_or_not_final: status notes for incomplete or not-final items.

## Pull commands if data are missing

Run these in WSL:

cd /mnt/c/Users/dawso/nano_tensile_TFvW && bash scripts/pull_qe_vacancy_conventional_results.sh
cd /mnt/c/Users/dawso/nano_tensile_TFvW && bash scripts/pull_dftpy_conventional_results.sh

Then rebuild this package in PowerShell:

powershell -ExecutionPolicy Bypass -File C:\Users\dawso\nano_tensile_TFvW\scripts\build_final_nus_upload_package_20260526.ps1

## Final PPT

Use this deck as the current final PPT copy:

00_FINAL_REPORT\FINAL_qe_bulk_vacancy_20260526.pptx

## Current copy status

- QE conventional copied: $copiedQe
- DFTpy same-cell copied: $copiedDftpySame
- DFTpy size/concentration copied: $copiedDftpySize
- DFTpy actual fmax summary copied: $copiedDftpyFmax
"@
$readme | Set-Content -Path (Join-Path $PackageRoot "README_FINAL_NUS_UPLOAD.md") -Encoding UTF8

$checklist = @"
# Upload Checklist

Upload this zip to NUS:

C:\Users\dawso\Desktop\FINAL_NUS_UPLOAD_20260526.zip

## Before upload

- [x] QE conventional 2x2x4 input/output/log/structure copied.
- [x] DFTpy conventional same-cell input/output/log/structure copied.
- [x] DFTpy size/concentration input/output/log/structure copied.
- [x] Final report copied.
- [x] Reproducibility scripts copied.
- [x] QE processed summary copied.
- [x] DFTpy spacing summary copied.
- [x] DFTpy size summary copied.
- [x] DFTpy actual fmax summary copied: $copiedDftpyFmax

## Key reportable numbers

- QE dense-k reference range: conventional 2x2x4, 4x4x4 to 6x6x6, E_f^vac about 0.5438-0.6778 eV.
- QE 6x6x6 latest value to include after pulling updated remote output: E_f^vac = 0.677782 eV.
- DFTpy same-cell spacing convergence: conventional 2x2x4, TFvW, spacing 0.30-0.16 Angstrom, E_f^vac about 2.901 eV.
- DFTpy same-cell actual fmax: all completed spacing cases satisfy fmax < 0.002 eV/Angstrom.
- DFTpy concentration scan accepted range: 32 to 500 pristine atoms, vacancy concentration 3.125% to 0.200%, E_f^vac decreases from 2.937948 to 2.883649 eV.

## Not final / do not overclaim

- If the local QE summary still marks 6x6x6 incomplete, rerun scripts/pull_qe_vacancy_conventional_results.sh before NUS upload.
- DFTpy conv_06x06x06 is not accepted because it did not meet fmax < 0.002 eV/Angstrom.
- QE fmax < 0.002 eV/Angstrom should not be claimed until explicitly verified from QE outputs.
"@
$checklist | Set-Content -Path (Join-Path $PackageRoot "UPLOAD_CHECKLIST.md") -Encoding UTF8

$pending = @"
# Pending / Not-Final Items

1. QE 6x6x6 kmesh is not accepted until the final vacancy relaxation output is pulled and collected.
2. QE final force should be verified from QE outputs before claiming fmax < 0.002 eV/Angstrom.
3. DFTpy conv_06x06x06 size scan is not accepted because actual fmax is 0.004161 eV/Angstrom and the result summary is not final.
4. The final PPT should be updated from FINAL_REPORT_20260526.md after the full DFTpy pull is present locally.
5. Nanocrystal tensile pilot should start only after inner/middle/outer vacancy structures pass visual inspection.
"@
$pending | Set-Content -Path (Join-Path $pendingDir "PENDING_NOT_FINAL.md") -Encoding UTF8

if (-not $copiedDftpySame -or -not $copiedDftpySize) {
    $pullNote = @"
DFTpy conventional data were not found locally at:

$dftpyPullRoot

Run this in WSL:

cd /mnt/c/Users/dawso/nano_tensile_TFvW && bash scripts/pull_dftpy_conventional_results.sh

Then rebuild the final package:

powershell -ExecutionPolicy Bypass -File C:\Users\dawso\nano_tensile_TFvW\scripts\build_final_nus_upload_package_20260526.ps1
"@
    $pullNote | Set-Content -Path (Join-Path $pendingDir "PULL_DFTPY_RESULTS_FIRST.txt") -Encoding UTF8
}

$zipPath = "$PackageRoot.zip"
if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -Path (Join-Path $PackageRoot "*") -DestinationPath $zipPath -Force

Write-Host "============================================================"
Write-Host "Final NUS upload package built"
Write-Host "============================================================"
Write-Host "[FOLDER] $PackageRoot"
Write-Host "[ZIP   ] $zipPath"
Write-Host "[QE copied        ] $copiedQe"
Write-Host "[DFTpy same copied] $copiedDftpySame"
Write-Host "[DFTpy size copied] $copiedDftpySize"
Write-Host "[DFTpy fmax copied] $copiedDftpyFmax"
