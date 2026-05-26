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

Copy-FileIfExists -Source (Join-Path $RepoRoot "FINAL_REPORT_20260526.md") -Destination (Join-Path $reportDir "FINAL_REPORT_20260526.md") | Out-Null
Copy-FileIfExists -Source "$env:USERPROFILE\OneDrive\Documents\qe_bulk_vacancy_updated_20260525.pptx" -Destination (Join-Path $reportDir "latest_slide_deck_qe_bulk_vacancy_updated_20260525.pptx") | Out-Null

$scriptNames = @(
    "pull_qe_vacancy_conventional_results.sh",
    "pull_dftpy_conventional_results.sh",
    "prepare_qe_vacancy_conventional_orthorhombic.py",
    "prepare_dftpy_vacancy_conventional.py",
    "run_dftpy_conventional_vacancy_one.py",
    "collect_dftpy_conventional_vacancy.py",
    "collect_dftpy_final_fmax.py",
    "build_final_nus_upload_package_20260526.ps1"
)

foreach ($name in $scriptNames) {
    Copy-FileIfExists -Source (Join-Path $RepoRoot "scripts\$name") -Destination (Join-Path $scriptsDir $name) | Out-Null
}

$readme = @"
# FINAL_NUS_UPLOAD_20260526

This folder is the clean local package for professor/NUS upload.

## Folder map

- 00_FINAL_REPORT: one final report and the latest slide deck copy.
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

## Current copy status

- QE conventional copied: $copiedQe
- DFTpy same-cell copied: $copiedDftpySame
- DFTpy size/concentration copied: $copiedDftpySize
"@
$readme | Set-Content -Path (Join-Path $PackageRoot "README_FINAL_NUS_UPLOAD.md") -Encoding UTF8

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
