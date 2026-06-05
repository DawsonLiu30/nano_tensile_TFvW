param(
    [string]$Source = "C:\Users\dawso\Desktop\LATEST_VACANCY_BENCHMARK_20260601",
    [string]$Destination = "C:\Users\dawso\Desktop\FINAL_NUS_VACANCY_DATABASE_20260601",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if (!(Test-Path -LiteralPath $Source)) {
    throw "Missing source folder: $Source"
}

$resolvedDestination = [System.IO.Path]::GetFullPath($Destination)
$expectedLeaf = "FINAL_NUS_VACANCY_DATABASE_20260601"
$allowedDestination = [System.IO.Path]::GetFullPath((Join-Path $env:USERPROFILE "Desktop\$expectedLeaf"))
if (!$resolvedDestination.Equals($allowedDestination, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to replace unexpected destination: $resolvedDestination"
}

if (Test-Path -LiteralPath $Destination) {
    if (!$Force) {
        throw "Destination already exists. Re-run with -Force to replace it safely: $Destination"
    }
    Remove-Item -LiteralPath $Destination -Recurse -Force
}

$dirs = @(
    "00_START_HERE",
    "01_PROCESSED_RESULTS",
    "02_QE_PBE_VCRELAX_3x3x3_RAW",
    "03_DFTPY_LDA_TFVW_RAW",
    "04_DFTPY_LDA_WT_RAW",
    "05_DFTPY_LDA_SM_RAW",
    "06_REPRODUCIBILITY",
    "07_PRESENTATION"
)

New-Item -ItemType Directory -Path $Destination | Out-Null
foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Path (Join-Path $Destination $dir) | Out-Null
}
New-Item -ItemType Directory -Path (Join-Path $Destination "06_REPRODUCIBILITY\remote_dftpy_support") | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Destination "06_REPRODUCIBILITY\local_organizer_scripts") | Out-Null

Copy-Item -LiteralPath (Join-Path $Source "README_LATEST_RESULTS.md") -Destination (Join-Path $Destination "00_START_HERE\README_LATEST_RESULTS.md")
Copy-Item -Path (Join-Path $Source "processed\*") -Destination (Join-Path $Destination "01_PROCESSED_RESULTS") -Recurse
Copy-Item -Path (Join-Path $Source "raw\QE\*") -Destination (Join-Path $Destination "02_QE_PBE_VCRELAX_3x3x3_RAW") -Recurse
Copy-Item -Path (Join-Path $Source "raw\DFTpy_TFVW\*") -Destination (Join-Path $Destination "03_DFTPY_LDA_TFVW_RAW") -Recurse
Copy-Item -Path (Join-Path $Source "raw\DFTpy_WT\*") -Destination (Join-Path $Destination "04_DFTPY_LDA_WT_RAW") -Recurse
Copy-Item -Path (Join-Path $Source "raw\DFTpy_SM\*") -Destination (Join-Path $Destination "05_DFTPY_LDA_SM_RAW") -Recurse
Copy-Item -Path (Join-Path $Source "raw\DFTpy_support\*") -Destination (Join-Path $Destination "06_REPRODUCIBILITY\remote_dftpy_support") -Recurse
Copy-Item -Path (Join-Path $Source "organizer_scripts\*") -Destination (Join-Path $Destination "06_REPRODUCIBILITY\local_organizer_scripts") -Recurse
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $Destination "06_REPRODUCIBILITY\local_organizer_scripts") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "audit_final_nus_vacancy_structures.py") -Destination (Join-Path $Destination "06_REPRODUCIBILITY\local_organizer_scripts") -Force

$qe5 = Import-Csv -LiteralPath (Join-Path $Source "processed\final_method_comparison.csv") |
    Where-Object { $_.method -eq "QE" -and $_.reference_setting -like "k=5x5x5,*" } |
    Select-Object -First 1
$qe5Value = if ($qe5) { "{0:F6}" -f [double]$qe5.Ef_vac_eV } else { "n/a" }
$qe5TableLine = if ($qe5 -and $qe5.status -eq "completed") {
    "| 5x5x5 | $qe5Value eV | completed |"
} else {
    "| 5x5x5 | $qe5Value eV | provisional only |"
}
$qe5Note = if ($qe5 -and $qe5.status -eq "completed") {
    "QE 5x5x5 has completed and is included as a formal k-mesh result."
} else {
    "QE 5x5x5 remains incomplete and must be treated as provisional only."
}

$readme = @"
# FINAL NUS Vacancy Benchmark Database

## Read First

This package contains the corrected Al vacancy benchmark using a conventional
cubic fcc 3x3x3 centered-vacancy supercell:

- pristine: 108 atoms
- vacancy: 107 atoms
- vacancy concentration: 1/108 = 0.925926%

## Folder Guide

| Folder | Contents |
|---|---|
| 00_START_HERE | Short benchmark summary |
| 01_PROCESSED_RESULTS | Compact CSV tables and plots |
| 02_QE_PBE_VCRELAX_3x3x3_RAW | QE/PBE PAW vc-relax inputs, outputs, and structures |
| 03_DFTPY_LDA_TFVW_RAW | DFTpy/LDA + TFVW full atom+cell relaxation data |
| 04_DFTPY_LDA_WT_RAW | DFTpy/LDA + WT full atom+cell relaxation data |
| 05_DFTPY_LDA_SM_RAW | DFTpy/LDA + SM full atom+cell relaxation data |
| 06_REPRODUCIBILITY | Pseudopotential and collection/runner scripts |
| 07_PRESENTATION | Concise PPT update brief; final PPT pending tool-compatible export |

## Key Result

Under the same LDA local pseudopotential, same 3x3x3 centered-vacancy cell, and
same full atom+cell relaxation workflow, the DFTpy vacancy formation energy is
strongly KEDF-dependent:

| KEDF | Ef_vac at spacing 0.20 A |
|---|---:|
| TFVW | 3.280521 eV |
| WT | 1.333804 eV |
| SM | 0.682758 eV |

Completed QE/PBE vc-relax results:

| k-mesh | Ef_vac | status |
|---|---:|---|
| 2x2x2 | 0.989725 eV | completed |
| 3x3x3 | 0.644942 eV | completed |
| 4x4x4 | 0.677875 eV | completed |
$qe5TableLine

$qe5Note

## Required Follow-up Before Tensile Production

DFTpy/LDA + SM reproduces the QE-scale vacancy energy, but its fully relaxed
pristine cell contracts from 12.119544 A to 11.694307 A:

- axial length change: -3.509%
- volume change: -10.161%

Validate the DFTpy/LDA + SM bulk equilibrium lattice constant before using SM
for production nanocrystal tensile calculations.
"@

$readme | Set-Content -LiteralPath (Join-Path $Destination "00_START_HERE\README_FIRST.md") -Encoding UTF8

$pptBrief = @"
# Concise PPT Update Brief

Use five slides only.

## 1. Corrected Al Vacancy Benchmark

- conventional cubic fcc 3x3x3 centered vacancy
- pristine / vacancy: 108 -> 107 atoms
- vacancy concentration: 1/108 = 0.925926%
- VESTA checked

## 2. Corrected Relaxation Workflow

- QE: PBE PAW, literal vc-relax
- DFTpy: LDA, al.lda.recpot
- DFTpy: full atom+cell relaxation with FrechetCellFilter + BFGS
- DFTpy target: final fmax < 0.002 eV/A

## 3. QE/PBE vc-relax Reference

Cutoff scan at k=3x3x3:

| ecut | Ef_vac |
|---:|---:|
| 400 eV | 0.644719 eV |
| 600 eV | 0.644901 eV |
| 800 eV | 0.644942 eV |

Completed k-mesh points at 800 eV:

| k-mesh | Ef_vac | status |
|---|---:|---|
| 2x2x2 | 0.989725 eV | completed |
| 3x3x3 | 0.644942 eV | completed |
| 4x4x4 | 0.677875 eV | completed |
$qe5TableLine

$qe5Note

## 4. DFTpy Vacancy Energy Is KEDF-Dependent

All rows use the same LDA pseudo, same 3x3x3 cell, and same full relaxation.

| KEDF | Ef_vac at spacing=0.20 A |
|---|---:|
| TFVW | 3.280521 eV |
| WT | 1.333804 eV |
| SM | 0.682758 eV |

Use: `01_PROCESSED_RESULTS/dftpy_kedf_spacing_comparison.png`

## 5. Conclusion And Next Step

- TFVW overestimates vacancy energy.
- SM is close to the completed QE range: 0.645-0.678 eV.
- The anomaly is KEDF-dependent, not a spacing or missing-relaxation failure.
- Before production tensile runs, validate the DFTpy/LDA + SM bulk equilibrium
  lattice constant because the relaxed pristine cell volume changes by -10.161%.
- After advisor signoff, resume axially periodic nanocrystal tensile tests:
  z-periodic, z > 10 A, inner/middle/outer vacancy positions, and vacancy concentration recorded.
"@

$pptBrief | Set-Content -LiteralPath (Join-Path $Destination "07_PRESENTATION\PPT_UPDATE_BRIEF.md") -Encoding UTF8

python (Join-Path $PSScriptRoot "audit_final_nus_vacancy_structures.py") --rootdir $Destination
if ($LASTEXITCODE -ne 0) {
    throw "Structure audit failed."
}

$files = Get-ChildItem -LiteralPath $Destination -Recurse -File | ForEach-Object {
    [PSCustomObject]@{
        relative_path = $_.FullName.Substring($Destination.Length + 1)
        size_bytes = $_.Length
        modified = $_.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
    }
}
$files | Export-Csv -LiteralPath (Join-Path $Destination "00_START_HERE\FILE_INDEX.csv") -NoTypeInformation -Encoding UTF8

$inventory = Get-ChildItem -LiteralPath $Destination -Directory | ForEach-Object {
    $dirFiles = Get-ChildItem -LiteralPath $_.FullName -Recurse -File
    [PSCustomObject]@{
        folder = $_.Name
        file_count = $dirFiles.Count
        size_mb = [math]::Round(($dirFiles | Measure-Object Length -Sum).Sum / 1MB, 3)
    }
}
$inventory | Export-Csv -LiteralPath (Join-Path $Destination "00_START_HERE\FOLDER_INVENTORY.csv") -NoTypeInformation -Encoding UTF8

$zip = "$Destination.zip"
if (Test-Path -LiteralPath $zip) {
    Remove-Item -LiteralPath $zip
}
Compress-Archive -LiteralPath $Destination -DestinationPath $zip -CompressionLevel Optimal

Write-Host "============================================================"
Write-Host "Final NUS vacancy database built"
Write-Host "============================================================"
Write-Host "[FOLDER] $Destination"
Write-Host "[ZIP   ] $zip"
