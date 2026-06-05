param(
    [string]$SourceRoot = "C:\Users\dawso\Desktop\FINAL_NUS_VACANCY_DATABASE_20260601",
    [string]$OutRoot = "C:\Users\dawso\Desktop\EMAIL_TO_PROF_VACANCY_DATA_20260603",
    [string]$TarGzPath = "C:\Users\dawso\Downloads\delivery_DFTpy_LDA_vacancy_dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_20260529.tar.gz"
)

$ErrorActionPreference = "Stop"

function Assert-Exists($Path, $Label) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label not found: $Path"
    }
}

$repo = Split-Path -Parent $PSScriptRoot
$zipPath = "$OutRoot.zip"
$allowedDesktopRoots = @(
    [Environment]::GetFolderPath("Desktop"),
    "C:\Users\dawso\Desktop"
) | Select-Object -Unique

Assert-Exists $SourceRoot "Source data package"
Assert-Exists $TarGzPath "Advisor tar.gz"

$outParent = (Resolve-Path -LiteralPath (Split-Path -Parent $OutRoot)).Path
$allowedParents = $allowedDesktopRoots | ForEach-Object {
    if (Test-Path -LiteralPath $_) {
        (Resolve-Path -LiteralPath $_).Path
    }
}
if ($outParent -notin $allowedParents) {
    throw "OutRoot must be directly under Desktop for safety: $OutRoot"
}

if (Test-Path -LiteralPath $OutRoot) {
    Remove-Item -LiteralPath $OutRoot -Recurse -Force
}
if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

New-Item -ItemType Directory -Path $OutRoot | Out-Null

Write-Host "============================================================"
Write-Host "Build email-only data package"
Write-Host "============================================================"
Write-Host "[SOURCE] $SourceRoot"
Write-Host "[OUT   ] $OutRoot"

$excludedTopDirs = @("07_PRESENTATION")
$excludedExtensions = @(".ppt", ".pptx")

Get-ChildItem -LiteralPath $SourceRoot -Force | ForEach-Object {
    if ($_.PSIsContainer -and ($excludedTopDirs -contains $_.Name)) {
        Write-Host "[SKIP  ] $($_.Name)"
        return
    }
    $dest = Join-Path $OutRoot $_.Name
    if ($_.PSIsContainer) {
        Copy-Item -LiteralPath $_.FullName -Destination $dest -Recurse -Force
    } elseif ($excludedExtensions -notcontains $_.Extension.ToLowerInvariant()) {
        Copy-Item -LiteralPath $_.FullName -Destination $dest -Force
    }
}

Get-ChildItem -LiteralPath $OutRoot -Recurse -File | Where-Object {
    $excludedExtensions -contains $_.Extension.ToLowerInvariant()
} | Remove-Item -Force

$tableDir = Join-Path $OutRoot "08_PROFESSOR_TFVW_TARGZ_VASP_TABLE"
New-Item -ItemType Directory -Path $tableDir | Out-Null
Copy-Item -LiteralPath $TarGzPath -Destination (Join-Path $tableDir (Split-Path -Leaf $TarGzPath)) -Force

$python = "python"
& $python (Join-Path $repo "scripts\summarize_targz_vasp_energy_table.py") `
    --tar $TarGzPath `
    --outdir $tableDir
if ($LASTEXITCODE -ne 0) {
    throw "VASP table generation failed with exit code $LASTEXITCODE"
}

$readme = Join-Path $OutRoot "README_EMAIL_PACKAGE.md"
@"
# Email data package for vacancy benchmark

Generated: 2026-06-03

This package is a data-only copy for email delivery. PowerPoint files and the presentation folder were intentionally excluded.

Top-level contents:
- `00_START_HERE`: short guide / package index.
- `01_PROCESSED_RESULTS`: compact CSV summaries and plots.
- `02_QE_PBE_VCRELAX_3x3x3_RAW`: QE input/output data.
- `03_DFTPY_LDA_TFVW_RAW`: DFTpy LDA + TFvW input/output data.
- `04_DFTPY_LDA_WT_RAW`: DFTpy LDA + WT input/output data.
- `05_DFTPY_LDA_SM_RAW`: DFTpy LDA + SM input/output data.
- `06_REPRODUCIBILITY`: scripts/pseudopotentials used for rerun/reproducibility.
- `08_PROFESSOR_TFVW_TARGZ_VASP_TABLE`: table requested for the previously sent TFvW tar.gz.

Professor-request table:
- `08_PROFESSOR_TFVW_TARGZ_VASP_TABLE/vasp_file_total_energy_table.csv`
- `08_PROFESSOR_TFVW_TARGZ_VASP_TABLE/vasp_file_total_energy_table.md`

Key compact result:
- QE/PBE vc-relax 3x3x3 k=3-5 at 800 eV: about 0.639-0.678 eV.
- DFTpy/LDA + SM: about 0.682-0.684 eV.
- DFTpy/LDA + WT: about 1.334-1.339 eV.
- DFTpy/LDA + TFvW: about 3.280-3.282 eV.

"@ | Set-Content -LiteralPath $readme -Encoding UTF8

Compress-Archive -Path (Join-Path $OutRoot "*") -DestinationPath $zipPath -CompressionLevel Optimal -Force

$sha = (Get-FileHash -Algorithm SHA256 -LiteralPath $zipPath).Hash
$sizeMB = [Math]::Round((Get-Item -LiteralPath $zipPath).Length / 1MB, 2)

Write-Host ""
Write-Host "============================================================"
Write-Host "Email data package built"
Write-Host "============================================================"
Write-Host "[FOLDER] $OutRoot"
Write-Host "[ZIP   ] $zipPath"
Write-Host "[SIZE  ] $sizeMB MB"
Write-Host "[SHA256] $sha"
