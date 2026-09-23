param(
    [string]$HandoffRoot = 'C:\Users\dawso\Desktop\AL_DEFECTS_USB_HANDOFF_20260717'
)

$ErrorActionPreference = 'Stop'

$workspace = 'C:\Users\dawso\nano_tensile_TFvW'
$desktop = 'C:\Users\dawso\Desktop'
$nasPackage = Join-Path $desktop 'DFTPY_QE_VACLM_NAS_20260701'
$iservicePackage = Join-Path $desktop 'DFTPY_VACLM_PROF_DELIVERY_20260629_ISERVICE_ONLY'
$copyLogRoot = Join-Path $HandoffRoot '06_MANIFEST_AND_CHECKSUMS\COPY_LOGS'
$script:copyIndex = 0

function Copy-Tree {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination,
        [string[]]$ExcludedDirectories = @()
    )

    if (-not (Test-Path -LiteralPath $Source)) {
        throw "Required source does not exist: $Source"
    }

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    $script:copyIndex += 1
    New-Item -ItemType Directory -Force -Path $copyLogRoot | Out-Null
    $leaf = Split-Path -Leaf $Destination
    $logPath = Join-Path $copyLogRoot ('{0:D2}_{1}.log' -f $script:copyIndex, $leaf)
    $robocopyArgs = @(
        $Source,
        $Destination,
        '/E',
        '/COPY:DAT',
        '/DCOPY:DAT',
        '/XJ',
        '/R:2',
        '/W:2',
        '/MT:8',
        '/NFL',
        '/NDL',
        '/NP',
        "/LOG+:$logPath"
    )

    if ($ExcludedDirectories.Count -gt 0) {
        $robocopyArgs += '/XD'
        $robocopyArgs += $ExcludedDirectories
    }

    Write-Host "[COPY] $Source"
    Write-Host "    -> $Destination"
    & robocopy @robocopyArgs
    if ($LASTEXITCODE -gt 7) {
        throw "robocopy failed with exit code $LASTEXITCODE for $Source"
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $HandoffRoot '00_START_HERE.md'))) {
    throw "Handoff scaffold is missing: $HandoffRoot"
}

# Preserve the current Git working tree, but keep generated run directories in
# the explicit dataset folders below so the project layout is unambiguous.
Copy-Tree `
    -Source $workspace `
    -Destination (Join-Path $HandoffRoot '01_CODE_AND_REPRODUCIBILITY\CURRENT_SOURCE_TREE') `
    -ExcludedDirectories @('.git', 'cases', 'results', 'outputs', 'tmp', 'iservice_packages', '.vscode', '__pycache__', '.mplconfig')

# The working tree excludes generated outputs, but a small curated subset is
# tracked by Git and must accompany the portable repository to avoid an
# accidental deletion state after the USB copy.
$codeDestination = Join-Path $HandoffRoot '01_CODE_AND_REPRODUCIBILITY\CURRENT_SOURCE_TREE'
$trackedOutputFiles = & git -C $workspace ls-files -- 'outputs'
foreach ($relativePath in $trackedOutputFiles) {
    $sourceFile = Join-Path $workspace $relativePath
    $destinationFile = Join-Path $codeDestination $relativePath
    if (-not (Test-Path -LiteralPath $sourceFile)) {
        throw "Tracked output is missing from source workspace: $relativePath"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destinationFile) | Out-Null
    Copy-Item -LiteralPath $sourceFile -Destination $destinationFile -Force
}

# This is the canonical iService lambda-mu package. It intentionally includes
# both the curated cases and the pulled remote-provenance stage.
Copy-Tree `
    -Source $iservicePackage `
    -Destination (Join-Path $HandoffRoot '02_ACTIVE_DFTPY_ISERVICE_LAMMU_RAW')

Copy-Tree `
    -Source (Join-Path $nasPackage '02_QE_vcrelax') `
    -Destination (Join-Path $HandoffRoot '03_ACTIVE_QE_VCRELAX_REFERENCE')

$analysisRoot = Join-Path $HandoffRoot '04_ANALYSIS_AND_PRESENTATION'
Copy-Tree -Source (Join-Path $nasPackage '03_compare') -Destination (Join-Path $analysisRoot '03_compare')
Copy-Tree -Source (Join-Path $nasPackage '04_slides') -Destination (Join-Path $analysisRoot '04_slides')
foreach ($fileName in @(
    'README_PACKAGE.md',
    'ADVISOR_POINT_BY_POINT_REPLY_20260713.md',
    'EMAIL_REPLY_DRAFT_20260713.txt'
)) {
    $sourceFile = Join-Path $nasPackage $fileName
    if (Test-Path -LiteralPath $sourceFile) {
        Copy-Item -LiteralPath $sourceFile -Destination (Join-Path $analysisRoot $fileName) -Force
    }
}

$legacyRoot = Join-Path $HandoffRoot '05_LEGACY_SUPPORTING_CALCULATIONS'
$legacyItems = @(
    [PSCustomObject]@{ Name = '01_TFVW_WEIGHT_SCAN_20260605'; Source = (Join-Path $desktop 'DFTPY_TFVW_WEIGHT_SCAN_20260605') },
    [PSCustomObject]@{ Name = '02_INITIAL_ISERVICE_LAMMU_96PT_20260610'; Source = (Join-Path $desktop 'DFTPY_TFVW_LAMBDA_MU_VACANCY_MATRIX_20260610') },
    [PSCustomObject]@{ Name = '03_DIVACANCY_RSCAN_20260616'; Source = (Join-Path $desktop 'DFTPY_DIVACANCY_RSCAN_20260616') },
    [PSCustomObject]@{ Name = '04_DIVACANCY_PILOT_REBUILT_20260622'; Source = (Join-Path $desktop 'DFTPY_DIVACANCY_PILOT_REBUILT_20260622') },
    [PSCustomObject]@{ Name = '05_LOCAL_OFFICIAL_RELAX_VACLM_10X10_20260629'; Source = (Join-Path $desktop 'LOCAL_DFTPY_OFFICIAL_RELAX_VACLM_10X10_20260629') },
    [PSCustomObject]@{ Name = '06_LOCAL_LBFGS_GUARDED_RERUN_20260626'; Source = (Join-Path $desktop 'LOCAL_DFTPY_VACLM_10X10_FULL_RERUN_LBFGS_GUARDED_20260626') },
    [PSCustomObject]@{ Name = '07_PROFESS_KEDF_SCREENING_20260604'; Source = (Join-Path $desktop 'LOCAL_PROFESS_KEDF_SWEEP_20260604') },
    [PSCustomObject]@{ Name = '08_PROFESS_LAMMU_CELL_RELAX_20260609'; Source = (Join-Path $desktop 'LOCAL_PROFESS_TFVW_LAMBDA_MU_CELL_RELAX_20260609') },
    [PSCustomObject]@{ Name = '09_QE_CONVENTIONAL_PULL_20260522'; Source = (Join-Path $desktop 'qe_conventional_pull_20260522') }
)

foreach ($item in $legacyItems) {
    Copy-Tree -Source $item.Source -Destination (Join-Path $legacyRoot $item.Name)
}

$manifestRoot = Join-Path $HandoffRoot '06_MANIFEST_AND_CHECKSUMS'
$topLevel = Get-ChildItem -LiteralPath $HandoffRoot -Directory | Sort-Object Name
$inventory = foreach ($directory in $topLevel) {
    $files = Get-ChildItem -LiteralPath $directory.FullName -Recurse -File -Force -ErrorAction SilentlyContinue
    $bytes = ($files | Measure-Object -Property Length -Sum).Sum
    [PSCustomObject]@{
        Directory = $directory.Name
        FileCount = $files.Count
        SizeBytes = $bytes
        SizeGiB = [math]::Round($bytes / 1GB, 3)
    }
}
$inventory | Export-Csv -LiteralPath (Join-Path $manifestRoot 'USB_HANDOFF_DIRECTORY_INVENTORY.csv') -NoTypeInformation -Encoding utf8

$keyFiles = @(
    (Join-Path $HandoffRoot '00_START_HERE.md'),
    (Join-Path $analysisRoot 'README_PACKAGE.md'),
    (Join-Path $analysisRoot '04_slides\DFTpy_QE_VACLM_minimal_discussion_20260701.pptx'),
    (Join-Path $analysisRoot '03_compare\qe_reference_summary.csv')
) | Where-Object { Test-Path -LiteralPath $_ }

$hashes = foreach ($file in $keyFiles) {
    $hash = Get-FileHash -Algorithm SHA256 -LiteralPath $file
    [PSCustomObject]@{
        RelativePath = $file.Substring($HandoffRoot.Length + 1)
        SHA256 = $hash.Hash
        SizeBytes = (Get-Item -LiteralPath $file).Length
    }
}
$hashes | Export-Csv -LiteralPath (Join-Path $manifestRoot 'KEY_FILE_SHA256.csv') -NoTypeInformation -Encoding utf8

Write-Host '[DONE] USB handoff prepared:'
Write-Host $HandoffRoot
$inventory | Format-Table -AutoSize
