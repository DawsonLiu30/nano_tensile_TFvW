param(
    [string]$RunRoot = "C:\Users\dawso\Desktop\LOCAL_DFTPY_VACLM_10X10_FULL_RERUN_LBFGS_GUARDED_20260626",
    [string]$MapsRoot = "C:\Users\dawso\Desktop\LOCAL_DFTPY_VACLM_SIMPLE_MAPS_20260629",
    [string]$OutRoot = "C:\Users\dawso\Desktop\DFTPY_VACLM_PROF_DELIVERY_20260629"
)

$ErrorActionPreference = "Stop"

function Copy-OneFile {
    param(
        [string]$Source,
        [string]$Destination
    )
    if (Test-Path -LiteralPath $Source) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
        Copy-Item -LiteralPath $Source -Destination $Destination -Force
    }
}

function Copy-Dir {
    param(
        [string]$Source,
        [string]$Destination
    )
    if (Test-Path -LiteralPath $Source) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
        Copy-Item -LiteralPath $Source -Destination $Destination -Recurse -Force
    }
}

function Copy-ScriptAsText {
    param(
        [string]$Source,
        [string]$DestinationDirectory
    )
    if (Test-Path -LiteralPath $Source) {
        New-Item -ItemType Directory -Force -Path $DestinationDirectory | Out-Null
        $name = Split-Path -Leaf $Source
        Copy-Item -LiteralPath $Source -Destination (Join-Path $DestinationDirectory "$name.txt") -Force
    }
}

$Full = Join-Path $OutRoot "DFTPY_VACLM_PROF_DELIVERY_20260629_FULL"
$Light = Join-Path $OutRoot "DFTPY_VACLM_PROF_DELIVERY_20260629_LIGHT_EMAIL_SAFE"

foreach ($target in @($Full, $Light)) {
    if (Test-Path -LiteralPath $target) {
        Remove-Item -LiteralPath $target -Recurse -Force
    }
}

New-Item -ItemType Directory -Force -Path $Full, $Light | Out-Null

$Readme = @'
# DFTpy TFvW lambda-mu scan package for professor

## Purpose

This package documents the local DFTpy TFvW lambda-mu scan for conventional
fcc Al 3x3x3 single-vacancy calibration.

The professor-facing raw table intentionally contains only the three requested
raw-output quantities:

1. Total energy from the final `pristine_dftpy.out` summary.
2. KEDF / kinetic energy from the final `pristine_dftpy.out` summary.
3. Relaxed pristine lattice constant, computed as `mean(|a|, |b|, |c|) / 3`.

The figure package separately plots:

1. relaxed pristine lattice constant `a0`
2. vacancy formation energy using Gillan-style supercell formula
3. pristine KEDF / kinetic energy

## Reference lines in the figures

- Al lattice constant reference: 4.05 A.
  WebElements lists fcc Al cell parameters a=b=c=404.95 pm.
- Gillan 1989 reports calculated Al vacancy energy 0.56 eV and experimental
  value 0.66 eV in the abstract/discussion of "Calculation of the vacancy
  formation energy in aluminium".

## Vacancy formation energy formula

For the single-vacancy map, the formation energy is computed using the
Gillan-style perfect/defective supercell formula:

```text
E_f^vac = E_vac(Al107) - (107/108) E_pristine(Al108)
```

## Important status note

This local rerun attempted all 100 lambda-mu points.

- 49 points produced complete pristine raw values for the three requested
  professor-table quantities.
- 36 points completed both pristine and vacancy calculations and therefore
  have vacancy formation energies.
- The remaining low-lambda points were stopped by the local guard because the
  relaxation became pathological, usually with very large forces or unstable
  cell behavior.

Blank/FAILED regions are intentionally not interpolated.

## Directory layout

- `01_PROFESSOR_RAW_TABLE`: Excel/CSV table requested by professor.
- `02_SIMPLE_MAPS`: simple three-map figure and plot data.
- `03_RAW_CASES`: complete raw 100 case folders with input/output/result files.
- `04_RUN_METADATA`: settings, process metadata, stdout/stderr, supervisor logs.
- `05_SCRIPTS_USED`: scripts used to run, collect, tabulate, and plot results.
- `06_REFERENCE_NOTES`: compact reference notes and source URLs.

## Source roots

```text
Run root:
__RUN_ROOT__

Maps root:
__MAPS_ROOT__
```
'@
$Readme = $Readme.Replace("__RUN_ROOT__", $RunRoot).Replace("__MAPS_ROOT__", $MapsRoot)

$ReferenceNotes = @'
# Reference notes

## Lattice constant

Reference value used in plot:

```text
a0(Al) = 4.05 A
```

Source:

```text
WebElements Aluminium crystal structure:
https://www.webelements.com/aluminium/crystal_structure.html

Listed cell parameters:
a = b = c = 404.95 pm = 4.0495 A
```

## Vacancy formation energy

Reference values used in plot:

```text
Gillan calculated value: 0.56 eV
experimental value cited by Gillan: 0.66 eV
```

Local PDF checked:

```text
C:\Users\dawso\Downloads\M_J_Gillan_1989_J._Phys.%3A_Condens._Matter_1_689 (1).pdf
```

Extracted meaning:

Gillan's abstract/discussion reports calculated Al vacancy energy 0.56 eV and
experimental value 0.66 eV.
'@

foreach ($root in @($Full, $Light)) {
    Set-Content -LiteralPath (Join-Path $root "README_PACKAGE.md") -Value $Readme -Encoding UTF8
    New-Item -ItemType Directory -Force -Path (Join-Path $root "06_REFERENCE_NOTES") | Out-Null
    Set-Content -LiteralPath (Join-Path $root "06_REFERENCE_NOTES\REFERENCE_VALUES.md") -Value $ReferenceNotes -Encoding UTF8
}

# Common lightweight deliverables.
Copy-Dir (Join-Path $RunRoot "10_professor_raw_log_table") (Join-Path $Full "01_PROFESSOR_RAW_TABLE")
Copy-Dir (Join-Path $MapsRoot ".") (Join-Path $Full "02_SIMPLE_MAPS")
Copy-Dir (Join-Path $RunRoot "10_professor_raw_log_table") (Join-Path $Light "01_PROFESSOR_RAW_TABLE")
Copy-Dir (Join-Path $MapsRoot ".") (Join-Path $Light "02_SIMPLE_MAPS")

# Full raw input/output cases.
Copy-Dir (Join-Path $RunRoot "03_runs") (Join-Path $Full "03_RAW_CASES\03_runs")

# Settings and metadata.
foreach ($root in @($Full, $Light)) {
    Copy-Dir (Join-Path $RunRoot "01_settings") (Join-Path $root "04_RUN_METADATA\01_settings")
    foreach ($f in @(
        "BACKGROUND_PROCESS.json",
        "SUPERVISOR_PROCESS.json",
        "LOCAL_RERUN_README.md",
        "local_rerun_progress.json",
        "local_rerun.stdout.log",
        "local_rerun.stderr.log",
        "local_rerun_supervisor.log",
        "supervisor.stdout.log",
        "supervisor.stderr.log"
    )) {
        Copy-OneFile (Join-Path $RunRoot $f) (Join-Path $root "04_RUN_METADATA\$f")
    }
}

# Scripts used. Full keeps executable extensions; light stores text copies.
$ScriptSources = @(
    "C:\Users\dawso\nano_tensile_TFvW\scripts\run_local_dftpy_vaclm_10x10_rerun.py",
    "C:\Users\dawso\nano_tensile_TFvW\scripts\start_local_dftpy_vaclm_10x10_rerun_20260626.ps1",
    "C:\Users\dawso\nano_tensile_TFvW\scripts\supervise_local_dftpy_vaclm_lbfgs_guarded_20260626.ps1",
    "C:\Users\dawso\nano_tensile_TFvW\scripts\benchmark_local_dftpy_vaclm_optimizers_20260626.ps1",
    "C:\Users\dawso\nano_tensile_TFvW\scripts\make_professor_raw_log_table_lambda_mu.py",
    "C:\Users\dawso\nano_tensile_TFvW\scripts\plot_local_vaclm_professor_simple_maps.py",
    "C:\Users\dawso\nano_tensile_TFvW\app\dft_engine.py"
)

foreach ($src in $ScriptSources) {
    Copy-OneFile $src (Join-Path $Full ("05_SCRIPTS_USED\" + (Split-Path -Leaf $src)))
    Copy-ScriptAsText $src (Join-Path $Light "05_SCRIPTS_USED_TEXT")
}

# Case inventory for quick review.
$Inventory = Join-Path $Full "03_RAW_CASES\CASE_INVENTORY.csv"
$rows = Get-ChildItem -LiteralPath (Join-Path $Full "03_RAW_CASES\03_runs") -Directory | ForEach-Object {
    [PSCustomObject]@{
        case = $_.Name
        has_pristine_input = Test-Path -LiteralPath (Join-Path $_.FullName "dftpy_pristine_input.ini")
        has_vacancy_input = Test-Path -LiteralPath (Join-Path $_.FullName "dftpy_vacancy_input.ini")
        has_pristine_output = Test-Path -LiteralPath (Join-Path $_.FullName "pristine_dftpy.out")
        has_vacancy_output = Test-Path -LiteralPath (Join-Path $_.FullName "vacancy_dftpy.out")
        has_result_json = Test-Path -LiteralPath (Join-Path $_.FullName "result.json")
        has_failure_marker = Test-Path -LiteralPath (Join-Path $_.FullName "LOCAL_RERUN_FAILED.txt")
    }
}
$rows | Export-Csv -LiteralPath $Inventory -NoTypeInformation -Encoding UTF8
Copy-OneFile $Inventory (Join-Path $Light "03_CASE_INVENTORY\CASE_INVENTORY.csv")

# Zip packages.
$FullZip = Join-Path $OutRoot "DFTPY_VACLM_PROF_DELIVERY_20260629_FULL.zip"
$LightZip = Join-Path $OutRoot "DFTPY_VACLM_PROF_DELIVERY_20260629_LIGHT_EMAIL_SAFE.zip"
if (Test-Path -LiteralPath $FullZip) { Remove-Item -LiteralPath $FullZip -Force }
if (Test-Path -LiteralPath $LightZip) { Remove-Item -LiteralPath $LightZip -Force }
Compress-Archive -Path (Join-Path $Full "*") -DestinationPath $FullZip
Compress-Archive -Path (Join-Path $Light "*") -DestinationPath $LightZip

$FullSize = [math]::Round((Get-ChildItem -LiteralPath $Full -Recurse -File | Measure-Object Length -Sum).Sum / 1MB, 2)
$LightSize = [math]::Round((Get-ChildItem -LiteralPath $Light -Recurse -File | Measure-Object Length -Sum).Sum / 1MB, 2)

[PSCustomObject]@{
    FullFolder = $Full
    FullFolderMB = $FullSize
    FullZip = $FullZip
    FullZipMB = [math]::Round((Get-Item -LiteralPath $FullZip).Length / 1MB, 2)
    LightFolder = $Light
    LightFolderMB = $LightSize
    LightZip = $LightZip
    LightZipMB = [math]::Round((Get-Item -LiteralPath $LightZip).Length / 1MB, 2)
} | ConvertTo-Json -Depth 3
