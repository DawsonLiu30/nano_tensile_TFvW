param(
    [string]$RemoteHost = "dawson666@twnia3.nchc.org.tw",
    [string]$RemoteTfvwRoot = "/work/dawson666/dftpy_project/relax/TFvW_test",
    [string]$RemoteCodeRoot = "/work/dawson666/dftpy_project/relax/OLD_dftpy45_column_scan",
    [string]$LocalDelivery = "C:\Users\dawso\Desktop\DFTPY_VACLM_PROF_DELIVERY_20260629_ISERVICE_ONLY"
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "Pull iService VACLM full provenance"
Write-Host "============================================================"
Write-Host "[REMOTE_HOST] $RemoteHost"
Write-Host "[REMOTE_TFVW] $RemoteTfvwRoot"
Write-Host "[REMOTE_CODE] $RemoteCodeRoot"
Write-Host "[LOCAL     ] $LocalDelivery"
Write-Host ""
Write-Host "This script creates one tar.gz on iService, downloads it, and"
Write-Host "merges missing provenance into the existing delivery package."
Write-Host "It is expected to ask for NCHC 2FA/password."
Write-Host "============================================================"

$remoteTar = "/work/dawson666/dftpy_project/relax/vaclm_full_provenance_20260629.tar.gz"
$remoteStage = "/work/dawson666/dftpy_project/relax/vaclm_full_provenance_stage_20260629"
$localTar = Join-Path $env:TEMP "vaclm_full_provenance_20260629.tar.gz"
$localStage = Join-Path $LocalDelivery "_REMOTE_PROVENANCE_STAGE_20260629"

New-Item -ItemType Directory -Force -Path $LocalDelivery | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $LocalDelivery "03_RAW_CASES") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $LocalDelivery "04_RUN_METADATA") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $LocalDelivery "05_SCRIPTS_USED") | Out-Null

Write-Host ""
Write-Host "[1/5] Build remote provenance tarball"

$remoteScript = @"
set -e
echo HOST=`$(hostname)
test -d '$RemoteTfvwRoot'
test -d '$RemoteTfvwRoot/03_runs'
echo RUN_DIRS=`$(find '$RemoteTfvwRoot/03_runs' -mindepth 1 -maxdepth 1 -type d | wc -l)
echo RESULT_JSON=`$(find '$RemoteTfvwRoot/03_runs' -name result.json | wc -l)
echo TRAJ=`$(find '$RemoteTfvwRoot/03_runs' -name '*.traj' | wc -l)
echo RELAX_LOGS=`$(find '$RemoteTfvwRoot/03_runs' \( -name 'pristine_relax.log' -o -name 'vacancy_relax.log' \) | wc -l)
rm -rf '$remoteStage'
mkdir -p '$remoteStage/TFvW_test'
mkdir -p '$remoteStage/code/scripts'
mkdir -p '$remoteStage/code/app'
for item in 03_runs 10_professor_raw_log_table 00_README 00_matrix_index 01_settings 05_tables 06_submission_scripts 07_audit 08_analysis_gillan_style 09_professor_table; do
  if [ -e '$RemoteTfvwRoot/'"`$item" ]; then
    cp -a '$RemoteTfvwRoot/'"`$item" '$remoteStage/TFvW_test/'
  else
    echo '[WARN] missing TFvW_test/'"`$item"
  fi
done
if [ -f '$RemoteCodeRoot/scripts/run_dftpy_vcrelax_vacancy_matrix_one.py' ]; then
  cp -a '$RemoteCodeRoot/scripts/run_dftpy_vcrelax_vacancy_matrix_one.py' '$remoteStage/code/scripts/'
else
  echo '[WARN] missing production runner'
fi
if [ -f '$RemoteCodeRoot/app/dft_engine.py' ]; then
  cp -a '$RemoteCodeRoot/app/dft_engine.py' '$remoteStage/code/app/'
else
  echo '[WARN] missing dft_engine.py'
fi
rm -f '$remoteTar'
tar -czf '$remoteTar' -C '$remoteStage' .
ls -lh '$remoteTar'
"@

ssh $RemoteHost $remoteScript

Write-Host ""
Write-Host "[2/5] Download remote tarball"
if (Test-Path $localTar) {
    Remove-Item -LiteralPath $localTar -Force
}
# NCHC login nodes may reject the SFTP subsystem used by modern OpenSSH scp.
# Force legacy SCP protocol so the tarball download works after 2FA login.
scp -O "${RemoteHost}:${remoteTar}" $localTar
Get-Item -LiteralPath $localTar | Format-List FullName, Length, LastWriteTime

Write-Host ""
Write-Host "[3/5] Extract locally"
if (Test-Path $localStage) {
    Remove-Item -LiteralPath $localStage -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $localStage | Out-Null
tar -xzf $localTar -C $localStage

Write-Host ""
Write-Host "[4/5] Merge into delivery package"
$tfvwStage = Join-Path $localStage "TFvW_test"
$codeStage = Join-Path $localStage "code"

function Copy-DirContents($Source, $Dest) {
    if (Test-Path $Source) {
        New-Item -ItemType Directory -Force -Path $Dest | Out-Null
        Copy-Item -LiteralPath (Join-Path $Source "*") -Destination $Dest -Recurse -Force
        Write-Host "[COPY] $Source -> $Dest"
    } else {
        Write-Host "[SKIP] missing $Source"
    }
}

Copy-DirContents (Join-Path $tfvwStage "03_runs") (Join-Path $LocalDelivery "03_RAW_CASES\03_runs")
Copy-DirContents (Join-Path $tfvwStage "10_professor_raw_log_table") (Join-Path $LocalDelivery "01_RAW_TABLE")

foreach ($sub in @("00_README", "00_matrix_index", "01_settings", "05_tables", "07_audit", "08_analysis_gillan_style", "09_professor_table")) {
    Copy-DirContents (Join-Path $tfvwStage $sub) (Join-Path $LocalDelivery "04_RUN_METADATA\$sub")
}

Copy-DirContents (Join-Path $tfvwStage "06_submission_scripts") (Join-Path $LocalDelivery "05_SCRIPTS_USED\06_submission_scripts")
Copy-DirContents (Join-Path $codeStage "scripts") (Join-Path $LocalDelivery "05_SCRIPTS_USED\scripts")
Copy-DirContents (Join-Path $codeStage "app") (Join-Path $LocalDelivery "05_SCRIPTS_USED\app")

Write-Host ""
Write-Host "[5/5] Local audit"
$runs = Join-Path $LocalDelivery "03_RAW_CASES\03_runs"
$metadata = Join-Path $LocalDelivery "04_RUN_METADATA"
New-Item -ItemType Directory -Force -Path $metadata | Out-Null
$audit = Join-Path $metadata "PROVENANCE_PULL_AUDIT_20260629.txt"

$caseDirs = @(Get-ChildItem -LiteralPath $runs -Directory -ErrorAction SilentlyContinue).Count
$resultJson = @(Get-ChildItem -LiteralPath $runs -Recurse -Filter "result.json" -File -ErrorAction SilentlyContinue).Count
$traj = @(Get-ChildItem -LiteralPath $runs -Recurse -Filter "*.traj" -File -ErrorAction SilentlyContinue).Count
$relaxLogs = @(Get-ChildItem -LiteralPath $runs -Recurse -Include "pristine_relax.log","vacancy_relax.log" -File -ErrorAction SilentlyContinue).Count
$inputs = @(Get-ChildItem -LiteralPath $runs -Recurse -Include "dftpy_pristine_input.ini","dftpy_vacancy_input.ini" -File -ErrorAction SilentlyContinue).Count
$submitSnapshots = @(Get-ChildItem -LiteralPath $runs -Recurse -Filter "submit_script_used_*" -File -ErrorAction SilentlyContinue).Count
$enginePath = Join-Path $LocalDelivery "05_SCRIPTS_USED\app\dft_engine.py"
$runnerPath = Join-Path $LocalDelivery "05_SCRIPTS_USED\scripts\run_dftpy_vcrelax_vacancy_matrix_one.py"

$auditText = @"
iService VACLM full provenance pull audit
Generated: $(Get-Date -Format s)

REMOTE_HOST=$RemoteHost
REMOTE_TFVW_ROOT=$RemoteTfvwRoot
REMOTE_CODE_ROOT=$RemoteCodeRoot
LOCAL_DELIVERY=$LocalDelivery

Local raw case folders: $caseDirs
Local result.json: $resultJson
Local .traj files: $traj
Local relax logs: $relaxLogs
Local DFTpy ini inputs: $inputs
Local submit snapshots: $submitSnapshots

Code provenance:
runner: $runnerPath
runner_exists: $(Test-Path $runnerPath)
engine: $enginePath
engine_exists: $(Test-Path $enginePath)

Sample trajectory files:
$((Get-ChildItem -LiteralPath $runs -Recurse -Filter "*.traj" -File -ErrorAction SilentlyContinue | Select-Object -First 20 -ExpandProperty FullName) -join "`n")
"@

$auditText | Set-Content -LiteralPath $audit -Encoding UTF8
Get-Content -LiteralPath $audit

$readme = Join-Path $LocalDelivery "README_PROVENANCE_UPDATE_20260629.md"
$readmeText = @"
# Provenance Update 2026-06-29

This delivery package was updated by pulling the cleaned iService TFvW lambda-mu
scan provenance from:

````text
$RemoteTfvwRoot
````

The update is intended to add the missing trajectory and production provenance
requested during review:

- complete raw case folders under ``03_RAW_CASES/03_runs``
- DFTpy input files
- DFTpy output summaries
- relaxation logs
- ASE trajectory files, if present on iService
- final relaxed VASP/XYZ structures
- result.json files
- submission scripts
- Python runner and ``app/dft_engine.py``

Local audit:

````text
04_RUN_METADATA/PROVENANCE_PULL_AUDIT_20260629.txt
````
"@
$readmeText | Set-Content -LiteralPath $readme -Encoding UTF8

Write-Host ""
Write-Host "============================================================"
Write-Host "Done."
Write-Host "[AUDIT ] $audit"
Write-Host "[README] $readme"
Write-Host "============================================================"
