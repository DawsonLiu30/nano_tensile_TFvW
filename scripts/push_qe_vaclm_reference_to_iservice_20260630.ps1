param(
    [string]$RemoteHost = "dawson666@twnia3.nchc.org.tw",
    [string]$RemoteRoot = "/work/dawson666/qe_cases/qe_runs",
    [string]$LocalFolder = "C:\Users\dawso\Desktop\QE_VACLM_REFERENCE_20260630"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $LocalFolder)) {
    throw "Missing local QE folder: $LocalFolder"
}

Write-Host "============================================================"
Write-Host "Push QE VACLM reference workflow to iService"
Write-Host "============================================================"
Write-Host "[LOCAL ] $LocalFolder"
Write-Host "[REMOTE] ${RemoteHost}:${RemoteRoot}/QE_VACLM_REFERENCE_20260630"
Write-Host ""
Write-Host "This uses scp -O because NCHC may reject the SFTP subsystem."
Write-Host "It is expected to ask for NCHC 2FA/password."
Write-Host "============================================================"

ssh $RemoteHost "mkdir -p '$RemoteRoot'"
scp -O -r $LocalFolder "${RemoteHost}:${RemoteRoot}/"

Write-Host ""
Write-Host "Upload complete. Submit on iService with:"
Write-Host "  cd ${RemoteRoot}/QE_VACLM_REFERENCE_20260630"
Write-Host "  sbatch submit_qe_vacancy_reference_array.sh"
