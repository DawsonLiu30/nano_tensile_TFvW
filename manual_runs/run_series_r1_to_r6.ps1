param(
    [string[]]$Diameters = @("1", "2", "3", "4", "5", "6"),
    [int]$Cycles = 20,
    [switch]$ResumeLatest
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Convert-DiameterList([string[]]$Values) {
    $out = @()
    foreach ($value in $Values) {
        foreach ($chunk in ([string]$value).Split(",")) {
            $token = $chunk.Trim()
            if ($token) {
                $out += [double]::Parse($token, [System.Globalization.CultureInfo]::InvariantCulture)
            }
        }
    }
    return $out
}

$DiameterValues = Convert-DiameterList $Diameters

Write-Host "This runs cases sequentially. On a 32 GB machine, do not run another DFT job in parallel."
foreach ($d in $DiameterValues) {
    Write-Host "========================================"
    Write-Host "[series] Starting diameter/r label $d"
    Write-Host "========================================"
    $caseArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "$PSScriptRoot\run_one_case.ps1",
        "-Diameter", "$d",
        "-Cycles", "$Cycles"
    )
    if ($ResumeLatest) {
        $caseArgs += "-ResumeLatest"
    }
    powershell @caseArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "[series] All requested cases finished."
