param(
    [Parameter(Mandatory = $true)]
    [int]$WaitPid,

    [string[]]$Diameters = @("3", "4"),

    [string]$RequiredSummary = "",
    [string]$RequiredFractureCsv = "",
    [int]$RequiredFinalCycle = 20,
    [int]$Cycles = 20,
    [switch]$AllowFractureStop
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

Write-Host "========================================"
Write-Host "[continue] Waiting for PID $WaitPid"
Write-Host "[continue] Then run diameters: $($DiameterValues -join ', ')"
Write-Host "========================================"

$proc = Get-Process -Id $WaitPid -ErrorAction SilentlyContinue
if ($proc) {
    Wait-Process -Id $WaitPid
    Write-Host "[continue] PID $WaitPid finished."
} else {
    Write-Host "[continue] PID $WaitPid is not running. Continuing validation."
}

if ($RequiredSummary.Trim()) {
    $summaryPath = Resolve-Path -LiteralPath $RequiredSummary -ErrorAction Stop
    $rows = Import-Csv -LiteralPath $summaryPath
    if (-not $rows -or $rows.Count -eq 0) {
        throw "[continue] Required summary is empty: $summaryPath"
    }
    $lastCycle = [int]$rows[-1].cycle
    Write-Host "[continue] Required summary last cycle = $lastCycle"
    if ($lastCycle -lt $RequiredFinalCycle) {
        $fractured = $false
        if ($AllowFractureStop -and $RequiredFractureCsv.Trim()) {
            $fracturePath = Resolve-Path -LiteralPath $RequiredFractureCsv -ErrorAction Stop
            $fractureRows = Import-Csv -LiteralPath $fracturePath
            $fractured = [bool]($fractureRows | Where-Object { "$($_.fractured)".ToLowerInvariant() -eq "true" } | Select-Object -First 1)
            Write-Host "[continue] Fracture stop allowed; fractured=$fractured"
        }
        if (-not $fractured) {
            throw "[continue] Required case did not finish cycle $RequiredFinalCycle and did not fracture. Not starting next cases."
        }
    }
}

foreach ($d in $DiameterValues) {
    Write-Host "========================================"
    Write-Host "[continue] Starting diameter/r label $d"
    Write-Host "========================================"
    powershell -NoProfile -ExecutionPolicy Bypass -File "$PSScriptRoot\run_one_case.ps1" -Diameter $d -Cycles $Cycles
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "[continue] All queued continuation cases finished."
