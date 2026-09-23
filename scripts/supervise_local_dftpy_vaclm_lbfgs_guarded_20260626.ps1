param(
    [string]$OutRoot = (Join-Path $env:USERPROFILE "Desktop\LOCAL_DFTPY_VACLM_10X10_FULL_RERUN_LBFGS_GUARDED_20260626"),
    [int]$CheckIntervalSeconds = 600,
    [int]$StaleMinutes = 90,
    [int]$TotalCases = 100
)

$ErrorActionPreference = "Continue"

$Repo = "C:\Users\dawso\nano_tensile_TFvW"
$Starter = Join-Path $Repo "scripts\start_local_dftpy_vaclm_10x10_rerun_20260626.ps1"
$SupervisorLog = Join-Path $OutRoot "local_rerun_supervisor.log"

function Write-SupervisorLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] $Message" | Add-Content -Path $SupervisorLog -Encoding UTF8
}

function Get-AttemptedCount {
    if (-not (Test-Path (Join-Path $OutRoot "03_runs"))) {
        return 0
    }
    $resultCount = @(Get-ChildItem (Join-Path $OutRoot "03_runs") -Recurse -Filter "result.json" -ErrorAction SilentlyContinue).Count
    $failedCount = @(Get-ChildItem (Join-Path $OutRoot "03_runs") -Recurse -Filter "LOCAL_RERUN_FAILED.txt" -ErrorAction SilentlyContinue).Count
    return ($resultCount + $failedCount)
}

function Get-ManagedPid {
    $infoPath = Join-Path $OutRoot "BACKGROUND_PROCESS.json"
    if (-not (Test-Path $infoPath)) {
        return $null
    }
    try {
        $info = Get-Content $infoPath -Raw | ConvertFrom-Json
        return [int]$info.pid
    } catch {
        return $null
    }
}

function Start-RerunResume {
    Write-SupervisorLog "Starting/resuming guarded LBFGS rerun."
    powershell -NoProfile -ExecutionPolicy Bypass -File $Starter `
        -Mode full `
        -AseOptimizer LBFGS `
        -AbortFmax 1.0 `
        -AbortAfterSteps 100 `
        -MinPristineA0 3.5 `
        -MaxPristineA0 4.5 `
        -OutRoot $OutRoot | Out-Null
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
Write-SupervisorLog "Supervisor started. OutRoot=$OutRoot"

while ($true) {
    $attempted = Get-AttemptedCount
    if ($attempted -ge $TotalCases) {
        Write-SupervisorLog "All cases attempted: $attempted/$TotalCases. Supervisor exiting."
        break
    }

    $managedPid = Get-ManagedPid
    $proc = $null
    if ($managedPid) {
        $proc = Get-Process -Id $managedPid -ErrorAction SilentlyContinue
    }

    if (-not $proc) {
        Write-SupervisorLog "Managed process is not running. attempted=$attempted/$TotalCases. Resuming."
        Start-RerunResume
        Start-Sleep -Seconds 10
    } else {
        $latest = Get-ChildItem (Join-Path $OutRoot "03_runs") -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -notin @("point_manifest.json", "CASE_README.md") } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1

        if ($latest) {
            $ageMinutes = ((Get-Date) - $latest.LastWriteTime).TotalMinutes
            Write-SupervisorLog ("Alive pid={0}; attempted={1}/{2}; latest_age_min={3:N1}; latest={4}" -f $managedPid, $attempted, $TotalCases, $ageMinutes, $latest.FullName)
            if ($ageMinutes -gt $StaleMinutes) {
                Write-SupervisorLog "No file updates for more than $StaleMinutes minutes. Restarting managed process."
                Stop-Process -Id $managedPid -Force -ErrorAction SilentlyContinue
                Start-Sleep -Seconds 5
                Start-RerunResume
            }
        } else {
            Write-SupervisorLog "Alive pid=$managedPid; attempted=$attempted/$TotalCases; no files found yet."
        }
    }

    Start-Sleep -Seconds $CheckIntervalSeconds
}
