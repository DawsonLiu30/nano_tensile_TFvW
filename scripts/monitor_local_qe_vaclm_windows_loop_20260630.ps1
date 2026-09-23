param(
    [int]$IntervalSeconds = 180
)

$ErrorActionPreference = "Continue"

$Workspace = "C:\Users\dawso\nano_tensile_TFvW"
$MonitorDir = "C:\Users\dawso\Desktop\QE_VACLM_LOCAL_MONITOR_20260630"
$Log = Join-Path $MonitorDir "monitor.log"
$Done = Join-Path $MonitorDir "MONITOR_DONE.txt"
$StatusScript = "/mnt/c/Users/dawso/nano_tensile_TFvW/scripts/status_local_qe_vaclm_run_20260630.sh"
$CopyScript = "/mnt/c/Users/dawso/nano_tensile_TFvW/scripts/copy_local_qe_vaclm_results_20260630.sh"
$DoneCountsScript = "/mnt/c/Users/dawso/nano_tensile_TFvW/scripts/qe_vaclm_done_counts_20260630.sh"

New-Item -ItemType Directory -Force -Path $MonitorDir | Out-Null
"$(Get-Date -Format o) START Windows QE monitor loop interval=${IntervalSeconds}s" | Add-Content -Encoding UTF8 $Log

function Add-LogBlock {
    param([string]$Title, [string[]]$Lines)
    "============================================================" | Add-Content -Encoding UTF8 $Log
    "$(Get-Date -Format o) $Title" | Add-Content -Encoding UTF8 $Log
    $Lines | Add-Content -Encoding UTF8 $Log
}

while ($true) {
    try {
        $status = & wsl.exe -d Ubuntu -- bash $StatusScript 2>&1
        Add-LogBlock -Title "STATUS" -Lines $status

        $mem = & wsl.exe -d Ubuntu -- bash -lc 'free -h; ps -eo pid,ppid,pcpu,pmem,etime,args | grep "pw.x -in pw.in" | grep -v grep; true' 2>&1
        Add-LogBlock -Title "MEMORY_AND_PW_PROCESSES" -Lines $mem

        $countsText = & wsl.exe -d Ubuntu -- bash $DoneCountsScript 2>&1
        Add-LogBlock -Title "DONE_COUNTS" -Lines $countsText

        $pristineDone = 0
        $vacancyDone = 0
        foreach ($line in $countsText) {
            if ($line -match "^pristine_done=(\d+)") { $pristineDone = [int]$Matches[1] }
            if ($line -match "^vacancy_done=(\d+)") { $vacancyDone = [int]$Matches[1] }
        }

        if ($pristineDone -ge 1 -and $vacancyDone -ge 1) {
            "$(Get-Date -Format o) BOTH_DONE copying results" | Add-Content -Encoding UTF8 $Log
            $copy = & wsl.exe -d Ubuntu -- bash $CopyScript 2>&1
            Add-LogBlock -Title "COPY_RESULTS" -Lines $copy
            "DONE $(Get-Date -Format o)" | Set-Content -Encoding UTF8 $Done
            exit 0
        }
    }
    catch {
        Add-LogBlock -Title "MONITOR_EXCEPTION" -Lines @($_.Exception.Message)
    }

    Start-Sleep -Seconds $IntervalSeconds
}
