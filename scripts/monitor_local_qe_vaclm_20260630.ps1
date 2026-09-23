$ErrorActionPreference = "Continue"

$LogDir = "C:\Users\dawso\Desktop\QE_VACLM_LOCAL_MONITOR_20260630"
$Log = Join-Path $LogDir "monitor.log"
$DoneFlag = Join-Path $LogDir "MONITOR_DONE.txt"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Log($Message) {
    $line = "$(Get-Date -Format s) $Message"
    Add-Content -Path $Log -Value $line
}

function Invoke-Wsl($Command) {
    & wsl.exe -d Ubuntu -- bash -lc $Command 2>&1
}

Write-Log "START monitor local QE VACLM"
Write-Log "Run root: /home/dawson666/qe_vaclm_local_20260630"

while ($true) {
    try {
        $status = Invoke-Wsl "cd /home/dawson666/qe_vaclm_local_20260630 2>/dev/null || exit 2; echo '[STATUS_FILE]'; cat LOCAL_QE_STATUS.tsv 2>/dev/null || true; echo '[PW_PROCS]'; ps -eo pid,ppid,pcpu,pmem,etime,args | grep 'pw.x -in pw.in' | grep -v grep || true; echo '[JOB_DONE]'; for c in pristine_vcrelax vacancy_vcrelax; do printf '%s ' \$c; grep -c 'JOB DONE' \$c/pw.out 2>/dev/null || true; done; echo '[TAIL]'; tail -20 pristine_vcrelax/pw.out 2>/dev/null || true; tail -20 vacancy_vcrelax/pw.out 2>/dev/null || true"
        Add-Content -Path $Log -Value "============================================================"
        Add-Content -Path $Log -Value "$(Get-Date -Format s)"
        Add-Content -Path $Log -Value $status

        $done = Invoke-Wsl "cd /home/dawson666/qe_vaclm_local_20260630 2>/dev/null || exit 2; p=\$(grep -c 'JOB DONE' pristine_vcrelax/pw.out 2>/dev/null || true); v=\$(grep -c 'JOB DONE' vacancy_vcrelax/pw.out 2>/dev/null || true); if [ \"\$p\" -ge 1 ] && [ \"\$v\" -ge 1 ]; then echo DONE; else echo RUNNING; fi"
        if (($done -join "`n") -match "DONE") {
            Write-Log "Both QE calculations have JOB DONE. Copying results back to Desktop."
            $copy = & wsl.exe -d Ubuntu -- bash /mnt/c/Users/dawso/nano_tensile_TFvW/scripts/copy_local_qe_vaclm_results_20260630.sh 2>&1
            Add-Content -Path $Log -Value $copy
            Set-Content -Path $DoneFlag -Value "DONE $(Get-Date -Format s)"
            break
        }
    } catch {
        Write-Log "ERROR $($_.Exception.Message)"
    }

    Start-Sleep -Seconds 600
}

Write-Log "END monitor"
