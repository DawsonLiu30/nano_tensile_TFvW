param(
    [string]$Settings = "tfvw_lam0p9_mu0p1,tfvw_lam1p0_mu0p2,tfvw_lam0p8_mu0p1,tfvw_lam0p1_mu0p3",
    [string[]]$Optimizers = @("BFGS", "LBFGS", "BFGSLineSearch", "SciPyFminCG", "MDMin")
)

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\dawso\nano_tensile_TFvW"
$BenchRoot = Join-Path $env:USERPROFILE "Desktop\LOCAL_DFTPY_VACLM_OPTIMIZER_BENCH_20260626"
$Runner = Join-Path $Repo "scripts\run_local_dftpy_vaclm_10x10_rerun.py"

New-Item -ItemType Directory -Force -Path $BenchRoot | Out-Null

$Summary = Join-Path $BenchRoot "optimizer_benchmark_runs.tsv"
"optimizer`tstatus`tseconds`toutroot" | Set-Content -Path $Summary -Encoding UTF8

foreach ($Opt in $Optimizers) {
    $OutRoot = Join-Path $BenchRoot $Opt
    Write-Host "============================================================"
    Write-Host "[RUN] optimizer=$Opt"
    Write-Host "[OUT] $OutRoot"
    $Start = Get-Date
    $Status = "OK"
    try {
        python $Runner `
            --outroot $OutRoot `
            --mode pristine `
            --ase-optimizer $Opt `
            --settings $Settings `
            --abort-fmax 100000 `
            --abort-after-steps 80 `
            --min-pristine-a0 3.5 `
            --max-pristine-a0 4.5 `
            --collect-every 1
    }
    catch {
        $Status = "FAILED"
        Write-Host "[FAILED] optimizer=$Opt"
        Write-Host $_
    }
    $Seconds = [math]::Round(((Get-Date) - $Start).TotalSeconds, 1)
    "$Opt`t$Status`t$Seconds`t$OutRoot" | Add-Content -Path $Summary -Encoding UTF8
}

Write-Host "============================================================"
Write-Host "[DONE] $Summary"
