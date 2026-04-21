$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

python scripts/prepare_grip_geometry_preview.py `
    --diameters 1,2,3,4,5,6 `
    --orientation 111 `
    --a0 4.118877004246 `
    --wire-length 21.0 `
    --min-wire-span 10.0 `
    --xy-vacuum 10.0 `
    --z-vacuum 10.0 `
    --grip-thickness 3.0 `
    --vacancy-z-window-fraction 0.35

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
