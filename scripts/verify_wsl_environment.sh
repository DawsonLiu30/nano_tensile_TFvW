#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo="$(cd "$script_dir/.." && pwd -P)"
runtime="${AL_DEFECTS_RUNTIME:-/var/tmp/al-defects-runtime-20260907}"
env_prefix="${AL_DEFECTS_ENV_PREFIX:-$runtime/env}"
data_root="${AL_DEFECTS_DATA_ROOT:-/mnt/c/OFDFT/AL_DEFECTS_USB_HANDOFF_20260717}"
python="$env_prefix/bin/python"
pwx="$env_prefix/bin/pw.x"
pseudo="$data_root/03_ACTIVE_QE_VCRELAX_REFERENCE/pseudo/Al_PAW_PBE.UPF"

for required in "$python" "$pwx" "$repo/al.gga.recpot" "$pseudo"; do
  if [[ ! -e "$required" ]]; then
    echo "[ERROR] Missing required path: $required" >&2
    exit 2
  fi
done

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

"$python" - <<'PY'
import sys
import ase
import dftpy
import matplotlib
import numpy
import openpyxl
import pandas
import pylibxc
import scipy

print(f"python={sys.version.split()[0]}")
print(f"dftpy={dftpy.__version__}")
print(f"pylibxc={pylibxc.__version__}")
print(f"numpy={numpy.__version__}")
print(f"scipy={scipy.__version__}")
print(f"ase={ase.__version__}")
print(f"pandas={pandas.__version__}")
print(f"matplotlib={matplotlib.__version__}")
print(f"openpyxl={openpyxl.__version__}")
PY

"$python" -m compileall -q "$repo"
(
  cd "$repo"
  "$python" -c 'import app.dft_engine'
  "$python" scripts/bulk_validate.py --help >/dev/null
  "$python" scripts/run_periodic_tensile.py --help >/dev/null
  "$python" scripts/run_dftpy_official_relax_vaclm_one.py --help >/dev/null
)

smoke_root="$(mktemp -d "$runtime/smoke-check.XXXXXX")"
cleanup_smoke() {
  rm -rf "$smoke_root"
}
trap cleanup_smoke EXIT

(
  cd "$repo"
  "$python" scripts/bulk_validate.py \
    --pp al.gga.recpot \
    --kedf TFVW \
    --spacing 0.8 \
    --strains 0.0 \
    --outdir "$smoke_root/dftpy" \
    >"$smoke_root/dftpy.log" 2>&1
)

mkdir -p "$smoke_root/qe_tmp"
cat >"$smoke_root/qe.in" <<EOF
&CONTROL
  calculation = 'scf'
  prefix = 'al_smoke'
  pseudo_dir = '$(dirname "$pseudo")'
  outdir = '$smoke_root/qe_tmp'
/
&SYSTEM
  ibrav = 2
  celldm(1) = 7.6343
  nat = 1
  ntyp = 1
  ecutwfc = 15.0
  ecutrho = 120.0
  occupations = 'smearing'
  smearing = 'mv'
  degauss = 0.02
/
&ELECTRONS
  conv_thr = 1.0d-6
  electron_maxstep = 30
/
ATOMIC_SPECIES
Al 26.9815385 $(basename "$pseudo")
ATOMIC_POSITIONS crystal
Al 0.0 0.0 0.0
K_POINTS automatic
1 1 1 0 0 0
EOF

"$pwx" -in "$smoke_root/qe.in" >"$smoke_root/qe.out" 2>&1
grep -q 'Program PWSCF v.7.5' "$smoke_root/qe.out"
grep -q 'convergence has been achieved' "$smoke_root/qe.out"
grep -q 'JOB DONE' "$smoke_root/qe.out"

echo "dftpy_compute=PASS"
echo "qe_7_5_compute=PASS"
echo "environment_verification=PASS"
