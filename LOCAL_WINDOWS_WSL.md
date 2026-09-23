# Local Windows + WSL Runtime

> Current computer, 2026-09-08: source `C:\OFDFT\nano_tensile_TFvW`, runtime
> `/var/tmp/al-defects-runtime-20260907`, launcher
> `C:\OFDFT\LOCAL_MACHINE_20260907\OFDFT.ps1`. Use `README.md` for current
> commands. The July paths and setup narrative below are retained as migration
> history and must not be copied as commands on this computer.

This file defines the supported local-computer setup created during the
2026-07-29 USB migration.

## Authoritative locations

- Git working repository:
  `C:\Users\s1070\Documents\Codex\2026-07-29\usb\work\nano_tensile_TFvW`
- Complete migrated handoff:
  `C:\Users\s1070\Documents\Codex\2026-07-29\usb\work\AL_DEFECTS_USB_HANDOFF_20260717`
- Canonical DFTpy raw data:
  `AL_DEFECTS_USB_HANDOFF_20260717\02_ACTIVE_DFTPY_ISERVICE_LAMMU_RAW`
- Canonical QE reference data:
  `AL_DEFECTS_USB_HANDOFF_20260717\03_ACTIVE_QE_VCRELAX_REFERENCE`
- WSL runtime, environment, scratch, and fast working copy:
  `/var/tmp/al-defects-runtime-20260717`

The Windows copies are authoritative and backed by the USB migration
verification. The WSL runtime is reproducible scratch space. Do not put new
scientific results back into the verified handoff directory; use a separate
run/output directory.

## Verified stack

- Windows 64-bit, WSL2, Ubuntu 24.04
- Python 3.11
- DFTpy 2.1.2
- pylibxc 7.0.0
- NumPy 2.4.6
- SciPy 1.17.1
- ASE 3.29.0
- Quantum ESPRESSO / PWSCF 7.5
- Open MPI from the same conda-forge environment

The exact direct dependencies are recorded in `environment-wsl.yml`.

## Rebuild or refresh

From WSL, while this repository is the current directory:

```bash
bash scripts/bootstrap_wsl_runtime.sh
bash scripts/verify_wsl_environment.sh
```

From PowerShell:

```powershell
wsl.exe -d Ubuntu-24.04 -- bash `
  /mnt/c/Users/s1070/Documents/Codex/2026-07-29/usb/work/nano_tensile_TFvW/scripts/bootstrap_wsl_runtime.sh

wsl.exe -d Ubuntu-24.04 -- bash `
  /mnt/c/Users/s1070/Documents/Codex/2026-07-29/usb/work/nano_tensile_TFvW/scripts/verify_wsl_environment.sh
```

Run Python scripts with:

```bash
/var/tmp/al-defects-runtime-20260717/env/bin/python scripts/bulk_validate.py --help
```

Run QE with controlled threading:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /var/tmp/al-defects-runtime-20260717/env/bin/pw.x -in pw.in
```

Prepare the migrated 108/107-atom local QE reference without starting the
multi-hour calculation:

```bash
bash scripts/run_local_qe_vaclm_wsl.sh --prepare-only
```

Start both cases with four MPI ranks only after reviewing the prepared inputs:

```bash
bash scripts/run_local_qe_vaclm_wsl.sh --run --case both --np 4
```

The original local QE outputs used four MPI processes and about 1.3 GB per
process. This WSL instance exposes about 7.5 GB RAM, so four ranks fit but leave
limited headroom. Keep OpenMP/BLAS at one thread per rank.

## Path policy

Many older reports and archival transfer scripts contain paths for the old
computer (`C:\Users\dawso`, `/mnt/c/Users/dawso`) or iService
(`/work/dawson666`). Those paths are provenance and must not be globally
rewritten. The supported active calculation scripts use relative paths, CLI
arguments, or the following overrides:

- `AL_DEFECTS_RUNTIME`
- `AL_DEFECTS_DATA_ROOT`
- `AL_DEFECTS_ENV_PREFIX`
- `QE_RUNROOT`

The `.gitattributes` file forces LF line endings for Linux, Python, Slurm, and
QE/DFTpy text files so Windows Git cannot make WSL or HPC scripts unusable.

## Presentation scripts

The four `.mjs` presentation scripts import `@oai/artifact-tool`. This computer
has that package in the Codex bundled runtime and Node.js 24.14.0. Those scripts
should be run through Codex's presentation workflow, not treated as ordinary
npm-only utilities.
