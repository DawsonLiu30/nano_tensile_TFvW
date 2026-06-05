# PROFESS 3.0 iservice Runbook

Date: 2026-06-05

## Status

PROFESS 3.0 has been compiled natively on NCHC iservice and passed both
official density and ion-relax smoke tests.

Use this binary for future iservice PROFESS jobs:

```text
/gpfs-work/dawson666/profess3_build_20260605/bin/PROFESS
```

Optional wrapper:

```text
/gpfs-work/dawson666/bin/profess3_iservice
```

## Build Record

- Build type: native serial PROFESS 3.0 on iservice
- Compiler: `/usr/bin/gfortran`, GNU Fortran 8.5.0
- Linked libraries:
  - `/lib64/liblapack.so.3`
  - `/lib64/libblas.so.3`
  - `/lib64/libfftw3.so.3`
  - `/lib64/libgfortran.so.5`
- Dynamic-library check: all libraries found for the iservice build.

The earlier local WSL binary should not be used on iservice because it required
newer runtime libraries (`GLIBC_2.34`, `GLIBC_2.29`, `GFORTRAN_10`).

## Patch Caveat

`Source/XC_PBE.f90::PBE_LibXC` was replaced by a no-libxc stub because the
iservice environment did not provide:

```text
xc_f90_types_m.mod
xc_f90_lib_m.mod
```

Use LDA or internal non-spin PBE paths with this binary. Do not use
spin-polarized PBE through Libxc.

## Smoke Test 1: optDen

Official case:

```text
/gpfs-work/dawson666/profess3_build_20260605/src/test/optDen
```

Command:

```bash
/gpfs-work/dawson666/profess3_build_20260605/bin/PROFESS test
```

Important result:

```text
Total energy converged in the last three iteration within: 1.00000E-06 Ha
Final total energy: -8.517049280096E+00 Ha, volume= 4.2196E+02 bohr^3, totQ=12.0
```

Observed output files:

```text
test.out
test.err
smoke_optDen_PROFESS_20260605_correct.log
```

`test.err` was 0 byte.

## Smoke Test 2: optIon

Official case:

```text
/gpfs-work/dawson666/profess3_build_20260605/src/test/optIon
```

Command:

```bash
/gpfs-work/dawson666/profess3_build_20260605/bin/PROFESS displaced
```

Important result:

```text
Final total energy: -8.515254302646E+00 Ha
CG iter: 7
maxForce = 2.9069E-05 Ha/bohr
END OF PROFESS, HAVE A GREAT DAY !
```

Observed output files:

```text
displaced.final.geom
displaced.out
displaced.err
smoke_optIon_PROFESS_20260605_correct.log
```

`displaced.err` was 0 byte.

## Correct Invocation

PROFESS expects the input basename, not the `.inpt` filename.

Correct:

```bash
PROFESS test
```

Wrong:

```bash
PROFESS test.inpt
```

If the input file is `case_name.inpt`, run:

```bash
/gpfs-work/dawson666/profess3_build_20260605/bin/PROFESS case_name
```

## Production Folder Layout

Use one shared structure source and keep calculator outputs separate:

```text
<series>/
  structures/<case>/
    pristine_start.vasp
    vacancy_start.vasp
    structure_metadata.json

  profess/<KEDF>/<case>/
    pristine_relax.inpt
    pristine_relax.ion
    vacancy_relax.inpt
    vacancy_relax.ion
    manifest.json
    *.out / *.stdout / *.stderr / *.final.geom

  dftpy/
    README_SAME_STRUCTURES.txt

  shared_structure_manifest.csv
  profess_case_manifest.csv
  settings.tsv
```

`structures/` is the source of truth. PROFESS and DFTpy must both be generated
from these same VASP files so the comparison is not contaminated by geometry
mismatch.

## Slurm Routing

For short PROFESS relax jobs:

```text
partition = ctest
time      = 02:00:00
array     = %2
```

For long/large cases:

```text
partition = ct56
time      = 4-00:00:00
```

## Current Submit Command

```bash
cd /mnt/c/Users/dawso/nano_tensile_TFvW

PARTITION=ctest TIME_LIMIT=02:00:00 MAX_PARALLEL=2 \
DIAMETERS="1.0,1.5,2.0" \
SHAPES="circle,hexagon" \
POSITIONS="inner,middle,outer" \
KEDFS="TFPLUS_DEFAULT,CAT" \
bash scripts/push_profess_periodic_vacancy_relax_to_iservice_20260605.sh
```

Pull results:

```bash
cd /mnt/c/Users/dawso/nano_tensile_TFvW
bash scripts/pull_profess_periodic_vacancy_relax_results_20260605.sh
```

