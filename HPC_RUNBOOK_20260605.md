# NCHC HPC Runbook: DFTpy / PROFESS Routing

Date: 2026-06-05

## Queue Routing Rule

Use `ctest` for short jobs that are expected to finish within 2 hours.

- Partition: `ctest`
- Time limit: `02:00:00`
- Practical concurrency: run 2 tasks at once, keep 1 waiting.
- Recommended Slurm array throttle: `%2`
- Best for: smoke tests, SCF checks, short DFTpy scans, short PROFESS SCF/small relax cases.

Use `ct56` for long jobs.

- Partition: `ct56`
- Time limit: normally `4-00:00:00` for long production jobs.
- Queue can be long because of `QOSGrpJobsLimit`.
- Best for: QE vc-relax, dense k-point jobs, long DFTpy vc-relax, long PROFESS relaxation/tensile jobs.

## PROFESS on iservice

The local WSL PROFESS binary should not be used directly on iservice. It failed the runtime check because it required newer libraries:

- `GLIBC_2.34`
- `GLIBC_2.29`
- `GFORTRAN_10`

PROFESS was then compiled directly on iservice and is usable there.

Compiled binary:

```text
/gpfs-work/dawson666/profess3_build_20260605/bin/PROFESS
```

Build/source directory:

```text
/gpfs-work/dawson666/profess3_build_20260605
```

Successful checks observed:

```text
Serial version of PROFESS
WELCOME TO PROFESS
You need to specify the file on the command line. Leaving.
OK: all dynamic libraries found
bin/PROFESS size: 1.4M
```

The binary also passed the official PROFESS `optDen` and `optIon` Al4 smoke tests:

```text
optDen final total energy: -8.517049280096E+00 Ha
optIon final total energy: -8.515254302646E+00 Ha
optIon final maxForce    :  2.9069E-05 Ha/bohr
```

This means future PROFESS production can be moved from local WSL to NCHC Slurm
using the compiled iservice binary.

Detailed build/smoke-test record:

```text
PROFESS_ISERVICE_RUNBOOK_20260605.md
```

## DFTpy TFvW Weight Scan Submission

The DFTpy TF/vW weight-scan push script now supports partition routing:

```bash
PARTITION=ctest TIME_LIMIT=02:00:00 MAX_PARALLEL=2 \
SERIES_NAME=<series_name> \
Y_LIST="<comma-separated-y-values>" \
RELAX_STEPS=<steps> \
bash scripts/push_dftpy_tfvw_weight_scan_to_iservice_20260605.sh
```

For long or uncertain relaxations:

```bash
PARTITION=ct56 TIME_LIMIT=4-00:00:00 MAX_PARALLEL=4 \
bash scripts/push_dftpy_tfvw_weight_scan_to_iservice_20260605.sh
```

Important shell rule: do not put blank lines between continued environment
assignments. If a command is written with blank lines after `\`, Bash can lose
the custom variables and the script will silently use defaults.

Correct one-block fine scan:

```bash
cd /mnt/c/Users/dawso/nano_tensile_TFvW

SERIES_NAME=dftpy_vacancy_tfvw_weight_fine_y0115_020_conv3x3x3_lda_20260605 \
Y_LIST="0.115,0.120,0.125,0.130,0.135,0.140,0.145,0.150,0.155,0.160,0.170,0.180,0.200" \
RELAX_STEPS=5000 \
PARTITION=ctest \
TIME_LIMIT=02:00:00 \
MAX_PARALLEL=2 \
bash scripts/push_dftpy_tfvw_weight_scan_to_iservice_20260605.sh
```

After submission, the printed header must show the custom values. If it prints:

```text
[SERIES] dftpy_vacancy_tfvw_weight_conv3x3x3_lda_20260605
[YLIST ] 0.0,0.1,...,1.0
[NSTEPS] 1500
```

then the custom fine scan was not submitted; the script used its defaults.

Fine weight-scan setting names must also be checked. The corrected script keeps
enough decimal places so values such as `0.115`, `0.120`, and `0.125` become
distinct folders (`y0p115`, `y0p12`, `y0p125`). If the remote
`settings_weight_scan.txt` shows repeated labels, do not submit that series.

If the prepare step succeeds but Slurm rejects submission with
`QOSMaxSubmitJobPerUserLimit`, do not rerun the full push/prepare script. Wait
for the job count to drop or cancel older arrays, then submit the existing
prepared series:

```bash
cd /mnt/c/Users/dawso/nano_tensile_TFvW

SERIES_NAME=dftpy_vacancy_tfvw_weight_fine_y0115_020_conv3x3x3_lda_20260605_v2 \
PARTITION=ctest \
TIME_LIMIT=02:00:00 \
MAX_PARALLEL=2 \
bash scripts/submit_existing_dftpy_tfvw_weight_scan_to_iservice_20260605.sh
```

For `ctest`, avoid submitting a large array at once. It effectively behaves like
"2 running + 1 waiting", so submit chunks of about 3 array elements. Use the
positional `START END` form to avoid pasted-line-continuation problems:

```bash
cd /mnt/c/Users/dawso/nano_tensile_TFvW

SERIES_NAME=dftpy_vacancy_tfvw_weight_fine_y0115_020_conv3x3x3_lda_20260605_v2 PARTITION=ctest TIME_LIMIT=02:00:00 MAX_PARALLEL=2 bash scripts/submit_existing_dftpy_tfvw_weight_scan_to_iservice_20260605.sh 0 2
```

After those finish, submit the next chunks:

```text
bash scripts/submit_existing_dftpy_tfvw_weight_scan_to_iservice_20260605.sh 3 5
bash scripts/submit_existing_dftpy_tfvw_weight_scan_to_iservice_20260605.sh 6 8
bash scripts/submit_existing_dftpy_tfvw_weight_scan_to_iservice_20260605.sh 9 11
bash scripts/submit_existing_dftpy_tfvw_weight_scan_to_iservice_20260605.sh 12 12
```

Useful queue cleanup checks:

```bash
squeue -u dawson666
squeue -u dawson666 -h -o "%A %T %j %P %M %R"
```

Cancel only the specific old array if it is unwanted:

```bash
scancel <jobid>
```

## Current Running Note

At the time this note was written, the fine DFTpy TFvW weight scan was submitted on `ct56` as:

```text
1459377_[1-10%4] ct56 dftpyWGT PD
1459377_0        ct56 dftpyWGT R
```

Future short scans should be routed to `ctest` unless expected runtime exceeds 2 hours.

## PROFESS Periodic Vacancy Relax Production

Now that PROFESS is compiled on iservice, use the NCHC binary for production:

```text
/gpfs-work/dawson666/profess3_build_20260605/bin/PROFESS
```

The first recommended production sweep is fixed-cell ionic relaxation of
axially periodic vacancy nanocolumn/faceted-nanocolumn structures.

Default first-wave setup:

- Shapes: `circle,hexagon`
- Orientation: `111`
- Diameters: `1.0,1.5,2.0` nm
- Vacancy positions: `inner,middle,outer`
- KEDFs: `TFPLUS_DEFAULT,CAT`
- Cell mode: fixed cell, ion relaxation only
- Minimum axial periodic length: `10 A`
- Vacancy concentration recorded as `1 / N_pristine`
- Default partition: `ctest`, because small/medium cases are expected to be short

Submit:

```bash
cd /mnt/c/Users/dawso/nano_tensile_TFvW

PARTITION=ctest \
TIME_LIMIT=02:00:00 \
MAX_PARALLEL=2 \
bash scripts/push_profess_periodic_vacancy_relax_to_iservice_20260605.sh
```

If a case times out on `ctest`, rerun that subset on `ct56` with a longer time limit.

Pull and collect:

```bash
cd /mnt/c/Users/dawso/nano_tensile_TFvW

bash scripts/pull_profess_periodic_vacancy_relax_results_20260605.sh
```

Main local outputs:

```text
C:\Users\dawso\Desktop\PROFESS_PERIODIC_VACANCY_RELAX_20260605\<series>\
  profess_periodic_vacancy_relax_summary.csv
  profess_periodic_vacancy_relax_completion.csv
  profess_periodic_vacancy_relax_Ef.png
```

The production layout now separates shared structures from calculator outputs:

```text
<series>/
  structures/<case>/          # canonical VASP structures
  profess/<KEDF>/<case>/      # PROFESS inputs/outputs
  dftpy/                      # reserved for DFTpy from the same structures
```

The important rule is: DFTpy and PROFESS must be generated from the same
`structures/<case>/pristine_start.vasp` and `structures/<case>/vacancy_start.vasp`.
