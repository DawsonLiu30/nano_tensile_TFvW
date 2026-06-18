# Weekend Handoff

Date: 2026-06-18

This is the minimal handoff file for continuing the vacancy/divacancy work from
another computer during the weekend.

## 1. Get The Repository At Home

Fresh clone:

```bash
git clone -b codex/cleanup-core-results https://github.com/DawsonLiu30/nano_tensile_TFvW.git
cd nano_tensile_TFvW
```

If the repo already exists:

```bash
cd nano_tensile_TFvW
git fetch origin
git switch codex/cleanup-core-results
git pull --ff-only
```

Main files to open first:

```text
CURRENT_WORKFLOW_INDEX_20260616.md
ADVISOR_RESPONSE_ACTIONS_20260618.md
WEEKEND_HANDOFF_20260618.md
```

## 2. Current Active Remote Calculations

Current NCHC work root:

```text
/work/dawson666
```

Current QE divacancy job:

```text
Job: 1553956_[0-4]
Partition: ct56
Name: QEVCR3
Status on 2026-06-18: all five tasks running
```

Remote QE series:

```text
/work/dawson666/qe_cases/qe_runs/qe_divacancy_vcrelax_conv3x3x3_rscan_20260616
```

Remote DFTpy divacancy series:

```text
/work/dawson666/dftpy_project/relax/dftpy45/results/dftpy_divacancy_vcrelax_conv3x3x3_rscan_20260616
```

Do not cancel or modify the QE jobs unless they clearly fail. They are the
missing DFT/QE reference requested by Professor Lueder.

## 3. Check QE Status On iservice

```bash
squeue -u dawson666

sacct -j 1553956 --format=JobID,JobName%20,Partition,State,Elapsed,Timelimit,AllocCPUS,ReqMem,ExitCode,Start,End
```

After `squeue` no longer shows `1553956`, inspect completion:

```bash
cd /work/dawson666/qe_cases/qe_runs/qe_divacancy_vcrelax_conv3x3x3_rscan_20260616

for d in pair_scan/*; do
  [ -d "$d" ] || continue
  echo "============================================================"
  echo "$d"
  grep -R "JOB DONE" "$d" --include="*.out" | wc -l
  grep -RInE "JOB DONE|convergence has been achieved|Total force|!    total energy|Error|MPI_ABORT|convergence NOT" \
    "$d" --include="*.out" --include="*.err" | tail -80
done
```

Each pair case should ideally have two successful QE `vc-relax` outputs:
pristine and divacancy.

## 4. Pull Results Back Home After QE Finishes

From WSL on the home computer:

```bash
cd /mnt/c/Users/dawso/nano_tensile_TFvW

DFTPY_SERIES=dftpy_divacancy_vcrelax_conv3x3x3_rscan_20260616 \
QE_SERIES=qe_divacancy_vcrelax_conv3x3x3_rscan_20260616 \
bash scripts/pull_divacancy_rscan_results_20260616.sh
```

This pulls:

- QE raw input/output.
- DFTpy raw input/output.
- Scheduler logs.
- Reproducibility scripts/pseudos.
- Processed DFTpy/QE summaries.
- A compact zip.

## 5. Advisor Feedback Priority

Do not send another large package before the following are fixed:

1. Single-vacancy `lambda/mu` scan must be treated as the calibration baseline.
2. QE/DFT reference values must be added.
3. Divacancy data must be labeled as pilot unless repeated with final selected
   `lambda/mu`.
4. Each raw case folder must directly include DFTpy input/output files.
5. Package must define:
   - `E_2vac(r)`
   - `E_2vac/2`
   - binding or relative-energy convention
6. Add compact evaluation slides/notebook before resubmission.

See:

```text
ADVISOR_RESPONSE_ACTIONS_20260618.md
```

## 6. Current DFTpy Divacancy Pilot Result

These are DFTpy/LDA/TFvW with `lambda=1.0`, `mu=0.13`, `spacing=0.20 A`,
full atom+cell relaxation, conventional fcc `3x3x3`, `108 -> 106` atoms.

| r (A) | E_2vac (eV) | E_2vac/2 (eV) | Status |
|---:|---:|---:|---|
| 2.8566 | 1.151372 | 0.575686 | pilot, force converged |
| 4.0398 | 1.191490 | 0.595745 | pilot, force converged |
| 5.7132 | 1.206444 | 0.603222 | pilot, force converged |
| 6.3876 | 1.199246 | 0.599623 | pilot, force converged |
| 8.5698 | 1.210838 | 0.605419 | pilot, force converged |

The small dip near `6.39 A` must be explained or checked against QE/larger cell
before making a physical claim.

## 7. Known Path Fixes

Use these current paths:

```text
DFTpy root: /work/dawson666/dftpy_project/relax/dftpy45
QE root:    /work/dawson666/qe_cases/qe_runs
QE pw.x:    /work/dawson666/q-e-qe-7.3.1/PW/src/pw.x
PROFESS:    /work/dawson666/profess3_build_20260605/bin/PROFESS
```

Old `/gpfs-work/dawson666` paths are historical and should not be used for new
submission scripts.

