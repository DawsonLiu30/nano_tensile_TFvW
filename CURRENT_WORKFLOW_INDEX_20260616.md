# Current Vacancy Workflow Index

Date: 2026-06-16

Latest handoff update: 2026-06-22.  See `WEEKEND_HANDOFF_20260618.md` and
`ADVISOR_RESPONSE_ACTIONS_20260618.md` before continuing from another computer.

This file is the current handoff map for the vacancy/divacancy work.  It is
intended to keep the calculation families separated and reproducible.

## Canonical NCHC Paths

The NCHC work filesystem was moved back from `/gpfs-work/dawson666` to:

```text
/work/dawson666
```

Current canonical project roots:

| Area | Current root |
|---|---|
| DFTpy | `/work/dawson666/dftpy_project/relax/dftpy45` |
| QE | `/work/dawson666/qe_cases/qe_runs` |
| QE `pw.x` | `/work/dawson666/q-e-qe-7.3.1/PW/src/pw.x` |
| PROFESS | `/work/dawson666/profess3_build_20260605/bin/PROFESS` if migrated; older notes may show `/gpfs-work/...` |

Important: older notes and scripts may still contain historical `/gpfs-work`
paths. New divacancy scripts use `/work/dawson666` by default.

## Current Method Conventions

| Item | Current setting |
|---|---|
| Al lattice constant used to build starting cells | `a0 = 4.039848 A` |
| Main vacancy supercell | conventional fcc `3x3x3`, `108 -> 107` atoms |
| Main divacancy supercell | conventional fcc `3x3x3`, `108 -> 106` atoms |
| `3x3x3` starting cell length | `12.119544 A` in each direction |
| Cell-size interpretation | cell sides exceed `10 A`; if professor means `vacancy-vacancy distance r > 10 A`, prepare `3x3x6` |
| DFTpy XC/pseudo | `LDA`, `al.lda.recpot` |
| DFTpy TFvW pilot setting | `kedf_x/lambda = 1.0`, `kedf_y/mu = 0.13`; not final because lattice constant was not jointly calibrated |
| DFTpy spacing | `0.20 A` |
| DFTpy relaxation | full atom+cell relaxation / vc-relax equivalent |
| DFTpy optimizer default | ASE `BFGS`; fallback options exist, but do not use FIRE |
| QE relaxation | `vc-relax` for pristine and defect |
| QE functional/pseudo | PBE, `Al_PAW_PBE.UPF` |

## Calculation Families

### 1. DFTpy TFvW Fine Weight Scan

Purpose: calibrate the von Weizsaecker weight for the `3x3x3` single-vacancy
cell after TFvW with `x=1,y=1` overestimated vacancy energies.

Key reference:

```text
DFTPY_TFVW_FINE_WEIGHT_SCAN_20260608.md
```

Important result:

| y/mu | Ef_vac (eV) | force status |
|---:|---:|---|
| `0.130` | `0.603451` | pristine and vacancy pass `fmax < 0.002 eV/A` |

Treat `x/lambda = 1.0`, `y/mu ~= 0.13` as a historical pilot setting only.
The final pair must be selected jointly from vacancy formation energy, relaxed
lattice constant, forces, and stress.

Main scripts:

| Purpose | Script |
|---|---|
| Prepare/submit weight scan | `scripts/push_dftpy_tfvw_weight_scan_to_iservice_20260605.sh` |
| Submit prepared scan chunks | `scripts/submit_existing_dftpy_tfvw_weight_scan_to_iservice_20260605.sh` |
| Pull results | `scripts/pull_dftpy_tfvw_weight_scan_results_20260605.sh` |
| Summarize results | `scripts/summarize_tfvw_fine_scan_20260608.py` |

### 2. 3x3x3 Lambda/Mu 100-Point Matrix

Purpose: scan both TF and vW weights instead of only fixing TF at 1.0.  The
professor requested total energy, KEDF energy, and lattice constant maps/tables.

Key local analysis package:

```text
C:\Users\dawso\Desktop\DFTpy_final_professor_20260611\analysis_final_professor_simple_100pt
```

Key pulled/packaged files:

```text
summary_long.csv
tfvw_lambda_mu_100pt_results.xlsx
tfvw_lambda_mu_100pt_20260616_gmail_safe.zip
```

Main scripts:

| Purpose | Script |
|---|---|
| Collect vacancy matrix | `scripts/collect_dftpy_tfvw_lambda_mu_vacancy_matrix.py` |
| Pull matrix from NCHC | `scripts/pull_dftpy_tfvw_lambda_mu_vacancy_matrix_20260610.sh` |
| Build professor workbook/package | `scripts/build_tfvw_lambda_mu_professor_package_20260610.py` |
| Plot professor figures | `scripts/plot_dftpy_tfvw_lambda_mu_professor_figures_20260611.py` |
| Plot clean/simple valid maps | `scripts/plot_dftpy_tfvw_lambda_mu_extreme_simple_valid_20260611.py` |
| Reconstruct missing pristine table points | `scripts/reconstruct_pristine_table_missing_points_20260611.py` |

Notes:

- The final run reached `100/100 result.json`.
- Quality flags and timeout markers should be retained in the analysis package.
- For professor plots, use valid points only unless explicitly showing missing/invalid cases.
- Formation-energy maps should plot vacancy formation energy, not raw pristine
  total energy. KEDF and lattice constant maps can be direct relaxed-pristine
  values if that is the requested table.

### 2a. Single-Vacancy Lambda/Mu Fine Scan

The 42-point refinement scans `lambda=0.90-0.95` and `mu=0.04-0.10` for the
same conventional `3x3x3`, `108 -> 107` cell. NCHC job `1569500` was submitted
on 2026-06-22 as three workers with at most two concurrent workers.

This scan is independent of the numerical QE target during execution. After
completion, re-rank the same DFTpy results against the corrected QE reference
described below; no DFTpy rerun is needed solely because the reference changed.

### 2b. Corrected QE Single-Vacancy Reference

The corrected KSDFT source is the conventional cubic `3x3x3`, `108 -> 107`,
QE/PBE `vc-relax` workflow at `800 eV`:

| k mesh | Ef_vac (eV) |
|---|---:|
| `3x3x3` | `0.644941904` |
| `4x4x4` | `0.677874700` |
| `5x5x5` | `0.638912226` |

Use `0.638912226 eV` as the best completed dense-k point and retain
`0.638912-0.677875 eV` as the visible k-point sensitivity range. The old
`0.601167 eV` target belongs to the historical conventional `2x2x4`,
`64 -> 63` dataset and must not be labeled as the corrected `3x3x3` reference.

Reference documentation and package builder:

| Purpose | File |
|---|---|
| Reference definition | `QE_SINGLE_VACANCY_REFERENCE.md` |
| Build complete input/output dossier | `scripts/build_qe_single_vacancy_reference_package.py` |
| Collect energies and actual final atomic fmax | `scripts/collect_qe_vcrelax_vacancy.py` |

### Submission Gate

Before submitting or packaging a vacancy workflow, run:

```bash
python scripts/audit_vacancy_submission_gate.py \
  --dftpy-root /path/to/prepared_dftpy_series \
  --qe-root /path/to/prepared_or_completed_qe_series
```

Add `--require-qe-outputs` when auditing completed QE data. The gate checks
atom counts, centered vacancy, cell dimensions, short contacts, `vc-relax`,
lambda/mu consistency, pseudo availability, submit-script threading, and raw
output completion. Do not submit or package a series when the status is
`FAIL`.

For a series prepared before case-local pseudopotentials and README files were
added, materialize them after pulling the raw results:

```bash
python scripts/materialize_dftpy_case_reproducibility.py \
  --rootdir /path/to/pulled_series \
  --pp /path/to/al.lda.recpot
```

The DFTpy INI `Optdensity` task is the inner electronic-density optimization.
The outer full atom-and-cell relaxation is performed by
`run_dftpy_vcrelax_vacancy_one.py` with ASE `FrechetCellFilter` and `BFGS`.

### 3. DFTpy Divacancy r-Scan

Purpose: respond to the request for two vacancies at the same height, scan the
vacancy-vacancy distance `r`, and evaluate `E_2vac(r)`.

Current completed package:

```text
C:\Users\dawso\Desktop\DFTPY_DIVACANCY_TO_PROF_20260616_gmail_safe.zip
```

Current local raw directory:

```text
C:\Users\dawso\Desktop\DFTPY_DIVACANCY_RSCAN_20260616\raw\dftpy_divacancy_vcrelax_conv3x3x3_rscan_20260616
```

Current remote source directory:

```text
/work/dawson666/dftpy_project/relax/dftpy45/results/dftpy_divacancy_vcrelax_conv3x3x3_rscan_20260616
```

DFTpy result summary:

| r (A) | E_2vac (eV) | E_2vac/2 (eV) | E_2vac - farthest (eV) | force status |
|---:|---:|---:|---:|---|
| 2.8566 | 1.151372 | 0.575686 | -0.059467 | pass |
| 4.0398 | 1.191490 | 0.595745 | -0.019348 | pass |
| 5.7132 | 1.206444 | 0.603222 | -0.004394 | pass |
| 6.3876 | 1.199246 | 0.599623 | -0.011592 | pass |
| 8.5698 | 1.210838 | 0.605419 | 0.000000 | pass |

Interpretation:

- The shortest pair is lower by about `0.059 eV` relative to the farthest
  sampled pair, suggesting short-range attraction/clustering.
- Larger `r` cases approach about `0.60 eV` per vacancy.
- Geometry audit shows final cell lengths remain above `12.60 A` and no obvious
  vacancy collapse.

Main scripts:

| Purpose | Script |
|---|---|
| Build same-height vacancy-pair structures | `scripts/prepare_al_double_vacancy_pair_structures.py` |
| Prepare DFTpy pair scan | `scripts/prepare_dftpy_divacancy_rscan_20260616.py` |
| Run DFTpy pair scan locally | `scripts/run_local_dftpy_divacancy_rscan_20260616.ps1` |
| Push DFTpy pair scan to NCHC | `scripts/push_dftpy_divacancy_rscan_to_iservice_20260616.sh` |
| Pull DFTpy/QE pair results | `scripts/pull_divacancy_rscan_results_20260616.sh` |

Professor package contents:

| Folder | Contents |
|---|---|
| `00_README` | README and short email draft |
| `01_SUMMARY_TABLES` | source dirs, computational details, summaries, geometry audit |
| `02_FIGURES` | formation-energy and binding plots |
| `03_RAW_CASES` | VASP structures, DFTpy outputs, relax logs, result.json |
| `04_SCHEDULER_LOGS` | Slurm logs for job `1533898` |
| `05_REPRODUCIBILITY` | pseudo and scripts-as-text for Gmail safety |

### 4. QE Divacancy r-Scan

Purpose: produce the QE/PBE `vc-relax` reference for the same divacancy `r` scan.

Current remote run:

```text
/work/dawson666/qe_cases/qe_runs/qe_divacancy_vcrelax_conv3x3x3_rscan_20260616
```

Current job:

```text
1553956_[0-4] QEVCR3 on ct56
```

Current status as of 2026-06-18:

- All five QE array tasks are running.
- `pw.x` path and runtime library issue were fixed.
- Working `pw.x` path:

```text
/work/dawson666/q-e-qe-7.3.1/PW/src/pw.x
```

Main scripts:

| Purpose | Script |
|---|---|
| Prepare QE divacancy pair scan | `scripts/prepare_qe_divacancy_vcrelax_rscan_20260616.py` |
| Push and submit QE divacancy scan | `scripts/push_qe_divacancy_rscan_to_iservice_20260616.sh` |
| Upload QE package without submitting | `scripts/upload_qe_divacancy_package_to_work_20260616.sh` |
| Pull QE/DFTpy results together | `scripts/pull_divacancy_rscan_results_20260616.sh` |
| Collect QE vc-relax output | `scripts/collect_qe_vcrelax_vacancy.py` |

After QE finishes:

```bash
cd /work/dawson666/qe_cases/qe_runs/qe_divacancy_vcrelax_conv3x3x3_rscan_20260616
python scripts/collect_qe_vcrelax_vacancy.py --rootdir .
```

Then pull both methods locally:

```bash
cd /mnt/c/Users/dawso/nano_tensile_TFvW

DFTPY_SERIES=dftpy_divacancy_vcrelax_conv3x3x3_rscan_20260616 \
QE_SERIES=qe_divacancy_vcrelax_conv3x3x3_rscan_20260616 \
bash scripts/pull_divacancy_rscan_results_20260616.sh
```

## Current Queue/Partition Rules

| Partition | Current usage |
|---|---|
| `ctest` | short DFTpy scans, target under 2 hours, usually `%2` |
| `ct56` | QE production, 4-day walltime |

For divacancy QE, the generated default array throttle is now `%5` so all five
pair distances can run when policy/resources permit.

## Common Failure Modes Fixed

| Symptom | Cause | Fix |
|---|---|---|
| `mkdir /gpfs-work/dawson666 permission denied` | NCHC work root moved | use `/work/dawson666` |
| QE job fails in 0 seconds, no QE output | submit script ran in Slurm spool dir | use `SLURM_SUBMIT_DIR` as `ROOT` |
| QE `execvp error ... /gpfs-home/.../bin/pw.x` | old QE binary path | use `/work/dawson666/q-e-qe-7.3.1/PW/src/pw.x` |
| QE runtime library errors | missing compatible Fortran runtime | prepend `/home/dawson666/miniconda3/envs/abinit-env/lib` and preload `libgfortran.so.5.0.0` |
| Gmail blocks package | executable/script attachment | rename scripts to `.txt` in professor package |

## Current Next Steps

1. Wait for QE job `1553956_[0-4]` to finish.
2. Collect QE output with `scripts/collect_qe_vcrelax_vacancy.py`.
3. Pull with `scripts/pull_divacancy_rscan_results_20260616.sh`.
4. Build a combined QE vs DFTpy `E_2vac(r)` plot.
5. If the professor confirms that `r` itself must exceed `10 A`, prepare a
   longer `3x3x6` divacancy scan.

## Advisor Feedback From 2026-06-18

Professor Lueder requested a cleanup before the next submission:

- Finish and evaluate the single-vacancy `lambda/mu` calibration before treating
  divacancy/nanostructure data as final.
- Add QE/DFT reference values.
- Put DFTpy input/output files directly inside the raw calculation folders.
- Add source directories and critical computational details to spreadsheets.
- Add compact evaluation slides or notebooks.
- Clarify `divacancy_start.vasp` vs `vacancy_start.vasp`.
- Explain or recheck the small divacancy trend drop near `6-7 A`.
- Clarify that reported `r` values are minimum-image distances under PBC.
- Avoid redundant main plots showing both `E_2vac` and `E_2vac/2`.

The detailed action list and draft reply are in:

```text
ADVISOR_RESPONSE_ACTIONS_20260618.md
```

## Monday 2026-06-22 Pipeline Update

New or corrected workflow components:

| Purpose | File |
|---|---|
| Energy, binding, and PBC definitions | `DIVACANCY_ENERGY_AND_PBC_DEFINITIONS.md` |
| Generate structure notebook | `notebooks/01_generate_divacancy_structures.ipynb` |
| Verify DFTpy formation energy notebook | `notebooks/02_read_dftpy_outputs_compute_formation_energy.ipynb` |
| Compare DFTpy and QE notebook | `notebooks/03_compare_dftpy_qe_divacancy.ipynb` |
| Geometry and strain-proxy audit | `scripts/analyze_divacancy_geometry_strain.py` |
| Rebuild professor package | `scripts/build_divacancy_professor_package.py` |
| Prepare 42-point single-vacancy fine scan | `scripts/prepare_dftpy_tfvw_lambda_mu_vacancy_fine_scan.py` |
| Push fine scan to NCHC | `scripts/push_dftpy_tfvw_lambda_mu_vacancy_fine_scan.sh` |

Corrected divacancy direction table:

| Initial r (A) | Direction family | DFTpy pilot E_2vac (eV) |
|---:|---|---:|
| 2.8566 | `[110]` | 1.151372 |
| 4.0398 | `[100]` | 1.191490 |
| 5.7132 | `[110]` | 1.206444 |
| 6.3876 | `[310]` | 1.199246 |
| 8.5698 | `[110]` | 1.210838 |

The apparent dip at `6.3876 A` is not part of a single fixed-direction trend;
the point changes from `[110]` to `[310]`. Future plots must show direction.

The old divacancy pilot used `(lambda,mu)=(1.0,0.13)`. It remains pilot data
because that setting matches the single-vacancy formation energy but does not
simultaneously match the relaxed lattice constant. The next calibration scan is:

```text
lambda = 0.90, 0.91, 0.92, 0.93, 0.94, 0.95
mu     = 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10
points = 42
```
