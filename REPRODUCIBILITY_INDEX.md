# Reproducibility Index

Date: 2026-05-19

This file maps each active work item to the raw input/output data, preprocessing scripts, postprocessing scripts, and summary outputs. It is intended to answer the request:

- input and output files from the codes
- pre- and post-processing scripts for each work item and figure/table
- report summary with work hypothesis, computational details, results, and summary
- clear locations for data/scripts so the results are reproducible

## Code Snapshot

Repository:

```text
https://github.com/DawsonLiu30/nano_tensile_TFvW
```

Current cleanup branch:

```text
codex/cleanup-core-results
```

Main workflow description:

```text
WORKFLOW.md
```

Core Python package:

```text
app/
```

Curated processed outputs in the repository:

```text
outputs/
```

## Local Raw Data Root

Latest pulled raw data archive on the local workstation:

```text
C:\Users\dawso\Desktop\latest_professor_pull_20260511
```

Older nanostructure QE/OFDFT comparison archive used only for actual-point plotting:

```text
C:\Users\dawso\Desktop\vacancy_qe_ofdft_results_2026-04-25
```

Suggested NUS upload root:

```text
C:\Users\dawso\Desktop\NUS_upload
```

## Work Item Map

### 1. QE Bulk EOS and Bulk Modulus Convergence

Purpose:

```text
Check convergence of fcc Al equilibrium lattice constant and bulk modulus
with respect to QE plane-wave cutoff and k-point density.
```

Raw data:

```text
C:\Users\dawso\Desktop\latest_professor_pull_20260511\qe_bulk_b_convergence_20260506
```

Remote source:

```text
/gpfs-work/dawson666/qe_cases/qe_runs/qe_bulk_b_convergence_20260506
```

Important raw subfolders:

```text
ecut_scan/
kmesh_scan/
final_reference/
processed_bulk_B_convergence/
```

Preparation / collection scripts:

```text
scripts/prepare_qe_bulk_b_convergence.py
scripts/collect_qe_bulk_b_convergence.py
```

Remote analysis scripts are also stored inside the raw archive when present:

```text
parse_qe_bulk_b_convergence.py
parse_final_bulk_reference.py
analyze_qe_bulk_high_kmesh.py
```

Primary processed files:

```text
processed_bulk_B_convergence/qe_bulk_ecut_fit_summary.csv
processed_bulk_B_convergence/qe_bulk_kmesh_fit_summary.csv
processed_bulk_B_convergence/qe_bulk_high_kmesh_fit_summary_with_k20.csv
processed_bulk_B_convergence/qe_bulk_high_kmesh_B0_convergence.png
processed_bulk_B_convergence/qe_bulk_high_kmesh_a0_convergence.png
processed_bulk_B_convergence/qe_bulk_high_kmesh_EOS_curves.png
```

Summary result:

```text
QE bulk B0 is converged with respect to cutoff above about 400-600 eV.
High-k validation up to 80x80x80 gives B0 around 77.96 GPa.
The 40x40x40 result differs from 80x80x80 by about 0.03 GPa.
```

### 2. QE Vacancy Formation Energy Convergence

Purpose:

```text
Check fcc Al vacancy formation energy convergence with respect to QE
cutoff and k-point density.
```

Raw data:

```text
C:\Users\dawso\Desktop\latest_professor_pull_20260511\qe_vacancy_convergence_20260506
```

Remote source:

```text
/gpfs-work/dawson666/qe_cases/qe_runs/qe_vacancy_convergence_20260506
```

Important raw subfolders:

```text
ecut_scan/
kmesh_scan/
dense_k05_ecut_scan/
processed_vacancy_convergence/
```

Preparation / collection / plotting scripts:

```text
scripts/prepare_qe_vacancy_convergence.py
scripts/prepare_qe_vacancy_extra_kmesh.py
scripts/collect_qe_vacancy_convergence.py
scripts/collect_all_qe_vacancy_recursive.py
scripts/plot_true_vacancy_formation_from_pulled_data.py
```

Primary processed files:

```text
processed_vacancy_convergence/qe_vacancy_all_recursive_summary.csv
outputs/true_vacancy_formation_20260518/01_qe_vacancy_kmesh_convergence.png
outputs/true_vacancy_formation_20260518/02_qe_dense_k5_ecut_convergence.png
outputs/true_vacancy_formation_20260518/vacancy_formation_combined_summary.csv
```

Vacancy formation energy formula:

```text
E_f^vac = E_vac^(N-1) - ((N-1)/N) E_pristine^N
```

Summary result:

```text
At k = 2x2x2, cutoff convergence is essentially complete from 400-600 eV.
At ecut = 600 eV, dense k-point values enter the literature-relevant range:
4x4x4 gives about 0.648 eV, 5x5x5 gives about 0.573 eV, and the later
6x6x6 terminal result gives about 0.615 eV if the completed output is included.
```

Important archive note:

```text
The local pulled archive used for the current processed plot does not include
the completed k_06x06x06 vacancy output. If the 6x6x6 point is reported,
re-pull qe_vacancy_convergence_20260506/kmesh_scan/k_06x06x06 and regenerate
processed_vacancy_convergence/qe_vacancy_all_recursive_summary.csv.
```

### 3. DFTpy Vacancy Grid-Spacing Convergence

Purpose:

```text
Check OFDFT/DFTpy vacancy formation energy convergence with respect to
real-space density-grid spacing. DFTpy does not use QE-style k-point sampling.
```

Raw data:

```text
C:\Users\dawso\Desktop\latest_professor_pull_20260511\dftpy_vacancy_convergence_primitive4_qe_a0_20260508
C:\Users\dawso\Desktop\latest_professor_pull_20260511\dftpy_vacancy_convergence_primitive4_20260508
```

Remote source:

```text
/gpfs-work/dawson666/dftpy_project/relax/dftpy45/results/dftpy_vacancy_convergence_primitive4_qe_a0_20260508
/gpfs-work/dawson666/dftpy_project/relax/dftpy45/results/dftpy_vacancy_convergence_primitive4_20260508
```

Preparation / run / collection scripts:

```text
scripts/prepare_dftpy_vacancy_convergence.py
scripts/run_dftpy_vacancy_convergence.py
scripts/collect_dftpy_vacancy_convergence.py
run_dftpy_vacancy_convergence_one_ct56.sbatch
```

Primary processed files:

```text
summary.csv
outputs/true_vacancy_formation_20260518/03_dftpy_spacing_convergence.png
```

Summary result:

```text
DFTpy/TFvW vacancy formation energy is numerically stable with respect to
spacing. Fixed QE-a0 gives about 2.94 eV, while DFTpy own-a0 gives about
3.22 eV. Both are much larger than QE and literature vacancy energies.
```

### 4. DFTpy Primitive Supercell-Size Convergence

Purpose:

```text
Check OFDFT/DFTpy vacancy formation energy versus primitive fcc supercell size
at fixed spacing = 0.20 A and fixed QE a0.
```

Raw data:

```text
C:\Users\dawso\Desktop\latest_professor_pull_20260511\dftpy_vacancy_size_primitive_qe_a0_spacing020_20260511
```

Remote source:

```text
/gpfs-work/dawson666/dftpy_project/relax/dftpy45/results/dftpy_vacancy_size_primitive_qe_a0_spacing020_20260511
```

Preparation / run / collection scripts:

```text
scripts/prepare_dftpy_primitive_size_scan_qea0.py
scripts/run_dftpy_primitive_size_one.py
scripts/collect_dftpy_primitive_size_scan.py
run_dftpy_primitive_size_one_ct56.sbatch
```

Primary processed files:

```text
dftpy_primitive_size_summary.csv
dftpy_primitive_size_summary_with_fmax.csv
outputs/true_vacancy_formation_20260518/04_dftpy_supercell_size_convergence.png
```

Summary result:

```text
Primitive supercell sizes from 4x4x4 to 14x14x14 remain around 2.89-2.95 eV.
The large QE-DFTpy discrepancy is therefore not mainly a grid-spacing or
small-cell artifact; it is likely a limitation of the current TFvW/local
pseudopotential OFDFT setup for localized vacancy energetics.
```

### 5. Vacancy-Containing Faceted Nanostructure Actual Points

Purpose:

```text
Replace the old normalized vacancy plot that artificially forced endpoint
agreement. The retained figure uses only actual data points from the pulled
QE/OFDFT comparison CSV.
```

Input archive:

```text
C:\Users\dawso\Desktop\vacancy_qe_ofdft_results_2026-04-25
```

Postprocessing script:

```text
scripts/plot_vacancy_nanocrystal_actual_points_only.py
```

Processed output:

```text
outputs/vacancy_nanocrystal_actual_points_20260518/
```

Important interpretation:

```text
These are actual vacancy total-energy-per-atom points, not vacancy formation
energies. The archive does not contain complete matching pristine pairs for
each faceted vacancy case, so a true faceted-nanostructure vacancy formation
energy plot requires paired pristine calculations.
```

### 6. Geometry and Vacancy-Formula Audit

Purpose:

```text
Check whether old nanocolumn/nanocrystal structures were finite-length models
and correct the vacancy formation energy reference formula.
```

Audit output:

```text
outputs/nanocolumn_nanocrystal_vacancy_audit_20260518.md
```

Summary result:

```text
The audited structures are periodic along the axial z direction. The maximum
empty z interval was 2.338 A, consistent with interlayer spacing rather than
axial vacuum. The old "vacancy nanocrystal" label should be revised to
"vacancy-containing faceted nanocolumns" or "prismatic nanostructures with
faceted cross sections."
```

### 7. Periodic Nanocolumn / Nanocrystal Tensile Workflow

Purpose:

```text
Maintain the corrected axial-periodic tensile workflow and compute wire Cauchy
stress using current wire cross-section area rather than finite-grip force.
```

Preparation / run scripts:

```text
app/ase_nanocrystal.py
scripts/prepare_paper_periodic_wire.py
scripts/prepare_vacancy_periodic_wire.py
scripts/run_periodic_tensile.py
scripts/run_periodic_series.py
scripts/run_vacancy_periodic_series.py
scripts/add_cauchy_stress_columns.py
```

Main stress conversion:

```text
cauchy_wire_zz_GPa = sigma_cell_zz_GPa * A_cell / A_wire,current
```

## Report Files

Current local presentation candidates:

```text
C:\Users\dawso\OneDrive\Documents\qe_bulk_vacancy.pptx
C:\Users\dawso\OneDrive\Documents\qe_bulk_vacancy_concise_english_report_20260511_comment_response_finalsync.pptx
```

Recommended exported PDF destination:

```text
C:\Users\dawso\Desktop\NUS_upload\reproducibility_package_20260519\04_reports
```

## Suggested NUS Upload Layout

```text
reproducibility_package_20260519/
  00_REPRODUCIBILITY_INDEX.md
  01_REPORT_SUMMARY_20260519.md
  02_code_snapshot/
  03_processed_outputs/
  04_raw_data/
  05_reports/
  PACKAGE_MANIFEST.txt
  file_manifest.csv
```

Build command:

```powershell
python scripts\build_reproducibility_package.py --zip
```

