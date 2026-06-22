# Advisor Response Action Plan

Date: 2026-06-18

## Progress Update: 2026-06-22

Completed locally:

- DFTpy pair scans now use explicit `divacancy_*` input/output names.
- Each new case contains readable DFTpy `.ini` provenance inputs, actual
  calculator-config JSON files, outputs, relax logs, final structures, and a
  case README.
- DFTpy Slurm defaults are now one task, one CPU, and one BLAS/OpenMP thread.
- The QE divacancy generator now writes a concise job script and uses the
  explicit `divacancy_vcrelax` directory name.
- Three workflow notebooks were added under `notebooks/`.
- The old DFTpy pilot package was rebuilt successfully with `5/5` complete
  directly discoverable case folders.
- Geometry/strain audit shows the `6.3876 A` point is force converged and has no
  obvious displacement/strain anomaly.

Important interpretation found on 2026-06-22:

- The old scalar-distance plot mixed crystallographic directions:
  `[110]`, `[100]`, `[110]`, `[310]`, `[110]`.
- The small drop near `6.39 A` occurs at the `[310]` point. It must be discussed
  as orientation dependence, not as a break in a single `[110]` radial trend.
- The `8.5698 A` point is the `[110]` vector `(L/2,L/2,0)` in a cubic cell with
  `L ~= 12.12 A`; it is valid under the minimum-image convention.
- The complete 10x10 single-vacancy matrix shows that `(lambda,mu)=(1,0.13)`
  matches the vacancy energy but does not match the relaxed lattice constant.
- A 42-point fine scan has therefore been prepared for
  `lambda=0.90-0.95`, `mu=0.04-0.10` before any final divacancy rerun.

This file translates Professor Lueder's 2026-06-18 feedback into concrete
calculation, packaging, and reporting actions.

## Immediate Interpretation

The main message is not that the divacancy calculations are useless. The
message is that the delivery order and provenance were not clear enough.

Priority order now:

1. Finish and document the single-vacancy `lambda/mu` scan.
2. Add the QE/DFT reference comparison and choose the calibrated `lambda/mu`.
3. Only then evaluate whether the current divacancy/nanostructure settings
   should be repeated with the final calibrated parameters.
4. Rebuild the data package so every case folder contains discoverable input,
   output, source directory, and critical computational details.
5. Add short evaluation slides/notebooks before sending another large package.

## Questions

### Q1. Why divacancy calculations before mu/lambda scan is complete?

Answer:

The divacancy run was an exploratory pipeline test while the calibrated
single-vacancy reference was still being finalized. It should not be presented
as the final production result yet.

Action:

- Mark current divacancy results as `pilot / pipeline validation`.
- Do not use them as final conclusions until the `lambda/mu` single-vacancy
  calibration and QE comparison are complete.
- If final calibrated `lambda/mu` differs from the divacancy run settings,
  repeat the divacancy scan.

### Q2. Difference between `divacancy_start.vasp` and `vacancy_start.vasp`

Current status:

In the DFTpy divacancy preparation script, both files are the same 106-atom
two-vacancy starting structure:

```text
vacancy_start.vasp    = backward-compatible generic defect start file
divacancy_start.vasp  = explicit two-vacancy start file
```

This naming is confusing.

Action:

- For future divacancy packages, keep `divacancy_start.vasp` as the main file.
- Either remove `vacancy_start.vasp` from professor-facing packages or add a
  local `README_CASE.txt` in each folder explaining that it is an alias.

### Q3. Why drop in trend between 6-7 A?

Current DFTpy pilot result:

```text
r = 5.7132 A: E_2vac = 1.206444 eV
r = 6.3876 A: E_2vac = 1.199246 eV
r = 8.5698 A: E_2vac = 1.210838 eV
```

The dip at `r = 6.3876 A` is about `0.0072 eV` relative to `5.7132 A`, and
about `0.0116 eV` below the farthest sampled point.

Possible explanations:

- Small residual finite-cell / periodic-image interaction in a `3x3x3` cell.
- Different local relaxation basin around the second vacancy.
- Numerical residual from vc-relax and stress/shape relaxation.
- Real weak short-range divacancy interaction, but this cannot be claimed
  without QE comparison and/or larger-cell confirmation.

Action:

- Do not over-interpret the dip yet.
- Add per-case final forces, stress, final vacancy-vacancy distance, and
  nearest-neighbor geometry checks.
- Compare against QE after QE finishes.
- If the dip remains suspicious, rerun the same `r` point and/or run a
  `3x3x6` check.

### Q4. How can r > 8 A when simulation cell is < 13 A?

Answer:

The reported `r` is the minimum-image vacancy-vacancy separation under periodic
boundary conditions. In a cubic cell of side about `12.12 A`, the maximum
minimum-image distance can be larger than `L/2` when the displacement includes
multiple Cartesian components, up to about `sqrt(3) * L/2`.

For `L = 12.12 A`, this geometric upper bound is about:

```text
sqrt(3) * 12.12 / 2 ~= 10.5 A
```

So `r = 8.57 A` is geometrically possible in a `3x3x3` periodic cell.

Action:

- Add the displacement vector `(dx, dy, dz)` and note `minimum-image distance`
  explicitly in the table.
- Clarify whether Professor's `>10 A` criterion means cell length or
  vacancy-vacancy distance. The current `3x3x3` cell length exceeds `10 A`;
  the sampled pair distance does not exceed `10 A`.
- If he wants actual vacancy-vacancy distances above `10 A`, prepare `3x3x6`.

## Comments

### C1. DFTpy output is missing / hard to find

Action:

- Rebuild the package so each `03_RAW_CASES/pair_xx.../` folder contains:
  - `README_CASE.txt`
  - `manifest.json`
  - `dftpy_pristine_input.ini`
  - `dftpy_vacancy_input.ini` or `dftpy_divacancy_input.ini`
  - `pristine_relax.log`
  - `vacancy_relax.log` / `divacancy_relax.log`
  - `pristine_dftpy.out`
  - `vacancy_dftpy.out` / `divacancy_dftpy.out`
  - final relaxed VASP structures
  - `result.json`
- Keep scheduler logs separately, but do not rely on them as the only place to
  find DFTpy output.

### C2. Evaluation slides are missing

Action:

- Build a compact evaluation deck after QE divacancy finishes.
- Keep it to 5-7 slides:
  1. Objective and status.
  2. Single-vacancy `lambda/mu` calibration.
  3. QE reference comparison.
  4. Divacancy pilot geometry and definitions.
  5. Divacancy energy vs distance.
  6. Strain-field visualization/discussion.
  7. Open issues and next calculations.

### C3. Difference between `dftpy_divacancy_Ef...` and binding table

Definitions to include in package:

```text
E_2vac(r) = E_defect(N-2, r) - ((N-2)/N) * E_pristine(N)
```

This is the two-vacancy formation energy.

```text
E_2vac_per_vacancy(r) = E_2vac(r) / 2
```

This is only a normalized plotting value and is redundant if `E_2vac` is shown.

```text
E_bind(r) = E_2vac(r) - 2 * E_1vac
```

or, if using the farthest sampled pair as the zero reference:

```text
Delta E_2vac(r) = E_2vac(r) - E_2vac(r_farthest)
```

Action:

- Do not show both `E_2vac` and `E_2vac/2` in the same main figure unless
  necessary.
- Rename tables clearly:
  - `divacancy_formation_energy_total.csv`
  - `divacancy_relative_energy_vs_farthest.csv`
  - `divacancy_binding_energy_requires_single_vacancy_reference.csv`

### C4. DFT reference values missing

Action:

- Wait for the five QE divacancy jobs to finish.
- Collect QE with `scripts/collect_qe_vcrelax_vacancy.py`.
- Add QE/PBE vc-relax data to the same r-scan table.
- If QE remains too slow or problematic, ask Steven/Slava for guidance, but
  first collect all completed QE outputs and identify exact failure modes.

### C5. Otherwise quite good

Interpretation:

The calculation direction is acceptable, but packaging and evaluation order need
to be cleaned up.

### C6. Why set OMP/MKL/OPENBLAS threads to CPUS_PER_TASK?

Current concern:

For DFTpy/NumPy/BLAS jobs, setting all thread variables to the full Slurm CPU
count can oversubscribe if the calculation also uses multiprocessing or nested
threading.

Action:

- For single-process DFTpy jobs on `ctest`, use controlled threading:

```bash
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
```

- If benchmarking shows a single DFTpy process benefits from 2-8 threads, set it
  explicitly and document it.
- Do not leave unexplained thread settings in professor-facing scripts.

### C7. Per-vacancy and two-vacancy formation energy are redundant

Action:

- Main plot: show `E_2vac(r)` or `Delta E_2vac(r)`, not both total and
  per-vacancy.
- Optional table may include per-vacancy as secondary context.

### Q5. What is folder `05_REPRODUCIBILITY`?

Answer:

It was a Gmail-safe archive of scripts and pseudopotentials used to recreate
the run. The better long-term delivery should be GitHub plus notebooks.

Action:

- Rename package folder to `05_CODE_POINTERS`.
- Include:
  - GitHub branch/commit.
  - Notebook path.
  - Minimal script list.
  - Pseudopotential hash/path.
- Add Jupyter notebooks:
  1. `notebooks/01_generate_divacancy_structures.ipynb`
  2. `notebooks/02_read_dftpy_outputs_compute_formation_energy.ipynb`
  3. `notebooks/03_compare_dftpy_qe_divacancy.ipynb`

## Next Required Work

### A. Complete single-vacancy `lambda/mu` scan

Status:

- 100/100 DFTpy result set exists.
- Need final professor-facing spreadsheet and figures with source dirs and
  computational details.

Action:

- Confirm source directory and critical details are written directly in the
  spreadsheet.
- Confirm final chosen `lambda/mu` is based on QE reference, not just DFTpy
  internal smoothness.

### B. Add QE/DFT reference

Status:

- QE single-vacancy reference exists from previous vc-relax work.
- QE divacancy r-scan is running.

Action:

- Pull and collect QE when job `1553956` finishes.
- Add QE values to comparison table and plots.

### C. Repeat divacancy if final `lambda/mu` differs

Action:

- If selected `lambda/mu` is not `(1.0, 0.13)`, repeat:
  - DFTpy single vacancy at final parameters.
  - DFTpy divacancy r-scan at final parameters.

### D. Evaluate `lambda/mu` before nanostructure

Action:

- Pause nanostructure production until single-vacancy calibration is accepted.
- Keep previous nanostructure/profess tests as exploratory only.

### E. Upload to NAS

Action:

- Update browser or use alternate upload method.
- If NAS blocks compressed/script files, use Gmail-safe package naming or upload
  data-only zip and GitHub commit link separately.

### F. Add strain-field discussion

Action:

- After full relaxation, compute displacement vectors from pristine to relaxed
  defect structure.
- Plot or tabulate:
  - atom displacement magnitude vs distance from vacancy pair center
  - local strain proxy
  - nearest-neighbor relaxations around vacancies
- Do this for final calibrated DFTpy and QE if possible.

## Short Reply Draft

```text
Dear Professor,

Thank you for the detailed feedback. I agree that the current divacancy data
should be treated as a pilot/pipeline test until the single-vacancy lambda/mu
calibration and QE reference comparison are finalized.

I will first complete and document the single-vacancy lambda/mu scan, add the
QE reference values, and only then decide whether the divacancy scan must be
repeated with the final calibrated parameters.

For the current divacancy package, I will also rebuild the data layout so each
case folder directly contains the DFTpy input/output files, source directory,
manifest, computational details, and definitions of the formation/binding-energy
tables. I will also add compact evaluation slides and notebooks showing how the
structures are generated, how DFTpy outputs are read, and how formation energies
are computed.

Regarding the >8 A pair distance in a ~12 A cell, the reported value is the
minimum-image vacancy-vacancy distance under periodic boundary conditions. I
will make this explicit by adding the displacement vectors and minimum-image
definition. If the intended requirement is vacancy-vacancy separation above
10 A rather than cell length above 10 A, I will prepare a 3x3x6 check.

Best regards,
Dawson
```
