# iservice Al_defects Snapshot Check - 2026-06-23

Local snapshot checked:

```text
C:\Users\dawso\Desktop\ISERVICE_AL_DEFECTS_SNAPSHOT_20260623_160121\dftpy45_snapshot
```

## Snapshot Summary

- `results/Al_defects` was pulled successfully with the new focused snapshot script.
- Local audit file:
  `C:\Users\dawso\Desktop\ISERVICE_AL_DEFECTS_SNAPSHOT_20260623_160121\dftpy45_snapshot\_audit\LOCAL_AUDIT.md`
- Snapshot counts:
  - `result.json`: 17
  - `run_status*.json`: 25
  - failed markers: 0
  - running locks: 3
  - logs with invalid timeout interval: 25

## Invalid Attempt: Job 1575328

Job `1575328` is not a physics or convergence result.

All 25 inspected tasks failed before entering the DFTpy Python runner:

```text
timeout: invalid time interval
python_or_timeout_rc = 125
has_result_json = false
status = python_failed
```

Interpretation:

```text
1575328 = invalid submission attempt caused by shell timeout formatting.
Do not include these tasks in scientific analysis.
```

## Current Running Attempt: Job 1575470

Remote `squeue` reported:

```text
1575470_0-4 running on ct56
```

The snapshot contains log files for:

```text
logs/Al_defects/vacancy_lammu_matrix_ct56/VACLM_0_4_1575470_0.out
logs/Al_defects/vacancy_lammu_matrix_ct56/VACLM_0_4_1575470_1.out
logs/Al_defects/vacancy_lammu_matrix_ct56/VACLM_0_4_1575470_2.out
logs/Al_defects/vacancy_lammu_matrix_ct56/VACLM_0_4_1575470_3.out
logs/Al_defects/vacancy_lammu_matrix_ct56/VACLM_0_4_1575470_4.out
```

All corresponding `.err` files were 0 bytes in the snapshot.

These tasks are the first five coarse single-vacancy lambda-mu points:

| task | setting | lambda | mu |
|---:|---|---:|---:|
| 0 | `tfvw_lam0p1_mu0p1` | 0.1 | 0.1 |
| 1 | `tfvw_lam0p1_mu0p2` | 0.1 | 0.2 |
| 2 | `tfvw_lam0p1_mu0p3` | 0.1 | 0.3 |
| 3 | `tfvw_lam0p1_mu0p4` | 0.1 | 0.4 |
| 4 | `tfvw_lam0p1_mu0p5` | 0.1 | 0.5 |

The submit script used by job `1575470` has the corrected timeout form:

```bash
timeout --signal=TERM --kill-after=120s 171000s \
  python scripts/run_dftpy_vcrelax_vacancy_matrix_one.py ...
```

Therefore, unlike job `1575328`, job `1575470` is actually running DFTpy.

## Early Runtime Observations

The snapshot was taken while the tasks were still running, so no `result.json`
for these five points should be expected yet.

Current logs show active density optimization and MDMin relaxation output.

Notable risk:

- `tfvw_lam0p1_mu0p1` shows extremely large stress and exploding ionic force in the partial `pristine_relax.log`.
- This likely indicates a numerically unstable or physically unreasonable low-lambda/low-mu point.
- It should be marked unqualified if it fails to converge or produces pathological stress/cell behavior.

Other points in the same row appear less pathological in the snapshot, but they were still incomplete.

## Existing Completed Results in Snapshot

The 17 existing `result.json` files are from mixed categories:

- 10 bulk lambda-mu results.
- 2 single-vacancy fine-scan results.
- 5 preliminary divacancy r-scan results.

The currently running coarse vacancy matrix does not yet have completed `result.json` files in this snapshot.

## Recommended Next Actions

1. Do not analyze `1575328` as scientific data.
2. Let `1575470_0-4` continue unless a task clearly runs to walltime with pathological output.
3. After `1575470` finishes or times out, pull another focused snapshot.
4. Classify each completed point as:
   - completed and qualified,
   - completed but unqualified by force/stress/cell quality,
   - timeout,
   - failed before runner,
   - failed during runner.
5. Only use qualified points for professor-facing contour plots and tables.

