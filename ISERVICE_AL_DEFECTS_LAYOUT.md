# NCHC Al defects canonical layout

Canonical project area created on 2026-06-22:

```text
/work/dawson666/dftpy_project/relax/dftpy45/results/Al_defects
```

Important active paths:

| Purpose | Canonical path relative to `results/Al_defects` |
|---|---|
| Single-vacancy lambda/mu fine scan | `01_calibration/single_vacancy/dftpy_tfvw_lambda_mu/fine_L0p90-0p95_M0p04-0p10` |
| Corrected QE/PBE references | `02_reference_dft/qe_pbe` |
| Preliminary DFTpy divacancy scan | `03_defect_cases/divacancy/dftpy_tfvw/preliminary_r_scan_L1p00_M0p13` |
| Analysis outputs | `04_analysis` |
| Professor delivery | `05_reporting/professor_delivery` |
| Reproducibility material | `06_reproducibility` |
| Audits | `07_audit` |

Legacy paths are symlinks only. New scripts use canonical absolute `SERIES_DIR`
values so nested series directories work without depending on a flat series
name. Keep the legacy symlinks until every active Slurm job and old notebook is
retired.

The file `settings_missing_after_1569500.txt` is a point-in-time migration
snapshot. It is not a live completion report. Use:

```bash
cd /work/dawson666/dftpy_project/relax/dftpy45
bash scripts/check_iservice_al_defects_status.sh
```
