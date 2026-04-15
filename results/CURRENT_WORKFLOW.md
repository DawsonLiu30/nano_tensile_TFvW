# Current Workflow Outputs

This repository now keeps only the active `TFVW + periodic [111] nanowire` workflow outputs.

## Core scripts

- `scripts/bulk_validate.py`
- `scripts/prepare_paper_periodic_wire.py`
- `scripts/run_periodic_tensile.py`
- `scripts/run_periodic_series.py`
- `scripts/plot_professor_summary.py`

## Professor-facing summary

- Directory: `results/professor_review`
- Summary page: `results/professor_review/SUMMARY.md`
- Bulk `a0` figure: `results/professor_review/01_bulk_a0_scan.png`
- Bulk validation figure: `results/professor_review/02_bulk_validation.png`
- Overlay figure: `results/professor_review/10_completed_short_wire_overlay.png`
- Strength figure: `results/professor_review/12_peak_strength_vs_diameter.png`

## Bulk validation

- Result directory: `results/bulk_Al_fcc_TFVW_refined_20260414`
- Summary: `results/bulk_Al_fcc_TFVW_refined_20260414/summary.txt`
- `a0` scan figure: `results/bulk_Al_fcc_TFVW_refined_20260414/a0_scan.png`
- Validation figure: `results/bulk_Al_fcc_TFVW_refined_20260414/bulk_validation.png`

Validated bulk references:

- `kedf = TFVW`
- `sampled_minimum_a0_A = 4.120`
- `a0_ref_A = 4.118877004246`
- `stress_slope_GPa = 128.562001853711`

## Completed short-wire runs

- `1.0 nm`: `cases/paper_periodic_111_1.0nm_tfvw/results/paper_r1_short_tfvw_20260413_185333`
- `2.0 nm`: `cases/paper_periodic_111_2.0nm_tfvw/results/paper_r2_short_tfvw_20260413_214559`
- `3.0 nm`: `cases/paper_periodic_111_3.0nm_tfvw/results/paper_r3_short_tfvw_20260414_030219`
- `4.0 nm`: `cases/paper_periodic_111_4.0nm_tfvw/results/paper_r4_short_tfvw_20260414_132005`

## Queue status

- `queue_periodic_series_20260413_222909.log` completed on `2026-04-15 02:28:16 +08:00`
