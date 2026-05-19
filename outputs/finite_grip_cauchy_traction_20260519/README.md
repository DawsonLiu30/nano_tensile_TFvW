# Finite-Grip Cauchy-Traction Postprocessing

Date: 2026-05-19

This folder contains the corrected postprocessing for the historical
finite-grip vacancy tensile data used in the proposal-stage figures.

## Scope

These are legacy finite-grip data, not the current axially periodic prism
tensile workflow. They are retained only to explain and defend the old proposal
figures.

The main finite-grip y-axis is:

```text
grip_nominal_primary_GPa
```

This is the preload-corrected nominal grip-reaction stress. It should not be
replaced by the raw simulation-cell stress, because the finite-grip cells
contain large vacuum regions and the cell stress is vacuum-diluted.

The current-area corrected stress is:

```text
grip_apparent_cauchy_primary_GPa
```

It is an apparent Cauchy-stress sensitivity check, not the primary reported
curve. The atomic-scale current cross-sectional area can become ambiguous after
surface relaxation, vacancy distortion, or necking.

## Script

Generated with:

```text
scripts/add_finite_grip_cauchy_traction_columns.py
```

Source summaries:

```text
results/professor_review/ppt_ready/data/r*/summary.csv
```

## Key Outputs

- `finite_grip_nominal_primary_all_radii.png`
  - Combined proposal-data stress-strain plot using the corrected main y-axis.
- `finite_grip_top_bottom_balance_all_radii.png`
  - Diagnostic plot of top/bottom grip-force balance.
- `finite_grip_cauchy_radius_summary.csv`
  - Radius-level summary of maximum stress and force-balance quality.
- `finite_grip_cauchy_all_points.csv`
  - All corrected points in one table.
- `r*_summary_with_grip_cauchy_traction.csv`
  - Corrected per-radius tables.
- `r*_finite_grip_stress_check.png`
  - Per-radius nominal-vs-apparent-Cauchy sensitivity plots.

## Main Finding

For all finite-grip cases, the current-area apparent Cauchy stress nearly
overlaps with the nominal grip-reaction stress. Therefore, the historical
finite-grip mechanical interpretation is insensitive to this correction, and
`grip_nominal_primary_GPa` remains the clean primary y-axis.

## Data-Quality Diagnostic

The top/bottom grip-force balance diagnostic separates the legacy data into
quality groups:

- Clean / usable: `r1`, `r4`, `r5`, `r6`
- Use with caution: `r2`, `r3`
- Limited low-strain reference only: `r8`

The caution is due to large cycle-0 or later top/bottom imbalance, not because
the stress-definition postprocessing failed.

## Recommended Wording

```text
The finite-grip proposal data were reprocessed using a
Cauchy-traction-consistent interpretation of the grip reaction forces. The main
y-axis is the preload-corrected nominal grip-reaction stress, not the raw
vacuum-diluted simulation-cell stress. A current-area apparent Cauchy stress was
also computed as a sensitivity check and nearly overlaps with the nominal
stress, indicating that the main trend is insensitive to the current-area
correction.
```
