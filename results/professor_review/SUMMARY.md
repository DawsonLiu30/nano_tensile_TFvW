# Current professor-review results

For complete raw calculation paths, see `CALCULATION_DATA_MAP.md`.

## Bulk validation

- `00_bulk_DFT_vs_OFDFT_QE_DFTpy.png`
- QE / KS-DFT fitted a0: 4.039721 A
- DFTpy / OF-DFT TFvW fitted a0: 4.118921 A
- OF-DFT vs QE relative difference: 1.9605 %

This is the preferred bulk figure because it compares DFT and OF-DFT directly.

## Vacancy finite-grip tensile test

- `01_r1_finite_grip_vacancy_tensile_FINAL.png`
- `01_r1_finite_grip_vacancy_tensile_FINAL.csv`
- `02_r2_finite_grip_vacancy_tensile_FINAL.png`
- `02_r2_finite_grip_vacancy_tensile_FINAL.csv`

Status: **invalidated; do not use for professor review.**

These curves were generated with an affine stretch that also elongated the fixed grip regions. A real tensile fixture should translate the top and bottom grips as rigid bodies, then relax only the mobile atoms. The tensile workflow has been corrected after this issue was identified on 2026-04-20, and these finite-grip curves must be rerun from cycle 0.

Workflow summary:

- Finite Al [111] nanowire, not periodic cell stretching.
- Bottom and top grip atoms are translated as rigid bodies during loading, then fixed during relaxation.
- A single vacancy is placed in the middle free region, outside the fixed grips and close to the surface.
- Each tensile cycle displaces the rigid grips by 1 percent in z, then relaxes only the non-grip region.
- Stress is computed from grip reaction force divided by the nanowire reference cross-section.

Invalidated r = 1.0 nm result:

- Completed cycle: 20 of 20
- Final engineering strain: 22.0190 %
- Final grip reaction stress: 14.1593 GPa
- Peak grip reaction stress: 23.3966 GPa at cycle 11, strain 11.5668 %
- Fracture check was incomplete because it used only the free-region z-gap and could miss grip/free-interface separation.
- Maximum free-region z-gap: 2.2014 A; threshold: 7.1341 A.

Invalidated r = 2.0 nm result:

- Completed cycle: 20 of 20
- Final engineering strain: 22.0190 %
- Final grip reaction stress: 7.9112 GPa
- Peak grip reaction stress: 26.6030 GPa at cycle 17, strain 18.4304 %
- Fracture check was incomplete because it used only the free-region z-gap and could miss grip/free-interface separation.
- Maximum free-region z-gap: 2.6141 A; threshold: 7.1341 A.

No corrected finite-grip vacancy tensile result is currently approved for use. Restart r = 1 and r = 2 from cycle 0 with the corrected rigid-grip loading code before extending to r = 3 and r = 4.
