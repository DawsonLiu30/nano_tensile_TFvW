# DFTpy TFvW Fine Weight Scan

Date: 2026-06-08

## Purpose

After the PROFESS KEDF tests showed that the vacancy formation energy is
strongly KEDF-dependent, the DFTpy-accessible path was tested by tuning the
TFvW `y` parameter while keeping:

- Cell: conventional fcc Al 3x3x3
- Pristine/vacancy atoms: 108 -> 107
- Vacancy concentration: 0.925926 %
- XC: LDA
- Pseudopotential: `al.lda.recpot`
- KEDF: `TFVW`
- Fixed TF weight: `x = 1.0`
- Spacing: `0.20 A`
- Target force: `fmax < 0.002 eV/A`

Remote result path:

```text
/gpfs-work/dawson666/dftpy_project/relax/dftpy45/results/dftpy_vacancy_tfvw_weight_fine_y0115_020_conv3x3x3_lda_20260605_v2/
```

Summary file:

```text
fine_weight_scan_summary_with_actual_fmax.csv
```

## Result

| y | Ef_vac (eV) | pristine fmax (eV/A) | vacancy fmax (eV/A) | force status |
|---:|---:|---:|---:|---|
| 0.115 | 0.540334 | 0.000801 | 0.001685 | pass |
| 0.120 | 0.560416 | 0.000709 | 0.001709 | pass |
| 0.125 | 0.581908 | 0.001360 | 0.001887 | pass |
| 0.130 | 0.603451 | 0.001312 | 0.001468 | pass |
| 0.135 | 0.625118 | 0.000905 | 0.003676 | vacancy fmax high |
| 0.140 | 0.647187 | 0.001064 | 0.001746 | pass |
| 0.145 | 0.669308 | 0.001775 | 0.001953 | pass |
| 0.150 | 0.690674 | 0.000615 | 0.001910 | pass |
| 0.155 | 0.712634 | 0.000823 | 0.000509 | pass |
| 0.160 | 0.734939 | 0.000454 | 0.003138 | vacancy fmax high |
| 0.170 | 0.778002 | 0.000660 | 0.001905 | pass |
| 0.180 | 0.822014 | 0.000659 | 0.001951 | pass |
| 0.200 | 0.908329 | 0.001578 | 0.001197 | pass |

## Interpretation

The vacancy formation energy increases smoothly with the TFvW `y` parameter.
This confirms that the earlier high TFvW result is controlled by the vW weight,
not by a failed relaxation or a structure mismatch.

Useful candidate values:

- `y = 0.130`: Ef = 0.603 eV, all forces pass.
- `y = 0.140`: Ef = 0.647 eV, all forces pass.
- `y = 0.145`: Ef = 0.669 eV, all forces pass.
- `y = 0.150`: Ef = 0.691 eV, all forces pass.

The best current production candidates are `y = 0.140` and `y = 0.145`,
because they sit near the QE/literature vacancy-energy range while satisfying
the force target.

Do not use `y = 0.135` or `y = 0.160` as final points without rerunning, because
the vacancy final force exceeds `0.002 eV/A`.

## Recommended Next Step

Use DFTpy/TFvW with `x = 1.0` and `y = 0.140` or `0.145` for the next vacancy
position/radius screening. If an exact fitted value is needed, run a smaller
scan around:

```text
y = 0.138, 0.140, 0.142, 0.144, 0.146
```

but this is not required before starting the nanocolumn vacancy-position tests.

