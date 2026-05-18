# True Vacancy Formation Energy Plots

Date: 2026-05-18

All plots in this folder were generated with pandas/matplotlib from pulled local CSV data, not from manually drawn or reference-normalized curves.

## Input data

- QE vacancy summary: `C:\Users\dawso\Desktop\latest_professor_pull_20260511\qe_vacancy_convergence_20260506\processed_vacancy_convergence\qe_vacancy_all_recursive_summary.csv`
- DFTpy spacing, fixed QE-a0: `C:\Users\dawso\Desktop\latest_professor_pull_20260511\dftpy_vacancy_convergence_primitive4_qe_a0_20260508\summary.csv`
- DFTpy spacing, own-a0: `C:\Users\dawso\Desktop\latest_professor_pull_20260511\dftpy_vacancy_convergence_primitive4_20260508\summary.csv`
- DFTpy primitive size: `C:\Users\dawso\Desktop\latest_professor_pull_20260511\dftpy_vacancy_size_primitive_qe_a0_spacing020_20260511\dftpy_primitive_size_summary.csv`

## Generated figures

- `vacancy_true_formation_summary_2x2.png`
- `01_qe_vacancy_kmesh_convergence.png`
- `02_qe_dense_k5_ecut_convergence.png`
- `03_dftpy_spacing_convergence.png`
- `04_dftpy_supercell_size_convergence.png`

## Key numbers from archive-verified data

QE kmesh convergence at ecut = 600 eV:

```text
kmesh  Ef_vac_eV  pristine_done  vacancy_done
1x1x1  -1.113260           True          True
2x2x2   0.734706           True          True
3x3x3   0.945387           True          True
4x4x4   0.648133           True          True
5x5x5   0.572529           True          True
```

QE ecut convergence at k = 2x2x2:

```text
 ecut_eV_round kmesh  Ef_vac_eV
           300 2x2x2   0.732419
           400 2x2x2   0.734549
           500 2x2x2   0.734604
           600 2x2x2   0.734706
```

QE dense k=5 cutoff check:

```text
 ecut_eV_round kmesh  Ef_vac_eV
           400 5x5x5   0.572389
           500 5x5x5   0.572440
           600 5x5x5   0.572529
```

DFTpy spacing convergence:

```text
           series  spacing_A  ecut_analogue_eV  Ef_vac_eV
DFTpy fixed QE-a0       0.30        417.811292   2.936613
DFTpy fixed QE-a0       0.25        601.648260   2.933222
DFTpy fixed QE-a0       0.22        776.921824   2.932903
DFTpy fixed QE-a0       0.20        940.075407   2.937061
DFTpy fixed QE-a0       0.18       1160.586922   2.937107
     DFTpy own-a0       0.30        417.811292   3.223579
     DFTpy own-a0       0.25        601.648260   3.224267
     DFTpy own-a0       0.22        776.921824   3.222361
     DFTpy own-a0       0.20        940.075407   3.221530
     DFTpy own-a0       0.18       1160.586922   3.220916
```

DFTpy primitive supercell-size check:

```text
      setting  repeat_n  N_pristine  N_vacancy    volume_A3  Ef_vac_eV
prim_04x04x04         4          64         63  1054.909146   2.937153
prim_06x06x06         6         216        215  3560.318368   2.902672
prim_08x08x08         8         512        511  8439.273169   2.912508
prim_10x10x10        10        1000        999 16482.955408   2.886483
prim_12x12x12        12        1728       1727 28482.546944   2.949975
prim_14x14x14        14        2744       2743 45229.229638   2.900601
```

## Important note about k=6

The local pulled archive at `latest_professor_pull_20260511` still contains an incomplete `k_06x06x06/vacancy_relax/relax.out`, so k=6 is not included in the archive-verified plot. The later terminal output reported `Ef_vac = 0.615195 eV` for k=6, but that completed output file is not present in this local archive copy.

## Interpretation

The discarded old vacancy nanostructure figure used a method-direction relative-energy normalization, which forced the largest-radius reference to zero. These new figures instead plot the physical vacancy formation energy:

```text
E_f^vac = E_vac^(N-1) - ((N-1)/N) E_pristine^N
```

The QE dense-k values are close to the literature vacancy-energy scale, while DFTpy/TFvW gives a much larger vacancy formation energy even after spacing and supercell-size checks. This is a physical/method discrepancy, not a plotting normalization artifact.

## Why the old faceted nanostructure plot is not regenerated as formation energy here

The old `vacancy_nanocrystal_relax` CSVs contain vacancy total energies per atom for QE and OFDFT, but they do not contain a complete matching pristine structure energy for each vacancy case with the same lattice sites/cross-section/repeat. Therefore those CSVs are not sufficient to compute:

```text
E_f^vac = E_vac^(N-1) - ((N-1)/N) E_pristine^N
```

To make a true faceted-nanostructure vacancy-formation-energy plot, each `Al_vac_*` case needs a paired pristine `Al_*` calculation with the same direction, radius, z-repeat, and lattice-site count before atom removal.
