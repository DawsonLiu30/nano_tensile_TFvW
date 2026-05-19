# Report Summary

Date: 2026-05-19

## Work Hypothesis

The project compares Kohn-Sham DFT/QE and OFDFT/DFTpy for aluminium bulk, vacancy, and nanostructure energetics. The current working hypothesis is:

```text
QE/PBE provides the reference for localized vacancy energetics, while OFDFT/TFvW
can be numerically converged but may show systematic error for localized defects.
For nanostructures, the corrected model definition is axial-periodic; nanocolumn
and nanocrystal differ by xy cross-section shape, not by finite versus infinite
length.
```

## Computational Details

### QE Bulk

- System: fcc Al bulk EOS.
- Convergence parameters: plane-wave cutoff and k-point density.
- Output quantities: equilibrium lattice constant `a0` and bulk modulus `B0`.
- Data root: `C:\Users\dawso\Desktop\latest_professor_pull_20260511\qe_bulk_b_convergence_20260506`.

### QE Vacancy

- System: 4x4x4 primitive fcc Al supercell.
- Pristine cell: 64 atoms.
- Vacancy cell: 63 atoms.
- Formation formula:

```text
E_f^vac = E_vac^(N-1) - ((N-1)/N) E_pristine^N
```

- Data root: `C:\Users\dawso\Desktop\latest_professor_pull_20260511\qe_vacancy_convergence_20260506`.

### DFTpy Vacancy

- Method: OFDFT/DFTpy with TFvW KEDF and local Al recpot.
- DFTpy convergence parameter: real-space density-grid spacing, not k-point sampling.
- Spacing scan: 0.30, 0.25, 0.22, 0.20, 0.18 A.
- Primitive-size scan: 4x4x4 to 14x14x14 primitive fcc supercells.
- Data roots:

```text
C:\Users\dawso\Desktop\latest_professor_pull_20260511\dftpy_vacancy_convergence_primitive4_qe_a0_20260508
C:\Users\dawso\Desktop\latest_professor_pull_20260511\dftpy_vacancy_convergence_primitive4_20260508
C:\Users\dawso\Desktop\latest_professor_pull_20260511\dftpy_vacancy_size_primitive_qe_a0_spacing020_20260511
```

### Nanostructure Data

- Corrected geometry definition: all active nanocolumn/nanocrystal models are axial-periodic.
- Nanocolumn: circular xy cross-section.
- Nanocrystal/faceted nanostructure: polygonal/faceted xy cross-section.
- Geometry audit: `outputs/nanocolumn_nanocrystal_vacancy_audit_20260518.md`.

## Results

### QE Bulk EOS

- QE bulk cutoff convergence is complete above about 400-600 eV.
- High-k validation up to 80x80x80 gives `B0` around 77.96 GPa.
- The 40x40x40 practical reference differs from the 80x80x80 validation result by about 0.03 GPa.

### QE Vacancy

QE ecut convergence at k = 2x2x2:

```text
300 eV: 0.732419 eV
400 eV: 0.734549 eV
500 eV: 0.734604 eV
600 eV: 0.734706 eV
```

QE dense k=5 cutoff check:

```text
400 eV: 0.572389 eV
500 eV: 0.572440 eV
600 eV: 0.572529 eV
```

QE k-point convergence at ecut = 600 eV:

```text
1x1x1: -1.113260 eV
2x2x2:  0.734706 eV
3x3x3:  0.945387 eV
4x4x4:  0.648133 eV
5x5x5:  0.572529 eV
6x6x6:  0.615195 eV, completed remotely after the current pulled archive
```

The gamma-only result is unphysical. The dense k-point results lie in the literature-relevant range.

### DFTpy Vacancy

DFTpy fixed QE-a0 spacing convergence:

```text
spacing 0.30 A: 2.936613 eV
spacing 0.25 A: 2.933222 eV
spacing 0.22 A: 2.932903 eV
spacing 0.20 A: 2.937061 eV
spacing 0.18 A: 2.937107 eV
```

DFTpy own-a0 spacing convergence:

```text
spacing 0.30 A: 3.223579 eV
spacing 0.25 A: 3.224267 eV
spacing 0.22 A: 3.222361 eV
spacing 0.20 A: 3.221530 eV
spacing 0.18 A: 3.220916 eV
```

DFTpy primitive-size convergence at spacing = 0.20 A and fixed QE-a0:

```text
4x4x4:  2.937153 eV
6x6x6:  2.902672 eV
8x8x8:  2.912508 eV
10x10x10: 2.886483 eV
12x12x12: 2.949975 eV
14x14x14: 2.900601 eV
```

The DFTpy vacancy value remains much larger than QE and literature values even after spacing and supercell-size checks.

### Literature Comparison

- Gillan 1989 reported an Al vacancy formation energy around 0.56 eV and compared with an experimental value around 0.66 eV.
- A later GGA/PBE defect study reported finite-size-converged GGA vacancy formation energy consistent with experiment and noted previous 4x4x4 GGA/PBE values around 0.61-0.64 eV.
- Our dense QE vacancy results, approximately 0.57-0.65 eV, are consistent with this literature scale.
- The DFTpy/TFvW values, approximately 2.9-3.2 eV, are much larger and should be interpreted as a method limitation for localized vacancy defects under the current setup.

## Summary

1. QE bulk convergence is established for both cutoff and k-point density.
2. QE vacancy cutoff convergence is established.
3. QE vacancy k-point sampling is metallic and oscillatory, but dense results enter the expected literature range.
4. DFTpy does not use QE-style k-point sampling; DFTpy convergence was checked using real-space spacing and primitive supercell size.
5. DFTpy vacancy formation energy is numerically stable but systematically higher than QE/literature.
6. The old normalized vacancy nanostructure figure should not be used because it forced endpoint agreement. The retained actual-point figure uses only pulled CSV data.
7. All reproducibility paths are listed in `REPRODUCIBILITY_INDEX.md`.

