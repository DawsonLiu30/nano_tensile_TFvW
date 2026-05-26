# Final Report: fcc Al Vacancy Benchmark and Next Nanocrystal Tensile Plan

Date: 2026-05-26  
Author: Dawson Liu  
Purpose: one clean reference report for professor review and NUS upload.

## 1. Executive Summary

The previous primitive/slanted vacancy-cell workflow has been withdrawn. The corrected benchmark now uses VESTA-checked conventional fcc Al cells.

Current best completed QE vacancy reference:

- Structure: conventional fcc 2x2x4 supercell
- Pristine / vacancy atoms: 64 -> 63
- Vacancy concentration: 1/64 = 1.5625%
- Best completed dense-k value: 5x5x5, 600-800 eV = about 0.6011-0.6012 eV

Current DFTpy conventional same-cell result:

- Structure: same conventional fcc 2x2x4 cell
- Pristine / vacancy atoms: 64 -> 63
- Vacancy concentration: 1.5625%
- KEDF: TFvW
- Grid spacing scan: 0.30 to 0.16 Angstrom
- Converged vacancy formation energy: about 2.901 eV
- All completed same-cell DFTpy vacancy relaxations satisfy actual final fmax < 0.002 eV/Angstrom

Main interpretation:

The old DFTpy result was correctly invalidated because the old cell geometry was not acceptable. However, after rebuilding the benchmark with VESTA-checked conventional fcc cells, DFTpy/TFvW still gives a high vacancy formation energy near 2.90 eV. Therefore, the high DFTpy vacancy energy is not solely caused by the old primitive/slanted-cell artifact.

## 2. Corrected Structure Definition

The corrected vacancy benchmark cell is:

| item | value |
|---|---:|
| cell type | conventional fcc Al |
| repeat | 2x2x4 |
| pristine atoms | 64 |
| vacancy atoms | 63 |
| vacancy concentration | 1.5625% |
| approximate cell | 8.0797 x 8.0797 x 16.1594 Angstrom |
| cell angles | 90 / 90 / 90 deg |
| volume | 1054.909 Angstrom^3 |

This cell is used to avoid the visual ambiguity of primitive/slanted cells in VESTA and to keep the QE benchmark below the practical 250-atom limit.

## 3. QE Conventional Vacancy Results

Vacancy formation energy formula:

```text
E_f^vac = E_vac^(N-1) - ((N-1)/N) E_perfect^N
```

### QE cutoff scan at 2x2x2 kmesh

| cutoff (eV) | E_f^vac (eV) |
|---:|---:|
| 300 | 1.058787 |
| 400 | 1.060793 |
| 500 | 1.060857 |
| 600 | 1.060956 |
| 800 | 1.061077 |

Interpretation: cutoff convergence is good, but 2x2x2 kmesh is not sufficient for the final vacancy energy.

### QE dense-k cutoff check at 5x5x5 kmesh

| cutoff (eV) | E_f^vac (eV) |
|---:|---:|
| 400 | 0.600951 |
| 500 | 0.600997 |
| 600 | 0.601085 |
| 800 | 0.601226 |

Interpretation: the dense-k cutoff effect from 400 to 800 eV is only about 0.000275 eV. The best completed QE dense-k reference is therefore about 0.601 eV.

### QE kmesh scan at 600 eV

| kmesh | E_f^vac (eV) | status |
|---|---:|---|
| 1x1x1 | 0.678188 | completed |
| 2x2x2 | 1.060956 | completed, under-sampled |
| 3x3x3 | 0.724412 | completed |
| 4x4x4 | 0.543750 | completed |
| 5x5x5 | 0.601085 | best completed dense-k reference |
| 6x6x6 | pending / incomplete in local pull | needs final collection |

Important wording: the 5x5x5 value is the best completed dense-k reference so far, not a fully locked final value until the 6x6x6 collection and QE force verification are complete.

## 4. DFTpy Conventional Same-Cell Spacing Scan

DFTpy setup:

- Method: OF-DFT / DFTpy
- KEDF: TFvW
- Cell: conventional fcc 2x2x4
- Pristine / vacancy atoms: 64 -> 63
- Vacancy concentration: 1.5625%
- Relaxation target: fmax < 0.002 eV/Angstrom

| spacing (Angstrom) | ecut analogue (eV) | E_f^vac (eV) | actual final fmax (eV/Angstrom) |
|---:|---:|---:|---:|
| 0.30 | 417.811 | 2.900725 | 0.001593 |
| 0.25 | 601.648 | 2.900797 | 0.000858 |
| 0.22 | 776.922 | 2.900844 | 0.001990 |
| 0.20 | 940.075 | 2.900849 | 0.001016 |
| 0.18 | 1160.587 | 2.900901 | 0.001119 |
| 0.16 | 1468.868 | 2.901197 | 0.001231 |

Interpretation:

The same-cell DFTpy result is numerically converged with respect to real-space grid spacing. Across 0.30 to 0.16 Angstrom, the vacancy formation energy changes by only about 0.000472 eV.

## 5. DFTpy Size and Vacancy-Concentration Scan

DFTpy size scan setup:

- Cell type: conventional cubic fcc
- Spacing: 0.20 Angstrom
- KEDF: TFvW
- Relaxation target: fmax < 0.002 eV/Angstrom

| cell | pristine atoms | vacancy atoms | vacancy concentration (%) | E_f^vac (eV) | actual final fmax (eV/Angstrom) | status |
|---|---:|---:|---:|---:|---:|---|
| conv_02x02x02 | 32 | 31 | 3.125000 | 2.937948 | 0.001349 | accepted |
| conv_03x03x03 | 108 | 107 | 0.925926 | 2.897041 | 0.000437 | accepted |
| conv_04x04x04 | 256 | 255 | 0.390625 | 2.887601 | 0.001197 | accepted |
| conv_05x05x05 | 500 | 499 | 0.200000 | 2.883649 | 0.001896 | accepted |
| conv_06x06x06 | 864 | 863 | 0.115741 | not accepted | 0.004161 | not converged |

Interpretation:

The vacancy concentration effect is visible but small compared with the QE-DFTpy discrepancy. Reducing the vacancy concentration from 3.125% to 0.200% lowers the DFTpy vacancy formation energy by about 0.054 eV, while the same-cell QE-DFTpy difference is about 2.30 eV.

## 6. Comparison with Literature

Literature anchors used in the slide/report comparison:

- Gillan 1989: calculated Al vacancy formation energy about 0.56 eV, experimental reference about 0.66 eV.
- GGA/DMC Al defect paper: GGA vacancy formation energies in converged supercells are reported around the same physical range, with finite-size and twist/k-point effects emphasized for metallic defects.

Current comparison:

| source / method | representative setting | E_f^vac (eV) |
|---|---|---:|
| Gillan 1989 | LDA pseudopotential supercell | about 0.56 |
| Experiment reference | literature reference | about 0.66 |
| QE, this work | conventional 2x2x4, 5x5x5, 600-800 eV | about 0.601 |
| DFTpy, this work | conventional 2x2x4, TFvW, spacing 0.20 Angstrom | about 2.901 |

## 7. Reproducibility and Data Locations

The NUS upload package should contain:

```text
00_FINAL_REPORT/
01_QE_conventional_vacancy_2x2x4/
02_DFTpy_conventional_same_cell_2x2x4/
03_DFTpy_conventional_size_concentration/
04_scripts_and_reproducibility/
05_pending_or_not_final/
```

Recommended local package root:

```text
C:\Users\dawso\Desktop\FINAL_NUS_UPLOAD_20260526
```

Remote sources:

```text
QE:
iservice:/gpfs-work/dawson666/qe_cases/qe_runs/qe_vacancy_conventional_2x2x4_20260522/

DFTpy same-cell:
iservice:/gpfs-work/dawson666/dftpy_project/relax/dftpy45/results/dftpy_vacancy_conventional_2x2x4_qe_a0_20260525/

DFTpy size/concentration:
iservice:/gpfs-work/dawson666/dftpy_project/relax/dftpy45/results/dftpy_vacancy_conventional_size_qe_a0_20260525/
```

Key local pull commands:

```bash
cd /mnt/c/Users/dawso/nano_tensile_TFvW && bash scripts/pull_qe_vacancy_conventional_results.sh
cd /mnt/c/Users/dawso/nano_tensile_TFvW && bash scripts/pull_dftpy_conventional_results.sh
```

After both pulls, build the final local upload package:

```powershell
powershell -ExecutionPolicy Bypass -File C:\Users\dawso\nano_tensile_TFvW\scripts\build_final_nus_upload_package_20260526.ps1
```

## 8. Next Calculations: Nanocrystal Tensile with Vacancy Concentration and Vacancy Position

Yes, the next main calculation should move to nanocrystal/nanocolumn tensile tests, but only using VESTA-checked periodic structures.

Required new metadata for every vacancy tensile case:

- pristine atom count: N_pristine
- vacancy atom count: N_vacancy
- number of vacancies: N_vac
- vacancy concentration: c_v = N_vac / N_pristine
- vacancy concentration percent: 100 * N_vac / N_pristine
- vacancy radial position class: inner / middle / outer
- vacancy radial coordinate: r_vac
- normalized radial coordinate: eta = r_vac / R_eff

Recommended vacancy-position definitions:

| class | normalized position eta |
|---|---|
| inner | eta <= 0.30 |
| middle | 0.45 <= eta <= 0.65 |
| outer | eta >= 0.80 |

Recommended production tensile settings:

| item | value |
|---|---|
| boundary condition | axially periodic nanocrystal / nanocolumn |
| stress output | area-corrected axial Cauchy wire stress |
| strain increment | 0.01 |
| maximum strain | up to 0.50 or until mechanical failure |
| relaxation force target | fmax < 0.002 eV/Angstrom |
| first pilot | one representative nanocrystal size, three vacancy positions |

Recommended step-size sensitivity check:

| step size | purpose |
|---:|---|
| 0.020 | coarse screening |
| 0.010 | production default |
| 0.005 | convergence check |

Do not launch the full matrix blindly. First generate one representative size with inner/middle/outer vacancy, inspect all three in VESTA, then submit the tensile jobs.

## 9. Current Pending Items

1. Pull or rerun QE 6x6x6 conventional vacancy and collect the final result.
2. Verify QE final force criteria consistently; do not claim QE fmax < 0.002 eV/Angstrom unless confirmed from output.
3. Pull the full DFTpy conventional input/output/log structure to local package if not already pulled.
4. Update the final PPT deck to match this report after the data package is finalized.
5. Start nanocrystal tensile pilot only after generated inner/middle/outer structures pass visual inspection.
