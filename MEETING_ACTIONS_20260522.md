# Professor Meeting Actions

Date: 2026-05-22

## Current Status Update: 2026-05-25

The most urgent structural problem has been stopped: the professor-facing QE
vacancy benchmark now uses a VESTA-checked conventional orthorhombic fcc
`2x2x4` supercell instead of the old rhombohedral primitive-cell visualization.

Current QE conventional vacancy cell:

| quantity | value |
|---|---:|
| conventional repeat | `2x2x4` |
| pristine atoms | `64` |
| vacancy atoms | `63` |
| cell lengths | `8.0797 x 8.0797 x 16.1594 A` |
| cell angles | `90 / 90 / 90 deg` |
| volume | `1054.909 A^3` |
| vacancy concentration | `1/64 = 1.5625%` |
| VESTA status | checked |

Latest QE conventional vacancy results:

| scan | setting | Ef_vac (eV) | status |
|---|---:|---:|---|
| cutoff at `2x2x2` | `300 eV` | `1.058787` | complete |
| cutoff at `2x2x2` | `400 eV` | `1.060793` | complete |
| cutoff at `2x2x2` | `500 eV` | `1.060857` | complete |
| cutoff at `2x2x2` | `600 eV` | `1.060956` | complete |
| cutoff at `2x2x2` | `800 eV` | `1.061077` | complete |
| dense-k cutoff at `5x5x5` | `400 eV` | `0.600951` | complete |
| dense-k cutoff at `5x5x5` | `500 eV` | `0.600997` | complete |
| dense-k cutoff at `5x5x5` | `600 eV` | `0.601085` | complete |
| dense-k cutoff at `5x5x5` | `800 eV` | `0.601226` | complete |
| kmesh at `600 eV` | `1x1x1` | `0.678188` | complete |
| kmesh at `600 eV` | `2x2x2` | `1.060956` | complete, under-sampled |
| kmesh at `600 eV` | `3x3x3` | `0.724412` | complete |
| kmesh at `600 eV` | `4x4x4` | `0.543750` | complete |
| kmesh at `600 eV` | `5x5x5` | `0.601085` | best completed dense-k reference |
| kmesh at `600 eV` | `6x6x6` | pending | vacancy relax hit time limit |

Current professor-facing interpretation:

- The cutoff convergence is clean.
- The `2x2x2` cutoff scan is useful only for cutoff stability, not as the final
  vacancy formation energy, because it is k-point under-sampled.
- The best completed QE conventional vacancy reference is the dense-k
  `5x5x5` result: `Ef_vac ~= 0.6011-0.6012 eV` for `600-800 eV`.
- The `6x6x6` case must be rerun with a longer wall time if a final dense-k
  uncertainty check is required.
- The pulled QE `relax.out` files report final `Total force` values around
  `0.006-0.011 eV/A` for completed runs. Do not overclaim that these completed
  QE reruns satisfy the professor's final `fmax = 0.002 eV/A` target without a
  dedicated tighter-force rerun/check.
- The old DFTpy `2.9-3.2 eV` primitive-cell result must not be used as a
  method-limitation conclusion. It was invalidated by the structure issue and
  must be recalculated with VESTA-checked conventional fcc cells.

Remaining work grouped by urgency:

| item | status | next action |
|---|---|---|
| QE conventional `6x6x6` | partial | rerun vacancy relaxation with longer wall time |
| QE final-force summary | partial | `scripts/collect_qe_relax_force_summary.py` extracts QE `Total force`; tighter rerun needed for strict `0.002 eV/A` claim |
| DFTpy same-cell conventional test | not complete | run conventional `2x2x4`, `64 -> 63`, same concentration as QE |
| DFTpy conventional spacing scan | not complete | rerun with conventional fcc cell; old primitive scan is invalid |
| DFTpy concentration/size extension | not complete | run conventional `4x4x4`, `256 -> 255`, `c_v = 0.390625%` |
| vacancy concentration table | partial | add to report/slides for QE and DFTpy cases |
| inner/middle/outer vacancy tensile | not started | define reproducible radial-position selection and generate cases |
| tensile force convergence | not started | use `fmax = 0.002 eV/A` for final reported runs |
| tensile step-size sensitivity | not started | compare `0.02`, `0.01`, `0.005` after force convergence |

## What Changed

The professor identified four required corrections:

1. The DFTpy vacancy convergence starting structures must be visually checked in
   VESTA before any calibration run.
2. The DFTpy calibration value near 2.9 eV is not trusted until the starting
   structure issue is removed.
3. Vacancy tensile tests for nanocrystals must report vacancy concentration and
   compare vacancy positions: inner, middle, and outer.
4. Tensile settings should be tightened: force convergence first, then energy,
   then strain step size. Suggested force threshold: `fmax = 0.002 eV/A`.

## New Rule For DFTpy Calibration

Do not use the old rhombohedral primitive 4x4x4 DFTpy calibration cell in the
slides. Even if the atom count is mathematically correct, the 60-degree cell is
visually misleading in VESTA and was rejected in the meeting.

The new formal route is:

1. Build a conventional cubic fcc supercell.
2. Use conventional 4x4x4 for the first clean spacing scan.
3. This means 256 pristine atoms and 255 vacancy atoms.
4. Open both `.vasp` files in VESTA.
4. Only then submit the DFTpy calculations.

Preparation command template:

```powershell
python scripts\prepare_dftpy_vacancy_conventional.py `
  --outdir "results\dftpy_vacancy_conventional_qe_a0_20260522" `
  --a0 4.039825 `
  --spacing-list "0.30,0.25,0.22,0.20,0.18" `
  --spacing-repeat 4 `
  --fmax 0.002
```

Then open representative files:

```powershell
Start-Process "C:\Users\dawso\AppData\Local\Microsoft\WinGet\Packages\KoichiMomma.VESTA_Microsoft.Winget.Source_8wekyb3d8bbwe\VESTA-win64\VESTA.exe" `
  "C:\Users\dawso\nano_tensile_TFvW\results\dftpy_vacancy_conventional_qe_a0_20260522\spacing_scan\spacing_0p20A\vacancy_start.vasp"
```

Important distinction:

- Old primitive 4x4x4: 64 pristine atoms, 63 vacancy atoms, 60-degree
  rhombohedral cell.
- New conventional 4x4x4: 256 pristine atoms, 255 vacancy atoms, 90-degree
  cubic cell.

Use the new conventional result for professor-facing DFTpy calibration.

## New Rule For QE Vacancy Convergence

Do not reuse the old primitive rhombohedral QE vacancy cell in the updated
slides. QE should also use a VESTA-friendly conventional-cell construction, but
it must stay below the practical 250-atom QE limit discussed in the meeting.

Use a conventional orthorhombic fcc 2x2x4 supercell:

- pristine atoms: 64
- vacancy atoms: 63
- cell angles: 90, 90, 90 degrees
- volume: 1054.909 A^3
- removed atom: central Al site at fractional coordinate (0.5, 0.5, 0.5)

Preparation command:

```powershell
python scripts\prepare_qe_vacancy_conventional_orthorhombic.py --outdir "results\qe_vacancy_conventional_2x2x4_20260522" --a0 4.039848 --repeat 2x2x4 --force-conv 0.002
```

The updated QE rerun includes the professor-requested 800 eV point:

- ecut scan at 2x2x2: 300, 400, 500, 600, 800 eV
- dense-k ecut check at 5x5x5: 400, 500, 600, 800 eV
- kmesh scan at 600 eV: 1x1x1 through 6x6x6

QE input rule: `forc_conv_thr = 0.0000777876 Ry/Bohr` belongs in `&CONTROL`,
not in `&IONS`.

Open `pristine_start.vasp` and `vacancy_start.vasp` in VESTA before sending the
package to iservice.

## Vacancy Concentration Definition

For a periodic nanocolumn/nanocrystal repeat cell:

```text
vacancy concentration fraction = N_vacancies / N_pristine_sites
vacancy concentration percent  = 100 * N_vacancies / N_pristine_sites
```

For one vacancy:

```text
c_v = 1 / N_pristine
```

The scripts also store a diagnostic concentration per nm^3 using:

```text
physical prism volume = model cross-section area * axial repeat length
```

This avoids using the vacuum-padded simulation-cell volume.

## Vacancy Position Study

For nanocrystal tensile tests, run the same diameter/shape/orientation with
three vacancy positions:

- `inner`: atom closest to the axial center line
- `middle`: atom near half of the outer radial distance
- `outer`: atom near the surface

Series command template:

```powershell
python scripts\run_vacancy_periodic_series.py `
  --diameters "1.0,2.0,3.0" `
  --cross-section-shape hexagon `
  --orientation 111 `
  --vacancy-radial-positions "inner,middle,outer" `
  --step 0.01 `
  --cycles 50 `
  --fmax 0.002
```

The generated `vacancy_branch_manifest.json` now records:

- vacancy radial position
- selected atom index and coordinates
- vacancy concentration fraction
- vacancy concentration percent
- vacancy concentration per nm^3 using the model prism volume

## Convergence Order

Use this order for the next clean production campaign:

1. Structure/VESTA validation
2. Force convergence
3. Energy convergence
4. Strain step-size convergence

Suggested force convergence scan:

| Stage | fmax (eV/A) | Purpose |
|---|---:|---|
| quick smoke test | 0.02 | only to catch runtime/script issues |
| intermediate | 0.01 | early trend check |
| tighter | 0.005 | force sensitivity |
| professor target | 0.002 | final reported runs |

Suggested strain step-size check after force/energy are stable:

| Step | Target strain range | Comment |
|---:|---:|---|
| 0.02 | up to 0.50 | coarse screening |
| 0.01 | up to 0.50 | default production check |
| 0.005 | near key transition region | only if peak/fracture behavior depends on step |

Do not use a single expensive step-size sweep before structure and force
convergence are clean.
