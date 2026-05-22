# Professor Meeting Actions

Date: 2026-05-22

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
