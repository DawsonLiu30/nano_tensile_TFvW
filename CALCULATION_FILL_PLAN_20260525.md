# Calculation Fill Plan

Date: 2026-05-25

This plan follows the professor's latest meeting comments. The goal is to
complete the missing data without mixing old invalid primitive-cell results with
the new VESTA-checked conventional-cell workflow.

## Priority 0: Structure Rule

Before any result is professor-facing:

1. Open the pristine and vacancy `.vasp` files in VESTA.
2. Confirm the intended cell type:
   - QE vacancy benchmark: conventional orthorhombic fcc `2x2x4`,
     `64 -> 63` atoms.
   - DFTpy same-cell comparison: same conventional orthorhombic fcc `2x2x4`,
     `64 -> 63` atoms.
   - DFTpy size/concentration extension: conventional cubic fcc `nxnxn`,
     e.g. `4x4x4`, `256 -> 255` atoms.
3. Do not use the old rhombohedral primitive DFTpy `2.9-3.2 eV` result as a
   method-limitation conclusion.

## Priority 1: QE Conventional Vacancy Completion

Current best completed reference:

```text
conventional fcc 2x2x4, 64 -> 63 atoms
5x5x5 kmesh, 600-800 eV
Ef_vac ~= 0.6011-0.6012 eV
```

Must complete:

| task | reason |
|---|---|
| rerun `kmesh_scan/k_06x06x06` | previous vacancy relax hit time limit |
| optional `kmesh_scan/k_07x07x07` | paid dense-k confidence check |
| force-threshold scan at `5x5x5`, `600 eV` | professor asked force first |
| force-threshold scan at `5x5x5`, `800 eV` | verifies final cutoff point |

Suggested force thresholds:

| label | fmax target |
|---|---:|
| smoke | `0.02 eV/A` |
| intermediate | `0.01 eV/A` |
| tight | `0.005 eV/A` |
| professor target | `0.002 eV/A` |

Important: the pulled QE `relax.out` files currently report final QE
`Total force` values around `0.006-0.011 eV/A`. Therefore, do not claim the
current completed jobs satisfy the strict `0.002 eV/A` target unless a tighter
rerun/check confirms it.

Remote rerun command for the incomplete `6x6x6` case:

```bash
cd /gpfs-work/dawson666/qe_cases/qe_runs/qe_vacancy_conventional_2x2x4_20260522/kmesh_scan/k_06x06x06 && sbatch -p ct56 --time=4-00:00:00 --mem=128G group_job.sh
```

After completion, pull locally:

```bash
cd /mnt/c/Users/dawso/nano_tensile_TFvW && bash scripts/pull_qe_vacancy_conventional_results.sh
```

Collect force diagnostics locally:

```bash
python /mnt/c/Users/dawso/nano_tensile_TFvW/scripts/collect_qe_relax_force_summary.py /mnt/c/Users/dawso/Desktop/qe_conventional_pull_20260522/qe_vacancy_conventional_2x2x4_20260522
```

## Priority 2: DFTpy Conventional Same-Cell Recalibration

Purpose: compare DFTpy and QE using the same conventional cell and the same
vacancy concentration.

| method | cell | atoms | vacancy concentration |
|---|---|---:|---:|
| QE | conventional `2x2x4` | `64 -> 63` | `1/64 = 1.5625%` |
| DFTpy | conventional `2x2x4` | `64 -> 63` | `1/64 = 1.5625%` |

Prepare locally:

```powershell
python scripts\prepare_dftpy_vacancy_conventional.py --outdir "results\dftpy_vacancy_conventional_2x2x4_qe_a0_20260525" --a0 4.039825 --spacing-repeat 2x2x4 --spacing-list "0.30,0.25,0.22,0.20,0.18,0.16" --fmax 0.002 --relax-steps 800
```

Open in VESTA before upload:

```powershell
Start-Process "C:\Users\dawso\AppData\Local\Microsoft\WinGet\Packages\KoichiMomma.VESTA_Microsoft.Winget.Source_8wekyb3d8bbwe\VESTA-win64\VESTA.exe" "C:\Users\dawso\nano_tensile_TFvW\results\dftpy_vacancy_conventional_2x2x4_qe_a0_20260525\spacing_scan\spacing_0p20A\vacancy_start.vasp"
```

Upload scripts and data to iservice:

```bash
rsync -avhP /mnt/c/Users/dawso/nano_tensile_TFvW/scripts/prepare_dftpy_vacancy_conventional.py /mnt/c/Users/dawso/nano_tensile_TFvW/scripts/run_dftpy_conventional_vacancy_one.py /mnt/c/Users/dawso/nano_tensile_TFvW/scripts/collect_dftpy_conventional_vacancy.py /mnt/c/Users/dawso/nano_tensile_TFvW/run_dftpy_conventional_vacancy_one_ct56.sbatch iservice:/gpfs-work/dawson666/dftpy_project/relax/dftpy45/
rsync -avhP /mnt/c/Users/dawso/nano_tensile_TFvW/results/dftpy_vacancy_conventional_2x2x4_qe_a0_20260525/ iservice:/gpfs-work/dawson666/dftpy_project/relax/dftpy45/results/dftpy_vacancy_conventional_2x2x4_qe_a0_20260525/
```

Submit spacing scan on iservice:

```bash
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45 && SERIES_NAME=dftpy_vacancy_conventional_2x2x4_qe_a0_20260525 SCAN=spacing SETTING=spacing_0p30A sbatch run_dftpy_conventional_vacancy_one_ct56.sbatch
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45 && SERIES_NAME=dftpy_vacancy_conventional_2x2x4_qe_a0_20260525 SCAN=spacing SETTING=spacing_0p25A sbatch run_dftpy_conventional_vacancy_one_ct56.sbatch
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45 && SERIES_NAME=dftpy_vacancy_conventional_2x2x4_qe_a0_20260525 SCAN=spacing SETTING=spacing_0p22A sbatch run_dftpy_conventional_vacancy_one_ct56.sbatch
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45 && SERIES_NAME=dftpy_vacancy_conventional_2x2x4_qe_a0_20260525 SCAN=spacing SETTING=spacing_0p20A sbatch run_dftpy_conventional_vacancy_one_ct56.sbatch
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45 && SERIES_NAME=dftpy_vacancy_conventional_2x2x4_qe_a0_20260525 SCAN=spacing SETTING=spacing_0p18A sbatch run_dftpy_conventional_vacancy_one_ct56.sbatch
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45 && SERIES_NAME=dftpy_vacancy_conventional_2x2x4_qe_a0_20260525 SCAN=spacing SETTING=spacing_0p16A sbatch run_dftpy_conventional_vacancy_one_ct56.sbatch
```

Collect on iservice:

```bash
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45 && python scripts/collect_dftpy_conventional_vacancy.py --rootdir results/dftpy_vacancy_conventional_2x2x4_qe_a0_20260525
```

## Priority 3: DFTpy Conventional Size / Concentration Extension

Purpose: separate same-cell method comparison from concentration/size effects.

Recommended conventional cubic scan:

| repeat | atoms | concentration |
|---|---:|---:|
| `2x2x2` | `32 -> 31` | `3.125%` |
| `3x3x3` | `108 -> 107` | `0.9259%` |
| `4x4x4` | `256 -> 255` | `0.3906%` |
| `5x5x5` | `500 -> 499` | `0.2000%` |
| `6x6x6` | `864 -> 863` | `0.1157%` |

Prepare:

```powershell
python scripts\prepare_dftpy_vacancy_conventional.py --outdir "results\dftpy_vacancy_conventional_size_qe_a0_20260525" --a0 4.039825 --spacing-repeat 4 --spacing-list "0.20" --size-repeats "2,3,4,5,6" --fmax 0.002 --relax-steps 1000
```

Run the `size_scan` settings with the same sbatch wrapper:

```bash
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45 && SERIES_NAME=dftpy_vacancy_conventional_size_qe_a0_20260525 SCAN=size SETTING=conv_02x02x02 sbatch run_dftpy_conventional_vacancy_one_ct56.sbatch
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45 && SERIES_NAME=dftpy_vacancy_conventional_size_qe_a0_20260525 SCAN=size SETTING=conv_03x03x03 sbatch run_dftpy_conventional_vacancy_one_ct56.sbatch
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45 && SERIES_NAME=dftpy_vacancy_conventional_size_qe_a0_20260525 SCAN=size SETTING=conv_04x04x04 sbatch run_dftpy_conventional_vacancy_one_ct56.sbatch
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45 && SERIES_NAME=dftpy_vacancy_conventional_size_qe_a0_20260525 SCAN=size SETTING=conv_05x05x05 sbatch run_dftpy_conventional_vacancy_one_ct56.sbatch
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45 && SERIES_NAME=dftpy_vacancy_conventional_size_qe_a0_20260525 SCAN=size SETTING=conv_06x06x06 sbatch run_dftpy_conventional_vacancy_one_ct56.sbatch
```

## Priority 4: Nanocrystal Vacancy Tensile Campaign

Only start this after the benchmark structures and force convergence are clean.

Minimum production matrix:

| variable | values |
|---|---|
| shape | `hexagon`, optionally `triangle` |
| orientation | `[111]` first |
| size | at least 3 diameters |
| vacancy position | `inner`, `middle`, `outer` |
| force target | `fmax = 0.002 eV/A` |
| strain step | production `0.01`; check `0.02` and `0.005` |
| max strain | up to `0.50` or fracture/force collapse |

Run template:

```powershell
python scripts\run_vacancy_periodic_series.py --diameters "1.0,2.0,3.0" --cross-section-shape hexagon --orientation 111 --vacancy-radial-positions "inner,middle,outer" --step 0.01 --cycles 50 --fmax 0.002
```

Step-size sensitivity should be done on one representative case first:

```powershell
python scripts\run_vacancy_periodic_series.py --diameters "2.0" --cross-section-shape hexagon --orientation 111 --vacancy-radial-positions "middle" --step 0.02 --cycles 25 --fmax 0.002
python scripts\run_vacancy_periodic_series.py --diameters "2.0" --cross-section-shape hexagon --orientation 111 --vacancy-radial-positions "middle" --step 0.01 --cycles 50 --fmax 0.002
python scripts\run_vacancy_periodic_series.py --diameters "2.0" --cross-section-shape hexagon --orientation 111 --vacancy-radial-positions "middle" --step 0.005 --cycles 100 --fmax 0.002
```

## What Not To Spend Time On Yet

- Do not rerun the old primitive DFTpy workflow.
- Do not use old `2.9-3.2 eV` DFTpy values in conclusions.
- Do not start a full nanocrystal tensile sweep before the conventional DFTpy
  vacancy benchmark is clean.
- Do not claim strict `0.002 eV/A` convergence unless the output force summary
  confirms it.
