# DFTpy LDA Vacancy Rerun

This note separates the advisor's LDA request from the corrected QE/PBE
`vc-relax` benchmark.

## Correct Interpretation

- QE remains a corrected PBE-PAW `vc-relax` reference.
- DFTpy is the workflow that must switch from the previous GGA/PBE local
  pseudopotential setup to an LDA-compatible local pseudopotential setup.
- The previous DFTpy value around `2.9-3.2 eV` should be treated as suspicious
  diagnostic output, not as a final method-limitation conclusion.

## Local Pseudopotential

The DFTpy LDA rerun uses:

```text
al.lda.recpot
```

Source:

```text
https://github.com/EACcodes/local-pseudopotentials/blob/master/BLPS/LDA/reci/al.lda.recpot
```

The local repo copy is:

```text
al.lda.recpot
```

## Push Updated DFTpy Files To iservice

Run from WSL:

```bash
cd /mnt/c/Users/dawso/nano_tensile_TFvW && bash scripts/push_vacancy_vcrelax_3x3x3_to_iservice.sh
```

## Prepare DFTpy LDA 3x3x3 Full-Cell-Relax Spacing Scan

Run on iservice:

```bash
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45 && python scripts/prepare_dftpy_vacancy_conventional.py --outdir results/dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_20260529 --a0 4.039848 --spacing-repeat 3x3x3 --spacing-list 0.30,0.25,0.22,0.20,0.18,0.16 --pp al.lda.recpot --xc LDA --kedf TFVW --fmax 0.002 --relax-steps 1500
```

## Relaxation Check Against Official DFTpy Workflow

The generated DFTpy `.ini` files contain:

```text
[JOB]
task = Optdensity
calctype = Energy Force Stress
```

This is expected. In DFTpy, `Optdensity` is the electronic density
optimization performed inside each energy/force/stress evaluation. The
structural relaxation is driven externally through ASE, following the official
DFTpy relaxation tutorial:

```text
DFTpyCalculator(config=conf)
ASE optimizer: BFGS / LBFGS / FIRE
stress relaxation: StrainFilter
force + stress relaxation: UnitCellFilter
```

Our current runner uses the same ASE-coupled workflow, with the modern ASE
cell filter:

```text
DFTpyCalculator(config=conf)
FrechetCellFilter(atoms, scalar_pressure=0)
BFGS(cell_filter).run(fmax=0.002)
```

Therefore, the DFTpy LDA vacancy workflow is not a single-point calculation.
It performs full atomic-position and cell relaxation, analogous to QE
`vc-relax`, while each BFGS step internally solves the OFDFT density
optimization.

Before submitting, open the generated structures in VESTA:

```text
/gpfs-work/dawson666/dftpy_project/relax/dftpy45/results/dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_20260529/spacing_scan/spacing_0p20A/pristine_raw.vasp
/gpfs-work/dawson666/dftpy_project/relax/dftpy45/results/dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_20260529/spacing_scan/spacing_0p20A/vacancy_start.vasp
```

## Submit DFTpy LDA Scan

Run on iservice:

```bash
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45 && SERIES_NAME=dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_20260529 sbatch submit_dftpy_vcrelax_conv3x3x3_ct56_array.sh
```

## Collect Results

After the Slurm array finishes:

```bash
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45 && python scripts/collect_dftpy_conventional_vacancy.py --rootdir results/dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_20260529 && python scripts/collect_dftpy_vcrelax_fmax.py results/dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_20260529 --out results/dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_20260529/dftpy_vcrelax_fmax_summary.csv
```

## KEDF Diagnostic: SM and WT

If the LDA + TFVW result remains too high, run the same VESTA-checked 3x3x3
cell with nonlocal kinetic-energy density functionals. The official DFTpy
configuration page lists `WT` and `SM` as supported KEDF options.

Prepare and submit `WT`:

```bash
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45 && python scripts/prepare_dftpy_vacancy_conventional.py --outdir results/dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_WT_20260529 --a0 4.039848 --spacing-repeat 3x3x3 --spacing-list 0.30,0.25,0.22,0.20,0.18,0.16 --pp al.lda.recpot --xc LDA --kedf WT --fmax 0.002 --relax-steps 1500 && SERIES_NAME=dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_WT_20260529 sbatch submit_dftpy_vcrelax_conv3x3x3_ct56_array.sh
```

Prepare and submit `SM`:

```bash
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45 && python scripts/prepare_dftpy_vacancy_conventional.py --outdir results/dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_SM_20260529 --a0 4.039848 --spacing-repeat 3x3x3 --spacing-list 0.30,0.25,0.22,0.20,0.18,0.16 --pp al.lda.recpot --xc LDA --kedf SM --fmax 0.002 --relax-steps 1500 && SERIES_NAME=dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_SM_20260529 sbatch submit_dftpy_vcrelax_conv3x3x3_ct56_array.sh
```

Collect `WT`:

```bash
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45 && python scripts/collect_dftpy_conventional_vacancy.py --rootdir results/dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_WT_20260529 && python scripts/collect_dftpy_vcrelax_fmax.py results/dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_WT_20260529 --out results/dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_WT_20260529/dftpy_vcrelax_fmax_summary.csv
```

Collect `SM`:

```bash
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45 && python scripts/collect_dftpy_conventional_vacancy.py --rootdir results/dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_SM_20260529 && python scripts/collect_dftpy_vcrelax_fmax.py results/dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_SM_20260529 --out results/dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_SM_20260529/dftpy_vcrelax_fmax_summary.csv
```

## PPT Wording

Use this wording:

```text
QE: corrected PBE-PAW conventional 3x3x3 vc-relax reference.
DFTpy: LDA-compatible local pseudopotential rerun required.
```

Do not write that QE must be rerun in LDA unless the advisor explicitly asks
for a separate QE/LDA benchmark.
