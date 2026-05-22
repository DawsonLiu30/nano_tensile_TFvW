# Supported Workflow

This repository now uses an axially periodic prism workflow. There should be
no finite-length tensile-machine model in the main analysis.

Terminology:

- `nanocolumn`: infinite along the axial direction, circular xy cross-section.
- `nanocrystal`: infinite along the axial direction, polygonal xy cross-section
  such as hexagon or triangle.
- The z length in the input files is the periodic repeat/supercell length, not
  a finite physical column length.

## Directory convention

- `cases/<case>/inputs/`: prepared periodic structures and scan outputs
- `cases/<case>/results/<run>/`: tensile outputs for one periodic run
- `results/`: bulk validation and cross-case analysis outputs

## 1) Validate bulk Al with TFVW

Run the bulk smoke/validation first so the lattice constant and small-strain stress are checked before any nanowire job.

```bash
python scripts/bulk_validate.py \
  --pp al.gga.recpot \
  --kedf TFVW \
  --ecut 1000
```

## 2) Prepare an axially periodic [111] prism

This builds a periodic repeat unit, scans the axial spacing, picks the
equilibrium repeat length, and optionally writes a replicated reference
supercell. The replicated cell is still periodic/infinite along z.

For a nanocolumn use:

```bash
python scripts/prepare_paper_periodic_wire.py \
  --case nanocolumn_circle_periodic_111_1.0nm_tfvw \
  --diameter-nm 1.0 \
  --cross-section-shape circle \
  --orientation 111 \
  --a0 4.118877004246 \
  --vacuum 10.0 \
  --replicate-z 30 \
  --scan-scales 0.95,0.96,0.97,0.98,0.99,1.00,1.01 \
  --pp al.gga.recpot \
  --kedf TFVW \
  --ecut 1000 \
  --fmax 0.002 \
  --relax-steps 120
```

For a polygonal nanocrystal use `--cross-section-shape hexagon` or
`--cross-section-shape triangle`:

```bash
python scripts/prepare_paper_periodic_wire.py \
  --case nanocrystal_hexagon_periodic_111_1.0nm_tfvw \
  --diameter-nm 1.0 \
  --cross-section-shape hexagon \
  --orientation 111 \
  --a0 4.118877004246 \
  --vacuum 10.0 \
  --replicate-z 30 \
  --pp al.gga.recpot \
  --kedf TFVW \
  --ecut 1000 \
  --fmax 0.002 \
  --relax-steps 120
```

## 3) Run periodic tensile with Cauchy stress

Start from `inputs/short_equilibrium.vasp` for the periodic repeat. The main
stress output is now the axial wire Cauchy stress:

```text
cauchy_wire_zz_GPa = sigma_cell_zz_GPa * A_cell / A_wire,current
```

```bash
python scripts/run_periodic_tensile.py \
  --case nanocolumn_circle_r1_tfvw \
  --workdir cases/nanocolumn_circle_periodic_111_1.0nm_tfvw \
  --init inputs/short_equilibrium.vasp \
  --pp al.gga.recpot \
  --kedf TFVW \
  --ecut 1000 \
  --step 0.01 \
  --cycles 50 \
  --fmax 0.002 \
  --relax-steps 80 \
  --plot-summary
```

## 4) Stress-definition guardrail

There is no finite-length column and no physical tensile grip in the supported
nanocolumn/nanocrystal workflow. All tensile cases are axially periodic. Any
older file or folder name containing `finite`, `grip`, `top`, or `bottom`
should be treated as historical/misleading naming and should not define the
formal thesis terminology.

The formal stress definition for tensile data is:

```text
sigma_wire,zz^Cauchy = sigma_cell,zz * A_cell / A_wire,current
```

where `sigma_cell,zz` is the simulation-cell Cauchy stress reported by the code,
`A_cell` is the full x-y supercell area, and `A_wire,current` is the current
projected wire/prism cross-section area. This correction removes the vacuum
dilution from the cell-averaged stress while keeping the axial-periodic
boundary condition.

Do not report any of the following as formal thesis quantities:

- finite-grip stress
- grip-reaction stress
- top/bottom grip balance
- finite physical column length
- preload-corrected grip stress

## 5) Vacancy nanocrystal tensile runs

After the 2026-05-22 professor meeting, vacancy tensile tests must record both
vacancy concentration and vacancy position. For one vacancy in a periodic
repeat cell:

```text
c_v = N_vacancies / N_pristine_sites = 1 / N_pristine
```

Use `scripts/run_vacancy_periodic_series.py` to run the three requested radial
locations:

```bash
python scripts/run_vacancy_periodic_series.py \
  --diameters 1.0,2.0,3.0 \
  --cross-section-shape hexagon \
  --orientation 111 \
  --vacancy-radial-positions inner,middle,outer \
  --step 0.01 \
  --cycles 50 \
  --fmax 0.002
```

The generated `vacancy_branch_manifest.json` records the selected atom,
radial location, vacancy concentration fraction, vacancy concentration
percentage, and a per-nm3 diagnostic based on the physical prism volume.

Convergence order for a clean campaign:

1. structure and VESTA validation
2. force convergence, targeting `fmax = 0.002 eV/A`
3. energy convergence
4. tensile step-size convergence

## 6) DFTpy bulk-vacancy calibration cell

For professor-facing DFTpy vacancy formation-energy calibration, use a
conventional cubic fcc supercell instead of the old rhombohedral primitive
4x4x4 cell. The old primitive cell is mathematically valid but visually
misleading in VESTA because it has 60-degree cell angles.

```bash
python scripts/prepare_dftpy_vacancy_conventional.py \
  --outdir results/dftpy_vacancy_conventional_qe_a0_20260522 \
  --a0 4.039825 \
  --spacing-list 0.30,0.25,0.22,0.20,0.18 \
  --spacing-repeat 4 \
  --fmax 0.002
```

This conventional 4x4x4 fcc cell contains 256 pristine atoms and 255 atoms in
the vacancy cell. Open `spacing_scan/spacing_0p20A/pristine_raw.vasp` and
`spacing_scan/spacing_0p20A/vacancy_start.vasp` in VESTA before submitting the
DFTpy runs.

## 7) QE bulk-vacancy calibration cell

For the QE vacancy formation-energy convergence rerun, use a VESTA-friendly
conventional fcc supercell while keeping the original 64/63 atom scale. A full
conventional cubic 4x4x4 fcc cell would contain 256/255 atoms, which exceeds
the practical QE limit discussed in the meeting. Therefore the QE rerun uses a
conventional orthorhombic 2x2x4 fcc supercell:

```bash
python scripts/prepare_qe_vacancy_conventional_orthorhombic.py \
  --outdir results/qe_vacancy_conventional_2x2x4_20260522 \
  --a0 4.039848 \
  --repeat 2x2x4 \
  --force-conv 0.002
```

This cell has 64 pristine atoms, 63 vacancy atoms, 90-degree cell angles, and
volume 1054.909 A^3. The vacancy atom is the central Al site at fractional
coordinate (0.5, 0.5, 0.5). The prepared QE rerun includes:

- cutoff convergence at fixed 2x2x2 k-mesh: 300, 400, 500, 600, and 800 eV
- dense-k cutoff check at fixed 5x5x5 k-mesh: 400, 500, 600, and 800 eV
- k-mesh convergence at fixed 600 eV: 1x1x1 through 6x6x6

The vacancy relaxation threshold is `forc_conv_thr = 0.0000777876 Ry/Bohr`,
corresponding to `0.002 eV/A`, and it must be placed in QE `&CONTROL`.

Open `pristine_start.vasp` and `vacancy_start.vasp` in VESTA before submitting
the QE runs.

## Notes

- The supported pseudopotential file is `al.gga.recpot`.
- The supported kinetic-energy functional is `TFVW`.
- The supported tensile driver is `scripts/run_periodic_tensile.py`.
- Legacy finite-length / grip terminology is excluded from the active thesis
  workflow to keep the model definition consistent: both nanocolumn and
  nanocrystal are infinite along z.
