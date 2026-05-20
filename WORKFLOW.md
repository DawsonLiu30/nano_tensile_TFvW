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
  --fmax 0.02 \
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
  --fmax 0.02 \
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
  --cycles 20 \
  --fmax 0.02 \
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

## Notes

- The supported pseudopotential file is `al.gga.recpot`.
- The supported kinetic-energy functional is `TFVW`.
- The supported tensile driver is `scripts/run_periodic_tensile.py`.
- Legacy finite-length / grip terminology is excluded from the active thesis
  workflow to keep the model definition consistent: both nanocolumn and
  nanocrystal are infinite along z.
