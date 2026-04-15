# Supported Workflow

This repository now uses the paper-style periodic nanowire workflow.

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

## 2) Prepare a paper-style periodic [111] wire

This builds a short periodic nanocolumn, scans the axial spacing, picks the equilibrium cell, and writes the 30x replicated long wire.

```bash
python scripts/prepare_paper_periodic_wire.py \
  --case paper_periodic_111_1.0nm_tfvw \
  --diameter-nm 1.0 \
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

## 3) Run periodic tensile

Start from `inputs/short_equilibrium.vasp` for the short-wire elastic workflow, then move to `inputs/long_equilibrium.vasp` for the replicated long wire.

```bash
python scripts/run_periodic_tensile.py \
  --case paper_r1_short_tfvw \
  --workdir cases/paper_periodic_111_1.0nm_tfvw \
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

## Notes

- The supported pseudopotential file is `al.gga.recpot`.
- The supported kinetic-energy functional is `TFVW`.
- The supported tensile driver is `scripts/run_periodic_tensile.py`.
- The older finite-wire/grip workflow is no longer the active path for new runs.
