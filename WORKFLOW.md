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

## 4) Postprocess historical finite-grip tensile summaries

The active workflow is axially periodic, but older proposal figures used a
finite-grip tensile setup. Do not use the raw simulation-cell stress as the
main y-axis for those data: the vacuum-padded cell stress is diluted by the
supercell area.

For finite grips, interpret the axial grip reaction as the surface integral of
Cauchy traction on a section normal to the wire axis:

```text
Fz = integral_A t_z dA = integral_A sigma_zz dA
```

The primary reported finite-grip stress should therefore be the
preload-corrected nominal grip-reaction stress:

```text
sigma_nominal_primary(i) = sigma_grip_raw(i) - sigma_grip_raw(0)
sigma_grip_raw = 0.5 * ((-F_top,z / A_ref) + (F_bottom,z / A_ref)) * 160.21766208
```

A current-area correction can be added as an apparent Cauchy-stress sensitivity
check:

```text
sigma_app_cauchy_raw = 0.5 * ((-F_top,z / A_current) + (F_bottom,z / A_current)) * 160.21766208
```

Use the postprocessor on any finite-grip summary CSV/TSV that contains top and
bottom z-force columns plus either area columns or x/y span columns:

```bash
python scripts/add_finite_grip_cauchy_traction_columns.py \
  --summary path/to/finite_grip_summary.csv \
  --out-csv path/to/finite_grip_summary_with_cauchy.csv \
  --plot path/to/finite_grip_stress_check.png
```

The script writes:

- `grip_nominal_primary_GPa`: main finite-grip y-axis
- `grip_apparent_cauchy_primary_GPa`: current-area sensitivity check
- `grip_top_bottom_balance_rel`: top/bottom traction balance diagnostic

Important distinction:

The periodic tensile workflow and the historical finite-grip workflow should
not be mixed into one stress definition.

For the current axially periodic workflow, the main tensile stress is the
area-corrected axial wire Cauchy stress derived from the simulation-cell stress:

```text
cauchy_wire_zz_GPa = sigma_cell_zz_GPa * A_cell / A_wire,current
```

For historical finite-grip data, the main tensile stress is not the raw cell
stress. It is the preload-corrected nominal grip-reaction stress:

```text
grip_nominal_primary_GPa = grip_nominal_raw_GPa(i) - grip_nominal_raw_GPa(0)
```

The apparent current-area Cauchy stress for finite-grip data is reported only
as a sensitivity check, because the atomic-scale current cross-sectional area
may become ambiguous after surface relaxation, vacancy-induced distortion, or
necking.

## Notes

- The supported pseudopotential file is `al.gga.recpot`.
- The supported kinetic-energy functional is `TFVW`.
- The supported tensile driver is `scripts/run_periodic_tensile.py`.
- Legacy finite-length tensile-machine workflows have been removed from the
  active repository workflow to avoid mixing them with the current
  axial-periodic definition.
