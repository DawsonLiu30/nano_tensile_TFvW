# DFTpy TF+vW Lambda-Mu Bulk Scan

## Definition

The kinetic-energy functional is scanned as

`T_s[n] = lambda_TF T_TF[n] + mu_vW T_vW[n]`.

In DFTpy's `TFVW` implementation:

- `x = lambda_TF`, the Thomas-Fermi coefficient.
- `y = mu_vW`, the von Weizsaecker coefficient.

The coefficients are independent. There is no `lambda_TF + mu_vW = 1`
constraint.

## Scientific motivation

The Alharbi 2026 KEDF review identifies the Thomas-Fermi homogeneous-electron
gas limit and the von Weizsaecker single-orbital limit as distinct anchors.
The Thomas-Fermi term is commonly retained as the leading baseline, while
gradient/vW corrections are varied. The advisor-requested calculation extends
the earlier fixed-`lambda_TF=1` scan by varying both coefficients.

## Requested grid

- Rows: `lambda_TF = 0.1, 0.2, ..., 1.0`.
- Columns: `mu_vW = 0.1, 0.2, ..., 1.0`.
- Total: 100 independent coefficient pairs.

For every pair, the workflow performs an fcc Al EOS/lattice scan and records:

1. equilibrium total energy in eV/atom;
2. equilibrium kinetic (KEDF) energy in eV/atom;
3. equilibrium lattice constant in Angstrom.

The raw long-form table also retains whole-cell energies, weighted TF and vW
components, fit method, fit RMSE, bulk modulus, and boundary-fit warnings.

## Numerical setup

- Material: conventional cubic fcc Al, 4 atoms.
- XC: LDA.
- Local pseudopotential: `al.lda.recpot`.
- DFTpy KEDF: `TFVW`.
- Grid spacing: `0.20 A`.
- Density optimizer: `CG-HS`, with `maxiter=maxfun=500`. This is more robust
  than LBFGS for the strongly contracted low-coefficient cells.
- Lattice scan: `2.20-5.00 A` in `0.10 A` steps. The broad range is required
  because small coefficients can predict a strongly contracted or collapsed
  lattice.
- Equilibrium lattice constant: zero-hydrostatic-stress interpolation between
  adjacent lattice points. Birch-Murnaghan/local-quadratic fitting is retained
  only as a fallback when no pressure sign change is available.
- Non-converged density-optimization points are retained in raw logs but
  excluded from the EOS fit.
- Scheduler: use `ctest`, at most two simultaneous tasks.

## Submission

Prepare the 100-point series and submit the first two row groups:

```bash
cd /mnt/c/Users/dawso/nano_tensile_TFvW

bash scripts/push_dftpy_tfvw_lambda_mu_bulk_scan_to_iservice_20260609.sh 0 1
```

Each ctest array task runs ten coefficient pairs serially, corresponding to
one complete `lambda_TF` row. Thus the full 10x10 map needs only ten array
tasks. Submit later row groups after scheduler capacity is available:

```bash
bash scripts/submit_existing_dftpy_tfvw_lambda_mu_bulk_scan_to_iservice_20260609.sh 2 3
bash scripts/submit_existing_dftpy_tfvw_lambda_mu_bulk_scan_to_iservice_20260609.sh 4 5
bash scripts/submit_existing_dftpy_tfvw_lambda_mu_bulk_scan_to_iservice_20260609.sh 6 7
bash scripts/submit_existing_dftpy_tfvw_lambda_mu_bulk_scan_to_iservice_20260609.sh 8 9
```

## Collection

```bash
cd /mnt/c/Users/dawso/nano_tensile_TFvW

bash scripts/pull_dftpy_tfvw_lambda_mu_bulk_scan_results_20260609.sh
```

The professor-style three-panel table is written to:

`tables/professor_three_panel_lambda_mu_table.csv`
