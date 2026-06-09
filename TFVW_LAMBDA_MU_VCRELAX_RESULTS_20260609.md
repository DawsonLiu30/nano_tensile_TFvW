# TF+vW Lambda-Mu Cell-Relaxation Cross-Check

## Setup

- System: conventional cubic fcc Al, 4 atoms.
- XC: LDA.
- Local pseudopotential: `al.lda.recpot`.
- KEDF: `T_s = lambda_TF T_TF + mu_vW T_vW`.
- Grid: `lambda_TF, mu_vW = 0.1, 0.2, ..., 1.0` (100 pairs).
- DFTpy: atom plus hydrostatic-cell relaxation using
  `FrechetCellFilter` and BFGS.
- PROFESS: `MINI cell`, `KINE TF+`, `PARA LAMB`, and `PARA MU`.
- Stable fcc acceptance range: `3.0 <= a0 <= 6.0 A`.

The relaxation itself contains the required electronic density minimizations.
A final SCF/evaluation is performed only on the already-relaxed structure to
report total and kinetic-energy components; it does not change the geometry.

## Completion

| Code | Numerical relaxation completed | Stable fcc equilibrium |
|---|---:|---:|
| DFTpy | 100/100 | 79/100 |
| PROFESS | 97/100 | 79/100 |

Status agreement across all 100 coefficient pairs:

- 79: stable fcc in both codes.
- 18: no stable fcc equilibrium in both codes.
- 3: DFTpy relaxed to a collapsed/non-fcc solution; PROFESS stopped on its
  nearest-atom safety check.

## Cross-Code Agreement

For the 79 coefficient pairs with a stable fcc equilibrium in both codes:

| Quantity | Mean absolute difference | Maximum absolute difference |
|---|---:|---:|
| Total energy | 0.0000138 eV/atom | 0.0002236 eV/atom |
| KEDF energy | 0.004783 eV/atom | 0.028004 eV/atom |
| Lattice constant | 0.000385 A | 0.002238 A |

At `(lambda_TF, mu_vW) = (1, 1)`:

- DFTpy: `a0 = 4.048793 A`, `E = -57.46498899 eV/atom`.
- PROFESS: `a0 = 4.049263 A`, `E = -57.46499095 eV/atom`.

## Boundary Restart

For `(lambda_TF, mu_vW) = (0.2, 0.8)`, the first PROFESS run from
`a0 = 4.039848 A` followed a dissociation branch. Restarting from the DFTpy
stable-cell solution gave:

- DFTpy: `a0 = 3.009817 A`.
- PROFESS restart: `a0 = 3.010036 A`.

Both PROFESS attempts are retained in the raw case directory.

## Output Locations

- DFTpy:
  `C:\Users\dawso\Desktop\LOCAL_DFTPY_TFVW_LAMBDA_MU_CELL_RELAX_20260609`
- PROFESS:
  `C:\Users\dawso\Desktop\LOCAL_PROFESS_TFVW_LAMBDA_MU_CELL_RELAX_20260609`
- Cross-code table:
  `C:\Users\dawso\Desktop\TFVW_LAMBDA_MU_DFTPY_PROFESS_COMPARISON_20260609.csv`
