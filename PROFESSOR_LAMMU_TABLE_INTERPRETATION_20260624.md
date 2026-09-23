# Professor Lambda-Mu Table Interpretation - 2026-06-24

Local package:

```text
C:\Users\dawso\Desktop\PROFESSOR_LAMMU_TABLE_GILLAN_STYLE_20260624
```

Main workbook:

```text
C:\Users\dawso\Desktop\PROFESSOR_LAMMU_TABLE_GILLAN_STYLE_20260624\09_professor_table\professor_lambda_mu_table_gillan_style.xlsx
```

## File Check

The professor table package was pulled locally.

Key files:

| file | size |
|---|---:|
| `professor_table_gillan_style_20260624.tar.gz` | 37,515 bytes |
| `professor_lambda_mu_table_gillan_style.xlsx` | 27,716 bytes |
| `professor_lambda_mu_table_flat_results.csv` | 37,890 bytes |
| `professor_lambda_mu_table_gillan_style.csv` | 3,522 bytes |
| `professor_lambda_mu_table_status.csv` | 2,154 bytes |

Remote workbook validation passed with `openpyxl`:

```text
Sheets: Professor_Table, Flat_Results, Status_Matrix, Notes
Professor_Table: 12 rows x 37 columns
Flat_Results: 101 rows x 17 columns
Status_Matrix: 11 rows x 11 columns
Notes: 6 rows x 2 columns
```

## What The Table Contains

The professor-style wide table contains three side-by-side matrices:

1. `Total`: vacancy formation energy, `E_f^vac`, in eV.
2. `KEDF`: KEDF contribution to vacancy formation energy, in eV.
3. `lattice constant`: relaxed pristine `a0`, in Angstrom.

This is not raw total energy. It is Gillan-style vacancy formation energy:

```text
E_f^vac = E_vac(Al107) - (107/108) E_pristine(Al108)
```

KEDF is treated consistently:

```text
KEDF_f^vac = KEDF_vac(Al107) - (107/108) KEDF_pristine(Al108)
```

Formula check against `pristine_energy_eV` and `vacancy_energy_eV`:

```text
checked numeric rows = 83
maximum absolute formula error = 0
```

## Status Summary

There are 100 lambda-mu points total.

| status | count |
|---|---:|
| qualified | 42 |
| has_result_but_not_qualified | 41 |
| slow_pristine_not_converged | 12 |
| pathological_fmax | 2 |
| killed_bad_fmax_pristine | 1 |
| killed_bad_fmax_vacancy | 1 |
| bad_vacancy_relaxation | 1 |

Only `qualified` points are shown as numeric values in the main professor table.
Non-qualified points are shown as `UNQ`.

## Best Point By Vacancy Formation Energy

QE corrected 3x3x3 reference:

```text
E_f^vac(QE/PBE, k=5x5x5, 800 eV) = 0.638912226 eV
```

Closest qualified DFTpy point:

| lambda | mu | E_f^vac (eV) | difference from QE (eV) | a0 (A) | KEDF_f^vac (eV) |
|---:|---:|---:|---:|---:|---:|
| 0.9 | 0.1 | 0.661660746 | +0.022748519 | 3.954580406 | -2.791795323 |

This point matches vacancy formation energy well, but the relaxed lattice
constant is too small.

## Best Points By Lattice Constant

Reference starting lattice constant:

```text
a0 = 4.039848 A
```

Closest qualified points:

| lambda | mu | a0 (A) | a0 - reference (A) | E_f^vac (eV) |
|---:|---:|---:|---:|---:|
| 1.0 | 0.7 | 4.041846656 | +0.001998656 | 2.593729066 |
| 1.0 | 0.6 | 4.042482013 | +0.002634013 | 2.322212598 |
| 1.0 | 0.8 | 4.043370258 | +0.003522258 | 2.841851601 |

These points match the lattice constant well, but their vacancy formation
energies are much too high compared with QE.

## Interpretation

The table is internally consistent and professor-readable.

The scientific result is a tradeoff:

- Matching vacancy formation energy favors approximately `lambda=0.9, mu=0.1`.
- Matching lattice constant favors approximately `lambda=1.0, mu=0.6-0.8`.
- No current qualified coarse-grid point simultaneously matches both `E_f^vac`
  and `a0`.

Safe wording:

```text
The Gillan-style perfect/defective vacancy formation workflow is correctly
implemented. The current coarse lambda-mu scan shows a tradeoff between fitting
the vacancy formation energy and fitting the relaxed pristine lattice constant.
Therefore, selected candidate regions should be refined before claiming a final
lambda-mu choice.
```

## Recommended Next Step

Use this Excel as the professor-facing summary table.

For the next calculation/refinement round:

```text
Ef candidate:       lambda=0.9, mu=0.1
lattice candidates: lambda=1.0, mu=0.6-0.8
compromise checks:  lambda=0.9-1.0, mu=0.1-0.4
```

