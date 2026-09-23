# Gillan-Style Single-Vacancy Lambda-Mu Analysis - 2026-06-23

Local snapshot:

```text
C:\Users\dawso\Desktop\ISERVICE_AL_DEFECTS_SNAPSHOT_20260623_194929\dftpy45_snapshot
```

Analysis folder:

```text
C:\Users\dawso\Desktop\ISERVICE_AL_DEFECTS_SNAPSHOT_20260623_194929\dftpy45_snapshot\_analysis_gillan_style
```

## Main Formula

The current DFTpy single-vacancy workflow already follows the Gillan-style
perfect/defective supercell construction.

For every lambda-mu point:

```text
perfect system   = pristine_raw.vasp  = Al108
defective system = vacancy_start.vasp = Al107
```

Total vacancy formation energy:

```text
E_f^vac = E_vac(Al107) - (107/108) E_pristine(Al108)
```

KEDF contribution to vacancy formation energy:

```text
KEDF_f^vac = KEDF_vac(Al107) - (107/108) KEDF_pristine(Al108)
```

The relaxed pristine lattice constant is defined as:

```text
a0 = relaxed pristine 3x3x3 cell length / 3
```

## Current Completion

For the coarse 10x10 single-vacancy lambda-mu matrix:

```text
total points        = 100
has result.json     = 78
qualified points    = 42
unqualified points  = 36
python_failed       = 18
running/stale lock  = 1
no result/status    = 3
```

The `python_failed` entries are mostly from invalid job `1575328`, which failed
before entering DFTpy due to a shell `timeout` formatting error. They should not
be interpreted as physics/convergence failures.

## Best Points by Vacancy Formation Energy

QE corrected 3x3x3 PBE dense-k reference used here:

```text
E_f^vac(QE, k=5x5x5, 800 eV) = 0.638912226 eV
```

Closest qualified DFTpy point:

| lambda | mu | E_f^vac (eV) | difference from QE (eV) | a0 (A) | KEDF_f^vac (eV) |
|---:|---:|---:|---:|---:|---:|
| 0.9 | 0.1 | 0.661660746 | +0.022748519 | 3.954580406 | -2.791795323 |

This point matches the vacancy formation energy well, but the relaxed pristine
lattice constant is too small relative to the QE/literature starting value
`a0 = 4.039848 A`.

## Best Points by Lattice Constant

Closest qualified points to `a0 = 4.039848 A`:

| lambda | mu | a0 (A) | a0 - target (A) | E_f^vac (eV) |
|---:|---:|---:|---:|---:|
| 1.0 | 0.7 | 4.041846656 | +0.001998656 | 2.593729066 |
| 1.0 | 0.6 | 4.042482013 | +0.002634013 | 2.322212598 |
| 1.0 | 0.8 | 4.043370258 | +0.003522258 | 2.841851601 |

These points reproduce the lattice constant well, but their vacancy formation
energies are much higher than the QE reference.

## Current Interpretation

The current qualified coarse scan suggests a strong tradeoff:

- Matching `E_f^vac` alone favors approximately `lambda=0.9, mu=0.1`.
- Matching pristine `a0` favors approximately `lambda=1.0, mu=0.6-0.8`.
- No currently qualified coarse point simultaneously matches both the QE
  vacancy formation energy and the pristine lattice constant.

This means the next discussion should not present a single final lambda-mu value
yet. The safe statement is:

```text
The Gillan-style vacancy formation workflow is correctly implemented, but the
coarse TFvW lambda-mu scan shows a tradeoff between vacancy formation energy and
relaxed lattice constant. Selected points require stricter refinement before a
final parameter choice.
```

## Professor-Facing Files

Use these generated files for the current discussion:

```text
_analysis_gillan_style/professor_three_metrics_long.csv
_analysis_gillan_style/vacancy_lambda_mu_gillan_long_summary.csv
_analysis_gillan_style/matrix_vacancy_formation_energy_eV_qualified_only.csv
_analysis_gillan_style/matrix_kedf_formation_energy_eV_qualified_only.csv
_analysis_gillan_style/matrix_lattice_constant_A_qualified_only.csv
```

The `all_completed` matrices can be used for diagnostics, but professor-facing
plots should use `qualified_only` unless explicitly discussing failed or
unqualified regions.

## Recommended Next Step

Wait for the remaining coarse points to finish or be classified, then refine only
selected candidate regions with the stricter final force threshold.

Recommended refinement candidates:

```text
vacancy-energy candidate: lambda=0.9, mu=0.1
lattice candidates:      lambda=1.0, mu=0.6-0.8
near-compromise checks:  lambda=0.9-1.0, mu=0.1-0.4
```

