# DFTpy Conventional Vacancy Results

Date: 2026-05-25

These results replace the old primitive/rhombohedral DFTpy vacancy calibration
numbers. The previous `2.9-3.2 eV` values were withdrawn until the starting
structure issue was removed. The new calculations use VESTA-checked
conventional fcc cells.

## Main Result

The corrected conventional-cell DFTpy rerun still gives a high vacancy
formation energy.

| method | cell | atoms | setting | Ef_vac (eV) |
|---|---|---:|---|---:|
| QE | conventional `2x2x4` | `64 -> 63` | `5x5x5`, `600 eV` | `0.601085` |
| DFTpy / TFvW | conventional `2x2x4` | `64 -> 63` | spacing `0.20 A` | `2.900849` |

Same-cell discrepancy:

```text
2.900849 - 0.601085 = 2.299764 eV
```

Current wording for slides/report:

> After rebuilding the benchmark with VESTA-checked conventional fcc cells, the
> DFTpy/TFvW same-cell vacancy formation energy remains around 2.90 eV, while
> the best completed QE dense-k reference is about 0.60 eV. Therefore, the large
> QE-DFTpy discrepancy is not solely caused by the previous primitive-cell
> geometry issue.

Use the protection clause:

> The QE value is still described as the best completed dense-k reference rather
> than a fully converged final value, because the `6x6x6` k-point case and
> stricter QE force verification are still pending.

## Same-Cell DFTpy Spacing Scan

Cell:

- conventional fcc `2x2x4`
- `64 -> 63` atoms
- vacancy concentration: `1/64 = 1.5625%`
- target vacancy relaxation force: `fmax < 0.002 eV/A`

| spacing (A) | ecut analogue (eV) | Ef_vac (eV) | actual final fmax (eV/A) | status |
|---:|---:|---:|---:|---|
| `0.30` | `417.81` | `2.900725` | `0.001593` | pass |
| `0.25` | `601.65` | `2.900797` | `0.000858` | pass |
| `0.22` | `776.92` | `2.900844` | `0.001990` | pass |
| `0.20` | `940.08` | `2.900849` | `0.001016` | pass |
| `0.18` | `1160.59` | `2.900901` | `0.001119` | pass |
| `0.16` | `1468.87` | `2.901197` | `0.001231` | pass |

Spacing convergence:

```text
max difference from 0.30 A to 0.16 A = 0.000472 eV
```

Interpretation:

> The DFTpy conventional same-cell vacancy spacing scan is converged with
> respect to real-space spacing, and all completed vacancy relaxations satisfy
> the target `fmax < 0.002 eV/A`.

## DFTpy Size / Concentration Scan

Spacing fixed at `0.20 A`.

| cell | atoms | vacancy concentration | Ef_vac (eV) | actual final fmax (eV/A) | status |
|---|---:|---:|---:|---:|---|
| `conv_02x02x02` | `32 -> 31` | `3.125%` | `2.937948` | `0.001349` | pass |
| `conv_03x03x03` | `108 -> 107` | `0.925926%` | `2.897041` | `0.000437` | pass |
| `conv_04x04x04` | `256 -> 255` | `0.390625%` | `2.887601` | `0.001197` | pass |
| `conv_05x05x05` | `500 -> 499` | `0.200000%` | `2.883649` | `0.001896` | pass |
| `conv_06x06x06` | `864 -> 863` | `0.115741%` | not accepted | `0.004161` | not converged |

Accepted concentration effect from `32` to `500` pristine atoms:

```text
2.937948 - 2.883649 = 0.054299 eV
```

Interpretation:

> The vacancy concentration effect is visible but small compared with the
> QE-DFTpy same-cell discrepancy. Reducing the vacancy concentration from
> `3.125%` to `0.200%` lowers the DFTpy vacancy formation energy by about
> `0.054 eV`, while the QE-DFTpy same-cell discrepancy is about `2.30 eV`.

Do not include `conv_06x06x06` in the formal trend until it satisfies the target
force criterion and produces an accepted `result.json`/summary value.

## Remaining Actions

| item | status | action |
|---|---|---|
| QE `6x6x6` | pending | let the longer-walltime rerun finish |
| QE force verification | pending | do not claim strict `0.002 eV/A` until confirmed |
| DFTpy `conv_06x06x06` | not accepted | rerun with more time/steps or treat as attempted-only |
| PPT update | needed | replace old DFTpy pending/reset page with same-cell and concentration tables |
| local archive | needed | pull full DFTpy input/output/log files to Desktop and NAS package |
