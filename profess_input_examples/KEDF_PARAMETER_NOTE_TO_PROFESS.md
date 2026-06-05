# KEDF Parameter Note for Local PROFESS Checks

## Short answer

The `lambda` and `mu` in my PROFESS inputs are not vacancy-formation-energy
chemical potentials. They are PROFESS KEDF parameters:

- `PARA LAMB` = coefficient for the Thomas-Fermi part.
- `PARA MU` = coefficient for the von Weizsaecker part.

This follows the PROFESS3 manual description of the `KINE` and `PARA`
keywords. In particular, `KINE TF+` is the sum of Thomas-Fermi times
`PARA LAMB` and von Weizsaecker times `PARA MU`.

## What I actually used

### TFPLUS_DEFAULT

Input:

```text
ecut 1600
meth NTN
KINE TF+
exch lda
geometryfile STRUCTURE.ion
print minimizer density 2
```

Here I did **not** manually set `PARA LAMB` or `PARA MU`; this run uses the
internal PROFESS default parameters for `KINE TF+`.

### TFVW_L1_M1

Input:

```text
ecut 1600
meth NTN
KINE TF+
PARA LAMB 1
PARA MU 1
exch lda
geometryfile STRUCTURE.ion
print minimizer density 2
```

This is the only case where I explicitly set lambda and mu. It was not used as
a recommended Al KEDF. It was a controlled diagnostic to reproduce a
TFvW-like setting with full Thomas-Fermi plus full von Weizsaecker
contribution.

## Important clarification

There are two different "mu" symbols in this project:

- `PARA MU` in PROFESS: a KEDF mixing coefficient for the von Weizsaecker term.
- `mu = E_pristine / N` in vacancy-formation-energy analysis: the bulk atomic
  chemical potential/reference energy.

These are unrelated and should not be mixed.

## Why WGC/CAT/HC were not used as the main radius plot

WGC, CAT, and HC were meaningful in the bulk-vacancy KEDF screening, but in the
vacuum-padded nanostructure radius sweep they did not produce complete valid
total-energy sets under the present simple SCF-only input:

- plain WGC produced no valid `TOTAL ENERGY` for the 11 nanostructure cases and
  showed divergent behavior in the output;
- CAT and HC were incomplete/too slow under the current setup;
- therefore they were retained only as diagnostic attempts, not as mainline
  radius-trend data.

This does not mean those KEDFs are universally invalid. It only means they were
not robust enough for a directly comparable full-radius data set with the
current vacuum nanostructure input. The PROFESS manual also lists vacuum-
stabilized variants such as WTV/WGV/CAV, which would be the next thing to test
if nonlocal KEDFs are required for vacuum-padded nanostructures.
