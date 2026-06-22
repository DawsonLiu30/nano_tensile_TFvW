# Divacancy Energy And PBC Definitions

## Structure Names

```text
pristine_raw.vasp       N-atom pristine structure
divacancy_start.vasp    (N-2)-atom structure before relaxation
divacancy_vc_relaxed    (N-2)-atom structure after full atom+cell relaxation
```

`vacancy_start.vasp` is reserved for a one-vacancy calculation. Older pilot
packages used it as a generic defect alias for the divacancy structure; new
packages do not create this duplicate alias.

## Formation Energy

For a divacancy calculation with `N` pristine atoms:

```text
E_2vac(r) = E_defect(N-2, r) - ((N-2)/N) E_pristine(N)
```

This is the main two-vacancy formation energy plotted against distance.

`E_2vac/2` is only a normalized per-vacancy value. It is redundant with
`E_2vac` and should not appear as a second main curve.

## Binding Or Relative Energy

If a separately converged one-vacancy reference is available:

```text
E_bind(r) = E_2vac(r) - 2 E_1vac
```

With this sign convention, a negative value indicates that the pair is lower
in energy than two isolated vacancies.

If a sufficiently separated one-vacancy reference is not available, only a
finite-cell relative energy may be reported:

```text
Delta E_2vac(r) = E_2vac(r) - E_2vac(r_farthest)
```

This must not be labeled as an infinite-separation binding energy.

## Periodic Distance

The reported pair distance is the minimum-image distance under periodic
boundary conditions:

```text
Delta s = s_2 - s_1 - round(s_2 - s_1)
r = |Delta s H|
```

`s_1` and `s_2` are fractional coordinates and `H` is the cell matrix.

For an orthogonal cubic cell with side `L`, each Cartesian component is limited
to `L/2`, but the vector length can reach:

```text
r_max = sqrt(3) L / 2
```

Therefore an `8.57 A` minimum-image distance is geometrically possible in a
cell with `L ~= 12.12 A`. It does not mean that one Cartesian displacement is
larger than half the cell.

For the current `3x3x3` pilot, the `8.57 A` vector is approximately
`(L/2, L/2, 0)`, whose length is `L/sqrt(2)`.

## Orientation Must Not Be Hidden

The pilot points are not all along one crystallographic direction:

```text
2.8566 A  [110]
4.0398 A  [100]
5.7132 A  [110]
6.3876 A  [310]
8.5698 A  [110]
```

Consequently, a scalar-distance plot mixes distance dependence and orientation
dependence. The small drop at `6.3876 A` occurs at the `[310]` point and must
not be interpreted as a broken `[110]` radial trend. Future figures should show
the direction family or use separate fixed-direction scans.
