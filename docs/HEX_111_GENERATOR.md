# Reproducible FCC [111] hexagonal geometry

`app.hex_111.build_hex_111` is an **opt-in, geometry-only** generator. The
`hex_111` CLI builder is also exposed by `app/ase_nanocrystal.py`. Existing
`periodic_prism`, `paper_circular`, and `wulff_prism` geometry functions and
numerical defaults are unchanged. This is not a new relaxation, energy result,
convergence test, or claim of a perfect atomic boundary.

## Fixed conventions

Let `d=a0/sqrt(2)`, `h=a0/sqrt(3)`. Relative to the target center, site `(i,j,k)` is

```
xy = (-d/2, 0) + i*(d, 0) + j*(d/2, sqrt(3)*d/2)
     + (k % 3)*(0, a0/sqrt(6))
z  = k*h
```

This explicit Cartesian triangular-layer basis and ABC phase, not an arbitrary
choice for other axes, determines the Wigner-Seitz cell: vertices at
`30 + 60*m` degrees and circumradius `d/sqrt(3)`. The target hexagon has the
same vertex angles. Its input `radius_A` is **circumradius in Angstrom**, not
apothem, diameter, an `a0` multiplier, or an equal-area circular radius.
The target area is `3*sqrt(3)*radius_A**2/2`; this is a geometric target area,
not a measured electronic or atomistic volume.

Only this [111] orientation, fixed transverse phase, and fixed hexagon angle
are implemented. `abc_repeats` is a positive integer, so there are
`3*abc_repeats` layers and `Lz=3*abc_repeats*h`. There is no independent Lz
strain or rounding from a requested length. All three PBC flags are true;
x/y contain vacuum and z is the periodic column axis.

The square box has `Lx=Ly=2*(radius_A+vacuum)` and its center is the target
center. The generator does **not** recenter the selected atoms by their
centroid. `vacuum` is nominal padding around the target circumcircle; selected
Voronoi-intersection centers can lie outside that circle, so it is not a
guaranteed atom-to-image gap. Actual axis-projected image gaps are recorded.
Invalid/empty selections and centers outside the box are rejected; users
must separately validate adequate vacuum for their calculation.

## Selection rules

All cells are the exact analytic 2D Voronoi cells of the infinite triangular
layer; no finite-cloud cap is invented. These are not 3D atomic Voronoi cells.

- `center_inside`: center lies in all six closed target half-planes.
- `positive_area_intersection`: cell/target intersection area exceeds
  `area_tolerance_A2`; point/edge contact alone is not selected with the
  default positive numerical threshold.
- `full_containment`: every complete cell vertex lies in all target
  half-planes. Cells are selected or omitted, never physically clipped.

Defaults are `length_tolerance_A=1e-9` Angstrom for closed half-plane tests
and `area_tolerance_A2=1e-10` square Angstrom for positive area. Both are
explicit metadata. A zero area threshold disables the numerical safeguard
against floating-point near-tangency; use the default for reproducible cases.

## Python API

```python
from app.ase_nanocrystal import build_hex_111

atoms = build_hex_111(
    a0=4.0, radius_A=6.0, vacuum=5.0, abc_repeats=2,
    selection_mode="center_inside",
)
```

`a0` and `radius_A` are required in the Python API. CLI defaults of the legacy
generator are intentionally not changed; **pass `--a0` explicitly** for a
specific model. Parameters used for the reproducible geometry profile are:

| Parameter | Value |
|---|---|
| a0 (Angstrom) | 3.947472873 |
| R/a0 | 1.5, 1.7, 2.0 |
| radius_A (Angstrom) | 5.9212093095, 6.7107038841, 7.894945746 |
| abc_repeats | 2 (ABCABC) |
| Lz (Angstrom, approximate) | 13.6744471550718 = 6*a0/sqrt(3) |
| vacuum parameter (Angstrom) | 5.0 |

For example, from the repository root:

```bash
python app/ase_nanocrystal.py --builder hex_111 \
  --a0 3.947472873 --radius-A 5.9212093095 --vacuum 5 \
  --hex-abc-repeats 2 --hex-selection-mode center_inside \
  --output R15_center_inside.extxyz
```

Change only `--radius-A` to use the other listed sizes. The new builder uses
`--hex-abc-repeats`, not the legacy `--length-z`. Explicit legacy length,
shape, rotation, size, or gamma options are rejected instead of silently
ignored. It requires [111]; its own angle is fixed as above. It writes an ASE extxyz and a `.extxyz.json` sidecar. Extxyz keeps the
IDs/metadata; JSON keeps full Python-float positions and cell without ASE's
version-dependent display rounding. Repeating the same inputs in the same
environment reproduces both files. Record Python/NumPy/ASE versions for
cross-environment work; do not promise universal bitwise equality.

## Site identity and provenance

The integer arrays `lattice_i`, `lattice_j`, `layer_k` define the canonical
sites; output ordering is `(k,i,j)`. `generated_site_id` is an injective int64
encoding `Cantor(Cantor(zigzag(i),zigzag(j)),k)`, namespace `hex111_v1`.
These IDs stay the same for retained sites across target radii and rules;
they are **not** atom row indices, old `parent_site_id`, or relaxed-structure
IDs. A source-to-lattice mapping must be retained separately when transferring
historical vacancies. Changing lattice basis/phase would require a new
namespace, not reuse of these IDs.

The function regenerates ideal coordinates from the explicit formulas; it
does not load or preserve decimal rounding from older structure files. A
historical file that deliberately retained input doubles may differ by its
stored-coordinate precision even when the selected lattice sites agree.
Do not overwrite frozen structures or substitute these IDs into old vacancy
maps. Geometry metadata also records rule, tolerances, basis, target center,
box convention, analytic WS method, and per-atom clipped area.

Implementation is independently written from the stated triangular-lattice
geometry and convex half-plane clipping. No code from uploaded builders or
third-party projects was copied. It uses the repository's existing NumPy/ASE
dependencies; SciPy is used only by the independent Qhull regression tests.
No pseudopotential, private case archive, energy result, or credential is
needed to run these tests.

## Verification

```bash
python -m unittest discover -s tests -p test_hex_111.py -v
```

Tests cover all three rules against bounded Qhull cells, enlarged mother
lattices, tangency/containment, deterministic identities and output,
scale covariance, ABC periodicity, axial seam neighbours, connectivity,
no duplicates or spurious lateral nearest neighbours for the synthetic
fixtures, and legacy circle-mask/wrapper behavior. These checks certify
implementation behavior for the tested geometry, not physical convergence.

This additive change is based on the verified research-branch commit
`505b9fc30d8d8a61f3efe95c38b69709e2afeeb8`. It does not synchronize uncommitted
local research changes. Review and reconcile those separately before merging;
no existing campaign/workflow snapshot is modified.
