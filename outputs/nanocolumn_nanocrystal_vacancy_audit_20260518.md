# Nanocolumn / Nanocrystal Geometry and Vacancy Formula Audit

Date: 2026-05-18

## 1. Main Decision

We do not need to delete everything from the proposal data. The latest geometry audit shows that the main `Al_col_*` and `vacancy_nanocrystal_relax/Al_vac_*` structures are periodic along the axial z direction, not finite-length clusters.

The corrected definition is:

- Nanocolumn: axially periodic/infinite model with a circular xy cross-section.
- Nanocrystal / faceted nanostructure: axially periodic/infinite model with a non-circular xy cross-section, e.g. faceted, hexagonal, or triangular.
- The difference is cross-section shape, not finite versus infinite length.

Therefore:

- Existing axially periodic circular cases can be kept as nanocolumn data.
- The audited `vacancy_nanocrystal_relax/Al_vac_*_zr*` cases can be kept, but should be renamed in the report as vacancy-containing faceted nanocolumns or prismatic nanostructures with faceted cross sections.
- New nanocrystal runs are only needed if we need additional clean hexagonal/triangular cross-section datasets beyond the audited faceted cases.

## 2. Geometry Audit Result

The latest structure audit scanned 95 structures and returned:

```text
95 OK_or_periodic_z
0 DANGER_z_vacuum_finite_length
maximum gap_z = 2.338 A
```

Representative audited structures include:

- `Al_col_111_r4.0_nz1_n30/POSCAR`
- `Al_col_111_r6.0_nz1_n64/POSCAR`
- `Al_col_111_r8.0_nz1_n104/POSCAR`
- `vacancy_nanocrystal_relax/Al_vac_111_r4.0_zr4_n55/...`
- `vacancy_nanocrystal_relax/Al_vac_111_r6.0_zr1_n33/...`
- `vacancy_nanocrystal_relax/Al_vac_111_r8.0_zr2_n103/...`

Interpretation:

- `gap_z = 2.338 A` is consistent with Al interlayer spacing, not axial vacuum.
- `zr1`, `zr2`, and `zr4` should be interpreted as z-repeat counts, not finite cluster lengths.
- The old slide label "vacancy nanocrystals" should be revised to "vacancy-containing faceted nanocolumns" or "vacancy-containing prismatic nanostructures".

## 3. Gillan 1989 Vacancy Formation Energy Formula

Gillan section 2.3 defines vacancy formation energy using a perfect system with N atoms on N lattice sites and a defective system with N - 1 atoms plus one vacancy on the same number of lattice sites.

The working supercell formula is:

```text
E_f^vac = E_vac^(N-1) - ((N-1)/N) E_pristine^N
```

Equivalently:

```text
E_f^vac = E_vac^(N-1) - E_pristine^N + E_pristine^N / N
```

Important correction:

- Correct scaling factor: `(N - 1) / N`, e.g. `63 / 64` for a 64-atom pristine cell.
- Incorrect scaling factor: `N / (N - 1)`, e.g. `64 / 63`.
- `E_vac - E_pristine + mu_bulk` is a different bulk-reservoir removal energy. It may be useful as a diagnostic but should not be the primary vacancy formation energy for nanostructures unless that reservoir definition is explicitly intended.

## 4. Code Formula Audit

The bulk periodic vacancy workflows already use the Gillan-style scaled formula correctly:

- `scripts/collect_all_qe_vacancy_recursive.py`
- `scripts/run_dftpy_primitive_size_one.py`
- `scripts/run_dftpy_vacancy_convergence.py`
- `scripts/run_bulk_vacancy_supercell_series.py`

The nanostructure vacancy preparation scripts previously used the bulk-reservoir formula as the main value. They have now been corrected to store both definitions:

- `formation_energy_same_geometry_eV`
- `formation_energy_gillan_scaled_eV`
- `formation_energy_bulk_reservoir_eV`

Updated scripts:

- `scripts/prepare_vacancy_periodic_wire.py`

Syntax check passed:

```text
python -m py_compile scripts\prepare_vacancy_periodic_wire.py scripts\collect_all_qe_vacancy_recursive.py scripts\run_dftpy_primitive_size_one.py scripts\run_dftpy_vacancy_convergence.py scripts\run_bulk_vacancy_supercell_series.py
```

## 5. Old Data Salvage Assessment

### Axially periodic circular cases

These can be kept as nanocolumn data, not nanocrystal data.

Recomputed from old manifests using the Gillan-style same-geometry formula:

| Case | N pristine | N vacancy | Old bulk-reservoir value (eV) | Corrected same-geometry value (eV) |
|---|---:|---:|---:|---:|
| `paper_periodic_111_1.0nm_vacancy_tfvw` | 60 | 59 | -1.128207 | 1.045027 |
| `paper_periodic_111_2.0nm_vacancy_tfvw` | 242 | 241 | -0.598695 | 0.756416 |

Interpretation:

- The old negative values were not reliable vacancy formation energies.
- The corrected values are usable as preliminary nanocolumn vacancy results if the geometry is axially periodic.

### Audited vacancy faceted cases

The latest z-gap audit indicates that `vacancy_nanocrystal_relax/Al_vac_*_zr*` structures are z-periodic and can be kept.

Recommended relabeling:

```text
old label: Vacancy nanocrystals
new label: Vacancy-containing faceted nanocolumns
alternative: Vacancy-containing prismatic nanostructures with faceted cross sections
```

## 6. Recommended Recompute Scope

Do not recompute all historical proposal data blindly.

The current audit suggests:

- Keep the audited `Al_col_*` pristine circular nanocolumn data.
- Keep the audited `vacancy_nanocrystal_relax/Al_vac_*_zr*` data after relabeling as faceted periodic nanostructures.
- Recompute only if a case lacks a matching pristine/vacancy pair, has failed relaxation, has a bad z-audit verdict, or if the professor specifically wants clean hexagon/triangle shape-controlled models.

If new shape-controlled reruns are required, the minimum useful set is:

```text
circle: pristine + vacancy, selected diameters already used in proposal
hexagon: pristine + vacancy, same selected diameters
triangle: pristine + vacancy, optional but useful if professor wants multiple nanocrystal shapes
```

## 7. Reporting Language

Recommended wording:

```text
Geometry audit confirmed that the pristine nanocolumn and vacancy-containing structures used in the static energy comparison are periodic along the axial z direction.
The maximum empty interval along z is only 2.338 A, corresponding to interlayer spacing rather than axial vacuum.
Therefore, these structures are not finite-length models.
The previous "vacancy nanocrystal" terminology is revised to "vacancy-containing faceted nanocolumns" or "prismatic nanostructures with faceted cross sections".
Vacancy formation energies are reported using the Gillan-style same-geometry supercell reference:
E_f^vac = E_vac^(N-1) - ((N-1)/N) E_pristine^N.
```

## 8. Single-Line Commands for Optional New Shape-Controlled Runs

Nanocolumn, circular:

```powershell
python scripts\run_periodic_series.py --diameters 1.0,2.0,3.0,4.0 --cross-section-shape circle --cycles 20
```

```powershell
python scripts\run_vacancy_periodic_series.py --diameters 1.0,2.0,3.0,4.0 --cross-section-shape circle --cycles 20
```

Nanocrystal, hexagonal:

```powershell
python scripts\run_periodic_series.py --diameters 1.0,2.0,3.0,4.0 --cross-section-shape hexagon --cycles 20
```

```powershell
python scripts\run_vacancy_periodic_series.py --diameters 1.0,2.0,3.0,4.0 --cross-section-shape hexagon --cycles 20
```

Nanocrystal, triangular:

```powershell
python scripts\run_periodic_series.py --diameters 1.0,2.0,3.0,4.0 --cross-section-shape triangle --cycles 20
```

```powershell
python scripts\run_vacancy_periodic_series.py --diameters 1.0,2.0,3.0,4.0 --cross-section-shape triangle --cycles 20
```
