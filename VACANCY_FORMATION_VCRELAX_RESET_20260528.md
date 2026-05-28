# Vacancy Formation Reset: 3x3x3 vc-relax Workflow

## Why the previous vacancy section is rejected

The previous vacancy-formation data used fixed-cell workflows:

- QE: pristine `scf` plus vacancy `relax`
- DFTpy: pristine single-point `evaluate_atoms()` plus vacancy fixed-cell ionic relaxation
- Main comparison cell: conventional `2x2x4`, 64 -> 63 atoms

After the professor's 2026-05-28 feedback, these are no longer acceptable as
final vacancy formation benchmarks. The final vacancy section must be rebuilt
with full relaxation.

## New accepted workflow

Use a VESTA-friendly conventional fcc Al supercell:

- conventional fcc `3x3x3`
- pristine: 108 atoms
- vacancy: 107 atoms
- vacancy concentration: `1/108 = 0.925926%`
- central Al atom removed
- both pristine and vacancy are fully relaxed

For QE this means literal:

```text
calculation = 'vc-relax'
```

For DFTpy this means an ASE/DFTpy full atom-and-cell relaxation using
`FrechetCellFilter`, recorded as:

```text
full_atom_and_cell_relaxation_vc_relax_equivalent
```

## New scripts

- `scripts/prepare_qe_vacancy_vcrelax_3x3x3.py`
- `scripts/collect_qe_vcrelax_vacancy.py`
- `scripts/run_dftpy_vcrelax_vacancy_one.py`
- `scripts/collect_dftpy_vcrelax_fmax.py`
- `run_dftpy_vcrelax_vacancy_one_ct56.sbatch`
- `submit_dftpy_vcrelax_conv3x3x3_ct56_array.sh`
- `scripts/push_vacancy_vcrelax_3x3x3_to_iservice.sh`
- `scripts/pull_vacancy_vcrelax_3x3x3_results.sh`

## Remote run order

1. Push scripts to iservice.
2. Prepare QE `3x3x3` vc-relax cases.
3. Submit QE vc-relax array.
4. Prepare DFTpy `3x3x3` full atom+cell relaxation cases.
5. Submit DFTpy full-relax spacing array.
6. Pull all raw input/output files back to local.
7. Collect summaries and only then update the final PPT.

Do not reuse the older `2x2x4` fixed-cell/scf vacancy numbers as final results.
