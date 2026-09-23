# Divacancy notebooks

Run with the project's rebuilt Python environment from the repository/notebooks directory. If opened elsewhere, set `AL_DEFECTS_REPO` to the full repository path. Windows paths in WSL must use `/mnt/c/...`.

1. `01_generate_divacancy_structures.ipynb`: corrected 3×3×3 fixed-[110] initial structures; writes a new timestamped demonstration directory only. Override its base using `AL_DEFECTS_NOTEBOOK_OUTPUT`.
2. `02_read_dftpy_outputs_compute_formation_energy.ipynb`: read-only raw-evidence and combined atom/cell convergence checks. Default is the corrected 20260831 package; override using `AL_DEFECTS_DIVACANCY_ROOT`.
3. `03_compare_dftpy_qe_divacancy.ipynb`: gated comparison. Without independently audited QE evidence it reports that comparison is unavailable. `AL_DEFECTS_QE_DIVACANCY_AUDIT` can point to a reviewed CSV. See `compatible_comparison` in `scripts/divacancy_analysis_checks.py` for required fields.

Qualification means the checked numerical evidence passes; finite-size convergence, model validation and thesis acceptance are separate. Distances are initial minimum-image distances. Do not join mixed crystal directions, mix grid spacings for binding energies, or infer a QE result from a DFTpy/monovacancy table.
