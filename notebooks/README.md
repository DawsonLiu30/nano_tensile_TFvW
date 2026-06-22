# Vacancy Workflow Notebooks

These notebooks demonstrate the professor-facing workflow without duplicating
the complete repository inside a data package.

Run them from the repository root or update the path variables in the first code
cell.

| Notebook | Purpose |
|---|---|
| `01_generate_divacancy_structures.ipynb` | Generate the pristine and same-height divacancy structures. |
| `02_read_dftpy_outputs_compute_formation_energy.ipynb` | Read DFTpy `result.json` files and verify the formation-energy formula. |
| `03_compare_dftpy_qe_divacancy.ipynb` | Join DFTpy and QE pair scans and plot the non-redundant two-vacancy formation energy. |

The production scripts remain under `scripts/` and are the authoritative code.

