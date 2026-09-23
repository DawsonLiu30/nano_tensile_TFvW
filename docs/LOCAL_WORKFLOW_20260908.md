# OFDFT Al vacancy research — current entry, 2026-09-08

This is the editable project on the new computer: `C:\OFDFT\nano_tensile_TFvW`.
The migrated raw datasets and USB remain the historical baseline. The current
project supports bulk, monovacancy and direction-resolved divacancy analysis.
The tensile/nanowire studies and dated proposal documents are earlier work;
they do not establish that a final thesis or a validated nanowire model exists.

## Current divacancy result

Use `C:\OFDFT\DFTPY_DIVACANCY_D110_L0p9_M0p1_RERUN_20260831`.
The protocol is recorded in `RESEARCH_PROTOCOL.json`.

| Initial minimum-image distance (Å) | Total two-vacancy formation energy (eV) |
|---:|---:|
| 2.796310621839333 | 1.2783895060420036 |
| 5.592621243678668 | 1.332160884008772 |
| 8.388931865518002 | 1.3349384808188915 |

All three removed-site vectors lie on the [110] axis, up to reversal and tied
periodic images. `[110]` denotes a crystal direction; `(110)` denotes a plane.
The second *sample* along [110] is not the second nearest-neighbour shell of
fcc Al. A true second-neighbour [100] calculation already exists in the
20260813 curated package and must be analysed as a separate direction.

The three points remove the dip from the corrected [110] curve. They do not
demonstrate that all orientations, larger cells, grid convergence, binding
energies or QE cross-validation have been completed. The farthest point has
two components at half the cell length; it is not an isolated-pair limit.

## Start on this computer

In PowerShell:

```powershell
& C:\OFDFT\LOCAL_MACHINE_20260907\OFDFT.ps1 Status
& C:\OFDFT\LOCAL_MACHINE_20260907\OFDFT.ps1 DivacancyStatus
& C:\OFDFT\LOCAL_MACHINE_20260907\OFDFT.ps1 DivacancyValidate
& C:\OFDFT\LOCAL_MACHINE_20260907\OFDFT.ps1 Shell
```

`DivacancyStatus` describes the prepared local calculation folder. The
20260907 prepared folder contains inputs, not the August completed results.
The corrected result source is exported as `AL_DEFECTS_DIVACANCY_ROOT`.

After editing Windows source, run `OFDFT.ps1 Sync`. It verifies every copied
file and retains the previous WSL execution copy in
`/var/tmp/al-defects-runtime-20260907/repo-history/<UTC timestamp>`.
Managed runners prevent concurrent source replacement. Saved scratch results
must also be copied to durable Windows storage before removing a WSL runtime.

`DivacancyRun` explicitly starts expensive production calculations.
Use a new `RUN_ROOT` inside the runtime `runs` directory to prepare a different
protocol. An existing folder is never overwritten by preparation. Resume
rejects parameter overrides that disagree with input manifests. Failed or
partial attempts are archived before retries. One local case runs at a time
by default with one BLAS/OpenMP thread per process.

## Reproduce analysis

In the provided WSL shell:

```bash
python -m unittest discover -s tests -v
python scripts/collect_dftpy_conventional_vacancy.py \
  --rootdir "$AL_DEFECTS_DIVACANCY_ROOT" \
  --output /mnt/c/OFDFT/LOCAL_ANALYSIS/divacancy_20260908
```

Analysis outputs go to the explicitly selected directory (otherwise
`analysis_current` beneath the supplied root). Always supply an output outside
the immutable historical dataset when reviewing it. The three notebooks in
`notebooks/` demonstrate geometry generation, evidence checks and a gated QE
comparison. They do not silently launch a production campaign.

`qualified` requires consistent energies, initial and final structures,
pseudopotential evidence and the recorded ASE combined atom/cell force test
for **both** pristine and defect. Missing results, missing evidence, malformed
results and unconverged cases are excluded from qualified plotted series.
The `*_dftpy.out` files produced by this project are energy/stress summaries;
they are not complete electronic-optimization transcripts. Historical
electronic convergence is not certified by that status. New executions also
retain `local_runner.log`, `run_provenance.json` and exact ASE convergence
metadata, with the electronic-convergence limitation explicitly recorded.

## Research boundaries

- The canonical 100-point monovacancy sweep contains 83 `result.json` files,
  not 100 completed pairs. The 91 successful pristine-log entries use a
  different denominator. Completion and force qualification are separate.
- Monovacancy and current divacancy grids, optimizers, force targets and
  references differ. Their energies must not be combined into a formal
  binding energy without matched and converged protocols.
- The canonical QE monovacancy reference uses PBE/PAW, a 3×3×3 k-point mesh
  and its own convergence criteria. It is not an LDA divacancy validation.
- Dated reports, decks, emails and old calculation scripts are provenance.
  This README and the current protocol supersede their operational defaults;
  a new audit does not retroactively alter an old calculation.

The comprehensive evidence index, advisor requirement matrix and verification
report are delivered under the task's `outputs` directory. Missing convergence
studies and scientific decisions must be resolved before a thesis is described
as complete or ready for submission.
