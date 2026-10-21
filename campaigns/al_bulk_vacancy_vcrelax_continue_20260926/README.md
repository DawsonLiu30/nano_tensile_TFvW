# Four-case QE continuation, prepared 2026-09-26

Prepared locally; deployment/submission status is recorded separately by the operator. Four cases in Slurm array order: 0=1V, 1=2V_1NN_D110, 2=2V_3NN_D211, 3=2V_5NN_D310. Run at most four concurrently (`0-3%4`), each 29 ranks / 64 GiB, Slurm 48 hours / QE 169200 seconds (47 hours).

Each starting cell and fractional coordinate set comes from the last verified completed electronic SCF step in its parent attempt. The parent endpoints were NOT accepted ionic minima: 1V stopped after a BFGS history/trust-radius failure; the other three reached the previous 39600-second budget during a subsequent incomplete SCF. The subsequent trial geometry is not used.

All four start new BFGS histories and fresh atomic electronic states, with new per-attempt prefixes and scratch. No wavefunction/density checkpoint is reused. BFGS remains the optimizer, including for 1V; force, SCF, pressure, energy, nstep, cutoffs, k points, pseudopotential and XC are unchanged. 1V may still require further diagnosis if fresh BFGS also stalls; it is not pre-labelled converged.

The new manifest supports independent per-case cells. Its hashes differ from the parent because the starting geometries and operational budget changed. `sources/<case>/` contains hashed parent input/output/attempt/manifest/geometry and aligned-step evidence. The continuation wrapper checks these against each other, verifies raw output line numbers and electronic convergence, and compares all scientific settings except geometry against the original parent input. Site mapping is retained. Every attempt snapshots both runner files and parent evidence in its immutable-file hash list.

Freeze this entire campaign plus `scripts/run_bulk_vacancy_qe_continue_20260926.py`, its dependency `scripts/run_bulk_vacancy_qe_vcrelax_20260925.py`, the continuation test, and `hpc/bulk_vacancy_qe_continue_20260926.sbatch`. Verify with the continuation runner, never the old eight-case campaign entry point. Use a new persistent output root and check current queues/submission records before submitting. Existing parent sources, outputs and snapshots remain unchanged.
