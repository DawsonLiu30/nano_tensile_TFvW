# CT56 finite-grip TFvW tensile runbook

This project root is expected on iService at:

```bash
/gpfs-work/dawson666/dftpy_project/relax/dftpy45
```

The production workflow is the rigid finite-grip tensile machine:

- Al [111] finite nanowire.
- Top and bottom grip atoms are translated as rigid bodies during loading.
- Grip atoms are fixed during each relaxation.
- One vacancy is selected in the central free region, outside the fixed grips, near the outer surface.
- Each tensile cycle applies 1 percent engineering strain by grip separation unless `STEP` is changed.
- Complete fracture is marked by `max atomic z-gap > 3*d111` or a major disconnected cluster.
- The main stress for event analysis is the offset grip-reaction stress.

## Recommended rsync

From Windows/WSL, sync the project root to the ct56 working directory. Keep `cases/` if you want to resume r=2 from local results; otherwise the cluster will start fresh.

```bash
rsync -av --info=progress2 /mnt/c/Users/dawso/nano_tensile_TFvW/ \
  dawson666@iservice.nchc.org.tw:/gpfs-work/dawson666/dftpy_project/relax/dftpy45/
```

## Start the recommended parallel array run

Default behavior submits one Slurm array task per diameter. It resumes r=2 if a previous `summary.csv` exists, and prepares/runs r=3 through r=8 independently. r=1 is not included by default because it already has a completed local run.

`MAX_PARALLEL` controls how many diameters can run at the same time. The updated array workflow now defaults to `CYCLES_PER_LAUNCH=1`, so each Slurm task repeatedly restarts a short tensile chunk instead of keeping one long-lived Python/DFTpy process in memory for the whole run. This is meant to avoid the slow memory buildup that caused the earlier `OUT_OF_MEMORY` failures for `r=3-5`.

You can also override `MEM`, `TIME_LIMIT`, `CPUS`, `PARTITION`, `SPACING`, and `ECUT` directly from the submission command.

```bash
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45
bash submit_finite_grip_array_ct56.sh
```

To run only r=3 to r=8:

```bash
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45
DIAMETERS=3,4,5,6,7,8 MAX_PARALLEL=3 bash submit_finite_grip_array_ct56.sh
```

To rerun the failed larger diameters more safely after the `120G` OOM events:

```bash
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45
DIAMETERS=3,4,5 MAX_PARALLEL=1 MEM=220G TIME_LIMIT=4-00:00:00 CYCLES_PER_LAUNCH=1 bash submit_finite_grip_array_ct56.sh
```

To continue r=2 to r=8 but use fewer cycles for a scout run:

```bash
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45
CYCLES=30 DIAMETERS=2,3,4,5,6,7,8 MAX_PARALLEL=3 bash submit_finite_grip_array_ct56.sh
```

If queue/account limits allow more simultaneous jobs:

```bash
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45
MAX_PARALLEL=5 bash submit_finite_grip_array_ct56.sh
```

If memory pressure persists, try a clearly labeled coarser-grid scout run before touching the production settings:

```bash
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45
DIAMETERS=5,6,7,8 MAX_PARALLEL=1 MEM=220G CYCLES_PER_LAUNCH=1 SPACING=0.220 bash submit_finite_grip_array_ct56.sh
```

## Conservative sequential fallback

Use this only if the cluster/account policy dislikes job arrays or if too many concurrent DFTpy jobs stress the filesystem:

```bash
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45
sbatch run_finite_grip_series_ct56_4d.sbatch
```

## Run one diameter

```bash
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45
DIAMETER=3 sbatch run_finite_grip_one_ct56.sbatch
```

To override the Slurm resources for one large-diameter restart:

```bash
cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45
DIAMETER=3 CYCLES_PER_LAUNCH=1 sbatch --mem=220G --time=4-00:00:00 run_finite_grip_one_ct56.sbatch
```

## Check progress

```bash
squeue -u dawson666
tail -f logs_ctest/fgTFvWa_<array_jobid>_<taskid>.out
ls -lh results/hpc_status/
tail -f results/hpc_status/r2_job<array_jobid>_0.csv
```

Each finished or in-progress case writes to:

```text
cases/finite_grip_111_<r>.0nm_vacancy_tfvw/results/grip_r<r>_vacancy_tfvw_<timestamp>/
```

The postprocessed plots are written to:

```text
results/professor_review/
```

The event-analysis files are copied/generated as:

```text
tensile_events.csv
tensile_event_summary.json
fracture_status.csv
summary.csv
```

## Notes

The r=3 and larger grids can hit `OUT_OF_MEMORY` at `120G`, not necessarily because one individual relaxation is impossible, but because repeated cycles in one Python process can accumulate memory. The chunked relaunch workflow is therefore preferred for large diameters.

If a large case still fails from memory, rerun only the failed diameter with more memory if the partition allows it. Only reduce `ECUT` or increase `SPACING` for a clearly labeled sensitivity/scout run.
