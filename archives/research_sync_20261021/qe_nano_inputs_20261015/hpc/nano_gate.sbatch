#!/usr/bin/env bash
# afterany of the SCF array lets failed upstream attempts produce a clear report.
# Both expensive arrays MUST use afterok of this gate job.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${CAMPAIGN_SNAPSHOT:?}" "${QE_RUN_ROOT:?}"
cd "$CAMPAIGN_SNAPSHOT"
exec /home/dawson666/miniconda3/bin/python3 -B scripts/run_nano_gated_20261015.py gate \
 --campaign "$CAMPAIGN_SNAPSHOT" --root "$QE_RUN_ROOT" \
 --release "$QE_RUN_ROOT/gate_release.json"
