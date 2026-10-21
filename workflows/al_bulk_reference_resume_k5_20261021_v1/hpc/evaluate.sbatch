#!/usr/bin/env bash
#SBATCH --account=mst114175
#SBATCH --partition=ct56
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
set -euo pipefail
exec /home/dawson666/miniconda3/bin/python3 -B "$CAMPAIGN_SNAPSHOT/scripts/run_bulk_reference.py" evaluate --package "$CAMPAIGN_SNAPSHOT" --root "$QE_RUN_ROOT"
