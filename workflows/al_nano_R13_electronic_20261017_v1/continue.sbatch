#!/bin/bash
#SBATCH --job-name=al-of-R13-TNfinish
#SBATCH --account=mst114175
#SBATCH --partition=ct56
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --output=/work/dawson666/dftpy_project/relax/AL_NANO_FIVE_RADII_CORRECTED_20261015/R13_0V/electronic-finish-%j.out
#SBATCH --error=/work/dawson666/dftpy_project/relax/AL_NANO_FIVE_RADII_CORRECTED_20261015/R13_0V/electronic-finish-%j.err
# Submit only with --dependency=afterok:<unique diagnostic job id>.
# The Python entry also validates gate contents and every accepted SCF artifact.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
snapshot=/work/dawson666/src/campaign_snapshots/al_nano_R13_electronic_20261017_v1
parent_snapshot=/work/dawson666/src/campaign_snapshots/al_nano_five_radii_of_20261015_20261015T124450Z
runroot=/work/dawson666/dftpy_project/relax/AL_NANO_FIVE_RADII_CORRECTED_20261015
cd "$snapshot"
exec /home/dawson666/miniconda3/envs/dftpy-env/bin/python -B recover_electronic.py run \
  --parent-snapshot "$parent_snapshot" --parent-runroot "$runroot" \
  --parent-attempt "$runroot/R13_0V/attempt_002_smallstep" \
  --output-root "$runroot" --gate "$runroot/R13_0V/diagnostics_003_electronic/gate_pass.json" \
  --attempt attempt_003_TNsmallstep --max-runtime-s 79200
