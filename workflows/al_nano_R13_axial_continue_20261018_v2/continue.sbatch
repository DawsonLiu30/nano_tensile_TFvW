#!/bin/bash
#SBATCH --job-name=al-of-R13-axC
#SBATCH --account=mst114175
#SBATCH --partition=ct56
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/work/dawson666/dftpy_project/relax/AL_NANO_FIVE_RADII_CORRECTED_20261015/R13_0V/axial-continue-%j.out
#SBATCH --error=/work/dawson666/dftpy_project/relax/AL_NANO_FIVE_RADII_CORRECTED_20261015/R13_0V/axial-continue-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
snapshot=/work/dawson666/src/campaign_snapshots/al_nano_R13_axial_continue_20261018_v2
original=/work/dawson666/src/campaign_snapshots/al_nano_five_radii_of_20261015_20261015T124450Z
electronic=/work/dawson666/src/campaign_snapshots/al_nano_R13_electronic_20261017_v1
axial=/work/dawson666/src/campaign_snapshots/al_nano_R13_axial_linesearch_20261018_v1
runroot=/work/dawson666/dftpy_project/relax/AL_NANO_FIVE_RADII_CORRECTED_20261015
cd "$snapshot"
exec /home/dawson666/miniconda3/envs/dftpy-env/bin/python -B recover_axial.py execute \
  --original-snapshot "$original" --electronic-snapshot "$electronic" --axial-snapshot "$axial" \
  --runroot "$runroot" --parent-attempt "$runroot/R13_0V/attempt_004_axial_linesearch" \
  --attempt attempt_005_axial_linesearch_continue --max-runtime-s 21000
