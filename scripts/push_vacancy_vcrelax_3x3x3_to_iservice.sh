#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-/mnt/c/Users/dawso/nano_tensile_TFvW}"
DFTPY_REMOTE="${DFTPY_REMOTE:-iservice:/gpfs-work/dawson666/dftpy_project/relax/dftpy45}"
QE_REMOTE="${QE_REMOTE:-iservice:/gpfs-work/dawson666/qe_cases/qe_runs}"

echo "============================================================"
echo "Push vc-relax 3x3x3 vacancy workflow"
echo "============================================================"
echo "[LOCAL ] $LOCAL_ROOT"
echo "[DFTpy ] $DFTPY_REMOTE"
echo "[QE    ] $QE_REMOTE"
echo

rsync -avhP "$LOCAL_ROOT/app/dft_engine.py" "$DFTPY_REMOTE/app/dft_engine.py"

rsync -avhP \
  "$LOCAL_ROOT/scripts/prepare_dftpy_vacancy_conventional.py" \
  "$LOCAL_ROOT/scripts/run_dftpy_vcrelax_vacancy_one.py" \
  "$LOCAL_ROOT/scripts/collect_dftpy_conventional_vacancy.py" \
  "$LOCAL_ROOT/scripts/collect_dftpy_vcrelax_fmax.py" \
  "$DFTPY_REMOTE/scripts/"

rsync -avhP \
  "$LOCAL_ROOT/run_dftpy_vcrelax_vacancy_one_ct56.sbatch" \
  "$LOCAL_ROOT/submit_dftpy_vcrelax_conv3x3x3_ct56_array.sh" \
  "$DFTPY_REMOTE/"

rsync -avhP \
  "$LOCAL_ROOT/scripts/prepare_qe_vacancy_vcrelax_3x3x3.py" \
  "$LOCAL_ROOT/scripts/collect_qe_vcrelax_vacancy.py" \
  "$QE_REMOTE/scripts/"

echo
echo "============================================================"
echo "Remote preparation commands"
echo "============================================================"
echo "cd /gpfs-work/dawson666/qe_cases/qe_runs && python scripts/prepare_qe_vacancy_vcrelax_3x3x3.py --outdir /gpfs-work/dawson666/qe_cases/qe_runs/qe_vacancy_vcrelax_conv3x3x3_20260528 --pseudo /gpfs-work/dawson666/qe_cases/qe_runs/qe_vacancy_conventional_2x2x4_20260522/psp/Al_PAW_PBE.UPF --repeat 3x3x3 --ecut-series 400,600,800 --fixed-kmesh 3x3x3 --fixed-ecut 800 --kmesh-series 2x2x2,3x3x3,4x4x4,5x5x5 --force-conv 0.002 --press-conv-kbar 0.5 --time-limit 4-00:00:00 --mem 128G --max-parallel 4"
echo "cd /gpfs-work/dawson666/qe_cases/qe_runs/qe_vacancy_vcrelax_conv3x3x3_20260528 && sbatch submit_vcrelax_ct56_array.sh"
echo
echo "cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45 && python scripts/prepare_dftpy_vacancy_conventional.py --outdir results/dftpy_vacancy_vcrelax_conv3x3x3_qe_a0_20260528 --a0 4.039848 --spacing-repeat 3x3x3 --spacing-list 0.30,0.25,0.22,0.20,0.18,0.16 --fmax 0.002 --relax-steps 1500"
echo "cd /gpfs-work/dawson666/dftpy_project/relax/dftpy45 && sbatch submit_dftpy_vcrelax_conv3x3x3_ct56_array.sh"
