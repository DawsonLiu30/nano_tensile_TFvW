#!/usr/bin/env bash
set -euo pipefail
usage() {
  cat <<'EOF'
Usage: run_local_dftpy_divacancy_wsl.sh --prepare-only | --status | --validate | --run
Prepare creates a new fixed-[110] run; --run explicitly starts expensive calculations.
RUN_ROOT must be beneath AL_DEFECTS_RUNTIME/runs. Existing attempts are archived.
Prepare overrides: KEDF_X=0.9 KEDF_Y=0.1 A0_START=3.9545804060131293
FMAX=0.005 RELAX_STEPS=1000 ASE_OPTIMIZER=BFGS DIRECTION=1,1,0 MAX_PAIRS=3
Execution: MAX_PARALLEL=1 FALLBACK_OPTIMIZER=LBFGS (bounded independent cases).
On resume, manifests define the protocol; conflicting environment values are rejected.
EOF
}
[[ $# == 1 ]] || { usage >&2; exit 2; }
mode="$1"
case "$mode" in
  --prepare-only|--run|--status|--validate) ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
runtime="$(realpath -m "${AL_DEFECTS_RUNTIME:-/var/tmp/al-defects-runtime-20260907}")"
repo_dir="${REPO_DIR:-$runtime/repo}"
python_bin="${AL_DEFECTS_ENV_PREFIX:-$runtime/env}/bin/python"
run_root="$(realpath -m "${RUN_ROOT:-$runtime/runs/divacancy_D110_prepare_20260907}")"
case "$run_root" in "$runtime"/runs/?*) ;; *) echo '[ERROR] Unsafe RUN_ROOT' >&2; exit 2;; esac
[[ -x "$python_bin" && -f "$repo_dir/al.lda.recpot" ]] || { echo '[ERROR] Missing environment or source' >&2; exit 1; }
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
max_parallel="${MAX_PARALLEL:-1}"
fallback="${FALLBACK_OPTIMIZER:-LBFGS}"
[[ "$max_parallel" =~ ^[1-9][0-9]*$ ]] || { echo '[ERROR] MAX_PARALLEL must be positive' >&2; exit 2; }
case "$fallback" in BFGS|LBFGS|BFGSLineSearch|SciPyFminBFGS|SciPyFminCG|MDMin) ;; *) echo '[ERROR] Invalid fallback optimizer' >&2; exit 2;; esac
exec 9>"$runtime/.source.lock"
flock -s -n 9 || { echo '[ERROR] Source sync is active' >&2; exit 3; }
if [[ "$mode" == --prepare-only ]]; then
  [[ ! -e "$run_root" ]] || { echo '[ERROR] RUN_ROOT exists; choose a new directory' >&2; exit 1; }
  "$python_bin" "$repo_dir/scripts/prepare_dftpy_divacancy_rscan_20260616.py" \
    --outdir "$run_root" --a0 "${A0_START:-3.9545804060131293}" --repeat 3x3x3 \
    --pp "$repo_dir/al.lda.recpot" --xc LDA --kedf TFVW \
    --kedf-x "${KEDF_X:-0.9}" --kedf-y "${KEDF_Y:-0.1}" --spacing 0.20 \
    --fmax "${FMAX:-0.005}" --relax-steps "${RELAX_STEPS:-1000}" \
    --ase-optimizer "${ASE_OPTIMIZER:-BFGS}" --pair-selection fixed_direction \
    --direction="${DIRECTION:-1,1,0}" --max-pairs "${MAX_PAIRS:-3}" \
    --partition local --time-limit 00:00:00 --cpus 1 --max-parallel "$max_parallel"
  echo "[PREPARED] $run_root"
  exit 0
fi
settings_text="$("$python_bin" "$repo_dir/scripts/divacancy_run_control.py" "$run_root")"
mapfile -t settings <<<"$settings_text"
if [[ "$mode" == --validate ]]; then
  printf '[VALID] %s cases; %s\n' "${#settings[@]}" "$run_root"
  exit 0
fi
qualify() { "$python_bin" "$repo_dir/scripts/divacancy_analysis_checks.py" "$run_root/pair_scan/$1"; }
if [[ "$mode" == --status ]]; then
  for setting in "${settings[@]}"; do printf '\n%s\n' "$setting"; qualify "$setting" || true; done
  exit 0
fi
exec 8>"$run_root/.runner.lock"
flock -n 8 || { echo '[ERROR] This run already has an active runner' >&2; exit 3; }
run_case() {
  local setting="$1" optimizer
  if qualify "$setting" >/dev/null; then echo "[SKIP] $setting: evidence qualified"; return 0; fi
  optimizer="${ASE_OPTIMIZER:-$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("ase_optimizer", "BFGS"))' "$run_root/pair_scan/$setting/point_manifest.json")}"
  for attempt in 1 2; do
    if (( attempt == 2 )); then
      [[ "$optimizer" != "$fallback" ]] || break
      optimizer="$fallback"
    fi
    echo "[RUN] $setting: $optimizer"
    if "$python_bin" "$repo_dir/scripts/run_dftpy_vcrelax_vacancy_one.py" \
        --rootdir "$run_root" --setting "$setting" --scan pair --restart \
        --ase-optimizer "$optimizer"; then
      if qualify "$setting"; then echo "[PASS] $setting"; return 0; fi
    fi
  done
  echo "[FAIL] $setting: no qualified attempt" >&2
  return 1
}
failures=0
pids=()
wait_group() {
  local pid
  for pid in "${pids[@]}"; do if ! wait "$pid"; then failures=$((failures + 1)); fi; done
  pids=()
}
for setting in "${settings[@]}"; do
  run_case "$setting" &
  pids+=("$!")
  if (( ${#pids[@]} >= max_parallel )); then wait_group; fi
done
wait_group
"$python_bin" "$repo_dir/scripts/collect_dftpy_conventional_vacancy.py" --rootdir "$run_root"
if (( failures )); then echo "[ERROR] $failures case(s) failed qualification" >&2; exit 1; fi
echo "[DONE] $run_root/analysis_current"
