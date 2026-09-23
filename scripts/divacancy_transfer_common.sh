#!/usr/bin/env bash
# Shared helpers for explicitly invoked remote transfer commands.
set -euo pipefail
LOCAL_ROOT="${LOCAL_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
DATA_ROOT="${AL_DEFECTS_DATA_ROOT:-$(dirname -- "$LOCAL_ROOT")/AL_DEFECTS_USB_HANDOFF_20260717}"
REMOTE_HOST="${REMOTE_HOST:-iservice}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${AL_DEFECTS_ENV_PREFIX:-/var/tmp/al-defects-runtime-20260907/env}/bin/python" ]]; then
    PYTHON="${AL_DEFECTS_ENV_PREFIX:-/var/tmp/al-defects-runtime-20260907/env}/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
  else
    PYTHON=python
  fi
fi

upload_new_divacancy_package() {
  local local_dir="$1" remote_dir="$2" remote_command
  [[ -d "$local_dir" && -f "$local_dir/manifest.json" ]] || { echo 'Missing prepared package' >&2; return 2; }
  # Pass the path as a shell argument; never interpolate it as remote source code.
  printf -v remote_command '%q ' bash -c \
    'set -euo pipefail; [[ "$1" = /* && "$1" != / ]] || exit 2; if [[ -e "$1" ]]; then echo "Refusing existing remote run: $1" >&2; exit 2; fi; mkdir -p -- "$1"' \
    _ "$remote_dir"
  ssh "$REMOTE_HOST" "$remote_command"
  rsync -avhP --protect-args "$local_dir/" "$REMOTE_HOST:$remote_dir/"
}

submit_divacancy_package() {
  local remote_dir="$1" submit_name="$2" settings_name="$3" max_parallel="$4" remote_command
  printf -v remote_command '%q ' bash -s -- "$remote_dir" "$submit_name" "$settings_name" "$max_parallel" "${ACCOUNT:-}"
  ssh "$REMOTE_HOST" "$remote_command" <<'REMOTE'
set -euo pipefail
root="$1"; submit="$2"; settings="$3"; parallel="$4"; account="$5"
[[ "$parallel" =~ ^[1-9][0-9]*$ ]] || exit 2
cd -- "$root"
count=$(awk 'NF { n++ } END { print n+0 }' "$settings")
[[ "$count" -gt 0 ]] || { echo 'Empty settings file' >&2; exit 2; }
mkdir -p logs_ctest logs_submit
echo "[SUBMIT] $count cases; array 0-$((count - 1))%$parallel"
submit_args=(--array="0-$((count - 1))%$parallel")
[[ -z "$account" ]] || submit_args+=(-A "$account")
ROOT="$root" SERIES_DIR="$root" sbatch "${submit_args[@]}" "$submit"
REMOTE
}
