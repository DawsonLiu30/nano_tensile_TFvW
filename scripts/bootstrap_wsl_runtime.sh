#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${WSL_DISTRO_NAME:-}" ]]; then
  echo "[ERROR] Run this script inside WSL2." >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source_repo="$(cd "$script_dir/.." && pwd -P)"
runtime="${AL_DEFECTS_RUNTIME:-/var/tmp/al-defects-runtime-20260907}"
runtime="$(realpath -m "$runtime")"
env_prefix="${AL_DEFECTS_ENV_PREFIX:-$runtime/env}"
mamba_root="$runtime/mamba-root"
micromamba="$runtime/bin/micromamba"
environment_file="$source_repo/environment-wsl.yml"

case "$runtime" in
  /|/var|/var/tmp)
    echo "[ERROR] Refusing unsafe runtime path: $runtime" >&2
    exit 3
    ;;
esac

mkdir -p "$runtime/bin" "$mamba_root"
exec 9>"$runtime/.source.lock"
flock -n 9 || { echo '[ERROR] A calculation or source sync is active; retry after it finishes.' >&2; exit 4; }

if [[ ! -x "$micromamba" ]]; then
  archive="$(mktemp "$runtime/micromamba.XXXXXX.tar.bz2")"
  trap 'rm -f "$archive"' EXIT
  curl -L --fail --retry 10 --retry-all-errors --retry-delay 2 \
    -o "$archive" \
    https://micro.mamba.pm/api/micromamba/linux-64/latest
  tar -xjf "$archive" -C "$runtime" bin/micromamba
  rm -f "$archive"
  trap - EXIT
fi

export MAMBA_ROOT_PREFIX="$mamba_root"

if [[ ! -x "$env_prefix/bin/python" || ! -x "$env_prefix/bin/pw.x" ]]; then
  "$micromamba" create \
    --yes \
    --strict-channel-priority \
    --prefix "$env_prefix" \
    --file "$environment_file"
fi

"$env_prefix/bin/python" "$source_repo/scripts/sync_wsl_source.py" --source "$source_repo" --runtime "$runtime"

cat <<EOF
[READY]
runtime=$runtime
environment=$env_prefix
repository=$runtime/repo
python=$("$env_prefix/bin/python" --version 2>&1)
EOF
