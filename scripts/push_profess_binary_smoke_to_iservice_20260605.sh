#!/usr/bin/env bash
set -euo pipefail

# Smoke-test the local Linux PROFESS binary on iservice without compiling.
# This uploads one already-tested local PROFESS case, checks dynamic-library
# dependencies on iservice, and submits a tiny Slurm job.

LOCAL_ROOT="${LOCAL_ROOT:-/mnt/c/Users/dawso/Desktop/LOCAL_PROFESS_KEDF_SWEEP_20260604}"
REMOTE_HOST="${REMOTE_HOST:-iservice}"
REMOTE_ROOT="${REMOTE_ROOT:-/gpfs-work/dawson666/profess_binary_smoke_20260605}"
ACCOUNT="${ACCOUNT:-MST114175}"
CASE_REL="${CASE_REL:-HC_default_beta/pristine_vc_relaxed}"
CASE_NAME="$(basename "${CASE_REL}")"
CASE_PARENT="$(basename "$(dirname "${CASE_REL}")")"
REMOTE_CASE="${REMOTE_ROOT}/${CASE_PARENT}/${CASE_NAME}"

if [[ ! -x "${LOCAL_ROOT}/PROFESS" && ! -f "${LOCAL_ROOT}/PROFESS" ]]; then
  echo "[ERROR] PROFESS binary not found at ${LOCAL_ROOT}/PROFESS" >&2
  exit 1
fi

if [[ ! -d "${LOCAL_ROOT}/${CASE_REL}" ]]; then
  echo "[ERROR] Local test case not found: ${LOCAL_ROOT}/${CASE_REL}" >&2
  exit 1
fi

cat <<EOF
============================================================
PROFESS binary smoke test on iservice
============================================================
[LOCAL ] ${LOCAL_ROOT}
[CASE  ] ${CASE_REL}
[REMOTE] ${REMOTE_HOST}:${REMOTE_ROOT}

This does not compile PROFESS. It only tests whether the local Linux
binary can run on iservice with its current system libraries.
EOF

ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_CASE}' '${REMOTE_ROOT}/logs'"

rsync -avhP "${LOCAL_ROOT}/PROFESS" "${REMOTE_HOST}:${REMOTE_ROOT}/PROFESS"
rsync -avhP "${LOCAL_ROOT}/${CASE_REL}/" "${REMOTE_HOST}:${REMOTE_CASE}/"

ssh "${REMOTE_HOST}" "cd '${REMOTE_ROOT}' && chmod +x PROFESS && { echo '===== file PROFESS ====='; file PROFESS; echo; echo '===== ldd PROFESS ====='; ldd PROFESS || true; } | tee 00_binary_dependency_check.txt"

if ssh "${REMOTE_HOST}" "cd '${REMOTE_ROOT}' && grep -E \"not found|version .* not found\" 00_binary_dependency_check.txt >/dev/null"; then
  cat <<EOF

============================================================
Binary dependency check failed
============================================================
The uploaded local PROFESS binary is not compatible with the current
iservice runtime libraries. I will not submit the Slurm smoke job.

Check:
  ssh ${REMOTE_HOST} "cd ${REMOTE_ROOT} && cat 00_binary_dependency_check.txt"

Likely fixes:
  1. compile PROFESS on iservice, or
  2. run through a compatible Apptainer/Singularity container, or
  3. build a binary against an older glibc/libgfortran compatible with iservice.
EOF
  exit 2
fi

ssh "${REMOTE_HOST}" "cat > '${REMOTE_ROOT}/run_profess_smoke.sbatch' <<'SBATCH'
#!/usr/bin/env bash
#SBATCH -J PROFsmoke
#SBATCH -A ${ACCOUNT}
#SBATCH -p ct56
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o logs/PROFsmoke_%j.out
#SBATCH -e logs/PROFsmoke_%j.err

set -euo pipefail
ROOT='${REMOTE_ROOT}'
CASE='${REMOTE_CASE}'

cd \"\${CASE}\"
echo \"[INFO] Host: \$(hostname)\"
echo \"[INFO] CWD : \$(pwd)\"
echo \"[INFO] Running PROFESS smoke SCF\"

\"\${ROOT}/PROFESS\" '${CASE_NAME}' > '${CASE_NAME}.remote.stdout' 2> '${CASE_NAME}.remote.stderr'

echo \"[INFO] PROFESS exit code: \$?\"
echo \"[INFO] Recent output lines:\"
tail -60 '${CASE_NAME}.out' 2>/dev/null || true
tail -60 '${CASE_NAME}.remote.stdout' 2>/dev/null || true
SBATCH
"

ssh "${REMOTE_HOST}" "cd '${REMOTE_ROOT}' && sbatch run_profess_smoke.sbatch"

cat <<EOF

============================================================
Submitted smoke test
============================================================
Monitor:
  squeue -u dawson666

After it finishes, inspect:
  ssh ${REMOTE_HOST} "cd ${REMOTE_ROOT} && cat 00_binary_dependency_check.txt && ls -lh logs && find . -type f -name '*remote*' -o -name '*.out' | sort | tail -40"

If ldd shows "not found", direct upload is not enough and we either need
module loading, bundled libraries, or an iservice-native compile.
EOF
