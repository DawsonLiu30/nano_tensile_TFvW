#!/usr/bin/env bash
set -euo pipefail

# Upload local PROFESS 3.0 source to iservice and compile it there.
# This avoids the GLIBC/libgfortran mismatch of the local WSL-built binary.

LOCAL_SRC="${LOCAL_SRC:-/mnt/c/Users/dawso/OneDrive/桌面/PROFESS3.0}"
REMOTE_HOST="${REMOTE_HOST:-iservice}"
REMOTE_ROOT="${REMOTE_ROOT:-/gpfs-work/dawson666/profess3_build_20260605}"
ACCOUNT="${ACCOUNT:-MST114175}"

if [[ ! -f "${LOCAL_SRC}/Makefile" || ! -d "${LOCAL_SRC}/Source" ]]; then
  echo "[ERROR] PROFESS source not found at ${LOCAL_SRC}" >&2
  exit 1
fi

cat <<EOF
============================================================
Upload and compile PROFESS 3.0 on iservice
============================================================
[LOCAL SRC ] ${LOCAL_SRC}
[REMOTE    ] ${REMOTE_HOST}:${REMOTE_ROOT}
[ACCOUNT   ] ${ACCOUNT}

This compiles a new iservice-compatible PROFESS binary instead of using
the local WSL binary that requires newer GLIBC/GFORTRAN versions.
EOF

ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_ROOT}'"
rsync -avhP --delete "${LOCAL_SRC}/" "${REMOTE_HOST}:${REMOTE_ROOT}/src/"

ssh "${REMOTE_HOST}" "cat > '${REMOTE_ROOT}/compile_profess3_ct56.sbatch' <<'SBATCH'
#!/usr/bin/env bash
#SBATCH -J PROFbuild
#SBATCH -A ${ACCOUNT}
#SBATCH -p ct56
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o compile_%j.out
#SBATCH -e compile_%j.err

set -euo pipefail

ROOT='${REMOTE_ROOT}'
SRC=\"\${ROOT}/src\"
cd \"\${SRC}\"

echo \"===== host =====\"
hostname
date

echo \"===== compiler and system =====\"
which gfortran || true
gfortran --version || true
ldd --version | head -5 || true

echo \"===== available FFTW/LAPACK hints =====\"
ldconfig -p 2>/dev/null | grep -E 'fftw3|lapack|blas|openblas' || true
find /usr /opt /pkg /apps -name 'fftw3.f03' -o -name 'fftw3.h' 2>/dev/null | head -40 || true

echo \"===== clean =====\"
make clean || true

echo \"===== compile attempt: system FFTW/LAPACK/BLAS, no libxc =====\"
make \\
  FC=gfortran \\
  FFLAGS='-O2 -cpp -I/usr/include' \\
  LDLIBS='-llapack -lblas -lfftw3 -lm' \\
  2>&1 | tee \"\${ROOT}/build_system_libs.log\"

echo \"===== built binary =====\"
file PROFESS
ldd PROFESS || true

mkdir -p \"\${ROOT}/bin\"
cp -f PROFESS \"\${ROOT}/bin/PROFESS\"
chmod +x \"\${ROOT}/bin/PROFESS\"

echo \"===== done =====\"
echo \"Binary: \${ROOT}/bin/PROFESS\"
SBATCH
"

ssh "${REMOTE_HOST}" "cd '${REMOTE_ROOT}' && sbatch compile_profess3_ct56.sbatch"

cat <<EOF

============================================================
Submitted compile job
============================================================
Monitor:
  squeue -u dawson666

After it finishes:
  ssh ${REMOTE_HOST} "cd ${REMOTE_ROOT} && ls -lh && tail -80 compile_*.out && tail -80 compile_*.err && ldd bin/PROFESS"

If this build succeeds, we can use:
  ${REMOTE_ROOT}/bin/PROFESS

for iservice Slurm PROFESS production.
EOF
