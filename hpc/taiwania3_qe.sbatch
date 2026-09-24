#!/usr/bin/env bash
# Submitted by taiwania3_qe.sh with explicit resources and absolute log paths.
#SBATCH --job-name=al-divac-qe
#SBATCH --nodes=1
#SBATCH --ntasks=29
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit via Slurm, not bash on a login node}"
: "${PROJECT_ROOT:?Missing checkout}"
: "${QE_BINARY:?Missing verified QE executable}"
: "${PW_COMMAND:?Missing verified launch command}"
[[ "${SLURM_NTASKS:-0}" == 29 && "${SLURM_CPUS_PER_TASK:-0}" == 1 ]] || {
  echo '[ERROR] This profile expects 29 ranks and one CPU per rank.' >&2; exit 2;
}
if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh
fi
# The site module reads HOSTNAME to choose the correct InfiniBand device.
# Reuse of login-node UCX settings would select the wrong device on cpn nodes.
export HOSTNAME="$(hostname)"
module purge
module load intel/2021
module load intelmpi/2021.11
export I_MPI_PMI_LIBRARY="${QE_PMI_LIBRARY:-/usr/lib64/libpmi2.so}"
module list 2>&1
printf '[COMPUTE NODE] host=%s UCX_NET_DEVICES=%s job=%s\n' "$HOSTNAME" "${UCX_NET_DEVICES:-unset}" "$SLURM_JOB_ID"
linkage="$(ldd "$QE_BINARY")"
printf '%s\n' "$linkage"
mpi_library="$(printf '%s\n' "$linkage" | awk '$1 ~ /^libmpi\.so/ {print $3}')"
[[ "$linkage" != *'not found'* && "$mpi_library" == /*/intel/* ]] || {
  echo '[ERROR] QE libraries or Intel MPI linkage are invalid on this node.' >&2; exit 2;
}
sha256sum -- "$QE_BINARY"
exec bash "$PROJECT_ROOT/hpc/qe_portable.sbatch"
