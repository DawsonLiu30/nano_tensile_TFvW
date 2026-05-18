from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BOHR_TO_ANG = 0.529177210903
RY_TO_EV = 13.605693122994
DEFAULT_A0_POINTS = "4.0000,4.0100,4.0200,4.0300,4.0400,4.0500,4.0600,4.0700,4.0800"


def _parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for chunk in str(text).split(","):
        token = chunk.strip()
        if token:
            values.append(float(token))
    if not values:
        raise ValueError("No valid floating-point values were provided.")
    return values


def _parse_kmesh_list(text: str) -> list[tuple[int, int, int]]:
    meshes: list[tuple[int, int, int]] = []
    for chunk in str(text).split(","):
        token = chunk.strip().lower()
        if not token:
            continue
        parts = token.split("x")
        if len(parts) != 3:
            raise ValueError(f"Invalid k-mesh token: {token}")
        meshes.append(tuple(int(v) for v in parts))
    if not meshes:
        raise ValueError("No valid k-point meshes were provided.")
    return meshes


def _kmesh_tag(mesh: tuple[int, int, int]) -> str:
    return f"{mesh[0]:02d}x{mesh[1]:02d}x{mesh[2]:02d}"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_job(path: Path, *, job_name: str, input_name: str, partition: str, ntasks: int, time_limit: str, mem: str) -> None:
    text = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --account=MST114175
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -euo pipefail

module purge
module load intel/2021 intelmpi/2021.11

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PWX="${{PWX:-/gpfs-home/dawson666/q-e-qe-7.3.1/bin/pw.x}}"

echo "[INFO] Host      : $(hostname)"
echo "[INFO] Workdir   : $(pwd)"
echo "[INFO] Job ID    : $SLURM_JOB_ID"
echo "[INFO] Job Name  : $SLURM_JOB_NAME"
echo "[INFO] Start     : $(date)"
echo "[INFO] PWX       : $PWX"

if [ ! -x "$PWX" ]; then
    echo "[ERROR] pw.x not found or not executable: $PWX"
    exit 1
fi

mkdir -p tmp
mpirun -np ${{SLURM_NTASKS}} "$PWX" -in {input_name} > scf.out
status=$?

echo "[INFO] End       : $(date)"
echo "[INFO] Exit code : $status"
exit $status
"""
    _write_text(path, text)


def _write_group_job(path: Path, *, job_name: str, partition: str, ntasks: int, time_limit: str, mem: str) -> None:
    text = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --account=MST114175
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -euo pipefail

module purge
module load intel/2021 intelmpi/2021.11

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PWX="${{PWX:-/gpfs-home/dawson666/q-e-qe-7.3.1/bin/pw.x}}"

echo "[INFO] Host      : $(hostname)"
echo "[INFO] Workdir   : $(pwd)"
echo "[INFO] Job ID    : $SLURM_JOB_ID"
echo "[INFO] Job Name  : $SLURM_JOB_NAME"
echo "[INFO] Start     : $(date)"
echo "[INFO] PWX       : $PWX"

if [ ! -x "$PWX" ]; then
    echo "[ERROR] pw.x not found or not executable: $PWX"
    exit 1
fi

for point in a0_*; do
  if [ ! -d "$point" ]; then
    continue
  fi
  echo "============================================================"
  echo "[RUN] $point"
  (
    cd "$point"
    mkdir -p tmp
    mpirun -np ${{SLURM_NTASKS}} "$PWX" -in scf.in > scf.out
  )
done

echo "[INFO] End       : $(date)"
"""
    _write_text(path, text)


def _write_scf_input(
    path: Path,
    *,
    prefix: str,
    a0_ang: float,
    ecut_ev: float,
    kmesh: tuple[int, int, int],
    pseudo_rel: str,
    pseudo_name: str,
) -> None:
    alat_bohr = a0_ang / BOHR_TO_ANG
    ecut_ry = ecut_ev / RY_TO_EV
    ecutrho_ry = 8.0 * ecut_ry
    text = f"""&CONTROL
    calculation = 'scf'
    prefix = '{prefix}'
    pseudo_dir = '{pseudo_rel}'
    outdir = './tmp'
    verbosity = 'high'
    tprnfor = .true.
    tstress = .true.
/

&SYSTEM
    ibrav = 2
    celldm(1) = {alat_bohr:.10f}
    nat = 1
    ntyp = 1
    ecutwfc = {ecut_ry:.10f}
    ecutrho = {ecutrho_ry:.10f}
    occupations = 'smearing'
    smearing = 'mp'
    degauss = 0.02
/

&ELECTRONS
    conv_thr = 1.0d-10
    electron_maxstep = 300
    mixing_beta = 0.3
/

ATOMIC_SPECIES
Al  26.9815385  {pseudo_name}

ATOMIC_POSITIONS crystal
Al  0.0000000000  0.0000000000  0.0000000000

K_POINTS automatic
{kmesh[0]} {kmesh[1]} {kmesh[2]}  0 0 0
"""
    _write_text(path, text)


def _write_submit_script(path: Path, search_root: str) -> None:
    text = f"""#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
find "{search_root}" -name job.sh | sort | while read -r job; do
  echo "[SUBMIT] $job"
  (cd "$(dirname "$job")" && sbatch job.sh)
done
"""
    _write_text(path, text)


def _write_group_submit_script(path: Path, search_root: str) -> None:
    text = f"""#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
find "{search_root}" -mindepth 1 -maxdepth 1 -type d | sort | while read -r setting; do
  echo "[SUBMIT-GROUP] $setting/group_job.sh"
  (cd "$setting" && sbatch group_job.sh)
done
"""
    _write_text(path, text)


def _write_readme(path: Path, *, fixed_kmesh: tuple[int, int, int], fixed_ecut_ev: float, a0_points: list[float], ecut_series: list[float], kmesh_series: list[tuple[int, int, int]]) -> None:
    ecut_labels = ", ".join(f"{v:.0f} eV" for v in ecut_series)
    kmesh_labels = ", ".join(_kmesh_tag(v) for v in kmesh_series)
    a0_labels = ", ".join(f"{v:.4f}" for v in a0_points)
    text = f"""# QE bulk B0 convergence workflow

This folder contains primitive-fcc bulk Al EOS scans for two numerical convergence studies:

1. `ecut_scan`: vary plane-wave cutoff at fixed k-mesh `{_kmesh_tag(fixed_kmesh)}`
2. `kmesh_scan`: vary k-mesh at fixed cutoff `{fixed_ecut_ev:.0f} eV`

Each setting contains the same `a0` scan:

{a0_labels}

Ecut series:

{ecut_labels}

K-mesh series:

{kmesh_labels}

After the jobs finish, run:

`python scripts/collect_qe_bulk_b_convergence.py --rootdir {path.parent}`
"""
    _write_text(path, text)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare primitive-fcc QE bulk EOS convergence studies for B0.")
    ap.add_argument("--outdir", default="", help="Output directory. Defaults to results/qe_bulk_b_convergence_<date>.")
    ap.add_argument(
        "--pseudo",
        default=str(ROOT / "results" / "prof_response_20260506" / "04_bulk_qe_raw" / "a0_scan" / "a0_4.0400" / "pseudo" / "Al_PAW_PBE.UPF"),
    )
    ap.add_argument("--a0-points", default=DEFAULT_A0_POINTS)
    ap.add_argument("--ecut-series", default="300,400,500,600,800,1000")
    ap.add_argument("--kmesh-series", default="8x8x8,10x10x10,12x12x12,14x14x14,16x16x16")
    ap.add_argument("--fixed-kmesh", default="16x16x16")
    ap.add_argument("--fixed-ecut", type=float, default=1000.0)
    ap.add_argument("--partition", default="ctest")
    ap.add_argument("--ntasks", type=int, default=8)
    ap.add_argument("--time-limit", default="00:30:00")
    ap.add_argument("--mem", default="8G")
    ap.add_argument("--group-partition", default="ct56")
    ap.add_argument("--group-time-limit", default="06:00:00")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve() if str(args.outdir).strip() else (ROOT / "results" / "qe_bulk_b_convergence_20260506").resolve()
    pseudo = Path(args.pseudo).expanduser().resolve()
    if not pseudo.exists():
        raise FileNotFoundError(f"Pseudo file not found: {pseudo}")

    a0_points = _parse_float_list(args.a0_points)
    ecut_series = _parse_float_list(args.ecut_series)
    fixed_kmesh = _parse_kmesh_list(args.fixed_kmesh)[0]
    kmesh_series = _parse_kmesh_list(args.kmesh_series)

    psp_dir = outdir / "psp"
    psp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pseudo, psp_dir / pseudo.name)

    for ecut_ev in ecut_series:
        setting_dir = outdir / "ecut_scan" / f"ecut_{int(round(ecut_ev)):04d}eV"
        for a0_ang in a0_points:
            leaf = setting_dir / f"a0_{a0_ang:.4f}"
            prefix = f"Al_bulk_ecut_{int(round(ecut_ev)):04d}_a0_{a0_ang:.4f}".replace(".", "p")
            _write_scf_input(
                leaf / "scf.in",
                prefix=prefix,
                a0_ang=a0_ang,
                ecut_ev=ecut_ev,
                kmesh=fixed_kmesh,
                pseudo_rel="../../../psp",
                pseudo_name=pseudo.name,
            )
            _write_job(
                leaf / "job.sh",
                job_name=f"QB_E{int(round(ecut_ev)):04d}_{str(a0_ang).replace('.', 'p')}",
                input_name="scf.in",
                partition=args.partition,
                ntasks=int(args.ntasks),
                time_limit=str(args.time_limit),
                mem=str(args.mem),
            )
        _write_group_job(
            setting_dir / "group_job.sh",
            job_name=f"QBG_E{int(round(ecut_ev)):04d}",
            partition=str(args.group_partition),
            ntasks=int(args.ntasks),
            time_limit=str(args.group_time_limit),
            mem=str(args.mem),
        )

    for kmesh in kmesh_series:
        setting_dir = outdir / "kmesh_scan" / f"k_{_kmesh_tag(kmesh)}"
        for a0_ang in a0_points:
            leaf = setting_dir / f"a0_{a0_ang:.4f}"
            prefix = f"Al_bulk_k_{_kmesh_tag(kmesh)}_a0_{a0_ang:.4f}".replace(".", "p")
            _write_scf_input(
                leaf / "scf.in",
                prefix=prefix,
                a0_ang=a0_ang,
                ecut_ev=float(args.fixed_ecut),
                kmesh=kmesh,
                pseudo_rel="../../../psp",
                pseudo_name=pseudo.name,
            )
            _write_job(
                leaf / "job.sh",
                job_name=f"QB_K{_kmesh_tag(kmesh).replace('x', '')}_{str(a0_ang).replace('.', 'p')}",
                input_name="scf.in",
                partition=args.partition,
                ntasks=int(args.ntasks),
                time_limit=str(args.time_limit),
                mem=str(args.mem),
            )
        _write_group_job(
            setting_dir / "group_job.sh",
            job_name=f"QBG_K{_kmesh_tag(kmesh).replace('x', '')}",
            partition=str(args.group_partition),
            ntasks=int(args.ntasks),
            time_limit=str(args.group_time_limit),
            mem=str(args.mem),
        )

    _write_submit_script(outdir / "submit_ecut_scan.sh", "ecut_scan")
    _write_submit_script(outdir / "submit_kmesh_scan.sh", "kmesh_scan")
    _write_group_submit_script(outdir / "submit_grouped_ecut_scan.sh", "ecut_scan")
    _write_group_submit_script(outdir / "submit_grouped_kmesh_scan.sh", "kmesh_scan")
    _write_readme(
        outdir / "README.md",
        fixed_kmesh=fixed_kmesh,
        fixed_ecut_ev=float(args.fixed_ecut),
        a0_points=a0_points,
        ecut_series=ecut_series,
        kmesh_series=kmesh_series,
    )

    print(f"[qe-bulk-B] Prepared workflow under: {outdir}")
    print(f"[qe-bulk-B] Ecut scan points : {', '.join(f'{v:.0f}' for v in ecut_series)} eV")
    print(f"[qe-bulk-B] K-mesh scan      : {', '.join(_kmesh_tag(v) for v in kmesh_series)}")


if __name__ == "__main__":
    main()
