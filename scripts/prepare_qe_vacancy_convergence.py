from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from ase.build import bulk
from ase.io import write
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BOHR_TO_ANG = 0.529177210903
RY_TO_EV = 13.605693122994
RY_PER_BOHR_TO_EV_PER_ANG = RY_TO_EV / BOHR_TO_ANG
DEFAULT_QE_A0_ANG = 4.039249525203


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


def _choose_vacancy_index(atoms) -> tuple[int, dict[str, float]]:
    pos = atoms.get_positions()
    cell = atoms.get_cell().array
    center = 0.5 * (cell[0] + cell[1] + cell[2])
    dists = np.linalg.norm(pos - center[None, :], axis=1)
    idx = int(np.argmin(dists))
    return idx, {
        "index": idx,
        "x_A": float(pos[idx, 0]),
        "y_A": float(pos[idx, 1]),
        "z_A": float(pos[idx, 2]),
        "distance_to_center_A": float(dists[idx]),
    }


def _remove_atom(atoms, atom_index: int):
    defect = atoms.copy()
    del defect[int(atom_index)]
    return defect


def _atoms_to_qe_card(atoms) -> tuple[str, str]:
    cell = atoms.get_cell().array
    scaled = atoms.get_scaled_positions(wrap=True)
    cell_lines = ["CELL_PARAMETERS angstrom"]
    for vec in cell:
        cell_lines.append(f"{vec[0]:18.10f} {vec[1]:18.10f} {vec[2]:18.10f}")
    pos_lines = ["ATOMIC_POSITIONS crystal"]
    for sym, pos in zip(atoms.get_chemical_symbols(), scaled):
        pos_lines.append(f"{sym:2s} {pos[0]:18.10f} {pos[1]:18.10f} {pos[2]:18.10f}")
    return "\n".join(cell_lines), "\n".join(pos_lines)


def _write_job(path: Path, *, job_name: str, input_name: str, output_name: str, partition: str, ntasks: int, time_limit: str, mem: str) -> None:
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
mpirun -np ${{SLURM_NTASKS}} "$PWX" -in {input_name} > {output_name}
status=$?

echo "[INFO] End       : $(date)"
echo "[INFO] Exit code : $status"
exit $status
"""
    _write_text(path, text)


def _write_group_job(
    path: Path,
    *,
    job_name: str,
    partition: str,
    ntasks: int,
    time_limit: str,
    mem: str,
) -> None:
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

echo "============================================================"
echo "[RUN] pristine_scf"
(
  cd pristine_scf
  mkdir -p tmp
  mpirun -np ${{SLURM_NTASKS}} "$PWX" -in scf.in > scf.out
)

echo "============================================================"
echo "[RUN] vacancy_relax"
(
  cd vacancy_relax
  mkdir -p tmp
  mpirun -np ${{SLURM_NTASKS}} "$PWX" -in relax.in > relax.out
)

echo "[INFO] End       : $(date)"
"""
    _write_text(path, text)


def _write_input(
    path: Path,
    *,
    calculation: str,
    prefix: str,
    atoms,
    ecut_ev: float,
    kmesh: tuple[int, int, int],
    pseudo_rel: str,
    pseudo_name: str,
    force_conv_eva: float | None,
) -> None:
    ecut_ry = ecut_ev / RY_TO_EV
    ecutrho_ry = 8.0 * ecut_ry
    cell_card, pos_card = _atoms_to_qe_card(atoms)
    control_extra: list[str] = []
    if calculation == "relax":
        force_conv_ry_per_bohr = float(force_conv_eva) / RY_PER_BOHR_TO_EV_PER_ANG
        control_extra.append(f"    forc_conv_thr = {force_conv_ry_per_bohr:.10f}")

    control_lines = [
        "&CONTROL",
        f"    calculation = '{calculation}'",
        f"    prefix = '{prefix}'",
        f"    pseudo_dir = '{pseudo_rel}'",
        "    outdir = './tmp'",
        "    verbosity = 'high'",
        "    tprnfor = .true.",
        "    tstress = .true.",
        *control_extra,
        "/",
        "",
        "&SYSTEM",
        "    ibrav = 0",
        f"    nat = {len(atoms)}",
        "    ntyp = 1",
        f"    ecutwfc = {ecut_ry:.10f}",
        f"    ecutrho = {ecutrho_ry:.10f}",
        "    occupations = 'smearing'",
        "    smearing = 'mp'",
        "    degauss = 0.02",
        "    nosym = .true.",
        "    noinv = .true.",
        "/",
        "",
        "&ELECTRONS",
        "    conv_thr = 1.0d-8",
        "    electron_maxstep = 300",
        "    mixing_beta = 0.2",
        "/",
    ]
    if calculation == "relax":
        control_lines.extend(
            [
                "",
                "&IONS",
                "    ion_dynamics = 'bfgs'",
                "/",
            ]
        )

    control_lines.extend(
        [
            "",
            "ATOMIC_SPECIES",
            f"Al  26.9815385  {pseudo_name}",
            "",
            cell_card,
            "",
            pos_card,
            "",
            "K_POINTS automatic",
            f"{kmesh[0]} {kmesh[1]} {kmesh[2]}  0 0 0",
            "",
        ]
    )
    _write_text(path, "\n".join(control_lines))


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


def _write_readme(path: Path, *, supercell_n: int, pristine_atoms: int, ecut_series: list[float], fixed_kmesh: tuple[int, int, int], fixed_ecut: float, kmesh_series: list[tuple[int, int, int]], force_conv_eva: float) -> None:
    text = f"""# QE vacancy formation-energy convergence workflow

This workflow follows the professor's practical constraints:

- QE vacancy cell is capped well below 250 atoms
- k-point tests are kept small and runnable (`1x1x1`, `2x2x2`, `3x3x3`)
- large supercell-size continuation should move to DFTpy

Chosen QE defect cell:

- primitive fcc Al repeated `{supercell_n}x{supercell_n}x{supercell_n}`
- pristine atoms: `{pristine_atoms}`
- vacancy atoms: `{pristine_atoms - 1}`

Two convergence studies are prepared:

1. `ecut_scan`: fixed k-mesh `{_kmesh_tag(fixed_kmesh)}`, ecut series `{", ".join(f"{v:.0f}" for v in ecut_series)} eV`
2. `kmesh_scan`: fixed cutoff `{fixed_ecut:.0f} eV`, k-mesh series `{", ".join(_kmesh_tag(v) for v in kmesh_series)}`

Vacancy runs use `relax` with a force threshold of `{force_conv_eva:.3f} eV/A`.

After the jobs finish, run:

`python scripts/collect_qe_vacancy_convergence.py --rootdir {path.parent}`
"""
    _write_text(path, text)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare primitive-cell QE vacancy convergence studies.")
    ap.add_argument("--outdir", default="", help="Output directory. Defaults to results/qe_vacancy_convergence_<date>.")
    ap.add_argument(
        "--pseudo",
        default=str(ROOT / "results" / "prof_response_20260506" / "04_bulk_qe_raw" / "a0_scan" / "a0_4.0400" / "pseudo" / "Al_PAW_PBE.UPF"),
    )
    ap.add_argument("--a0", type=float, default=DEFAULT_QE_A0_ANG)
    ap.add_argument("--supercell-n", type=int, default=4, help="Primitive-cell repetition for the fixed QE vacancy cell.")
    ap.add_argument("--ecut-series", default="300,400,500,600")
    ap.add_argument("--fixed-kmesh", default="2x2x2")
    ap.add_argument("--fixed-ecut", type=float, default=600.0)
    ap.add_argument("--kmesh-series", default="1x1x1,2x2x2,3x3x3")
    ap.add_argument("--force-conv", type=float, default=0.02, help="Vacancy relaxation force threshold in eV/A.")
    ap.add_argument("--pristine-partition", default="ct56")
    ap.add_argument("--pristine-ntasks", type=int, default=8)
    ap.add_argument("--pristine-time-limit", default="04:00:00")
    ap.add_argument("--pristine-mem", default="24G")
    ap.add_argument("--vacancy-partition", default="ct56")
    ap.add_argument("--vacancy-ntasks", type=int, default=8)
    ap.add_argument("--vacancy-time-limit", default="1-00:00:00")
    ap.add_argument("--vacancy-mem", default="64G")
    ap.add_argument("--group-partition", default="ct56")
    ap.add_argument("--group-ntasks", type=int, default=8)
    ap.add_argument("--group-time-limit", default="1-04:00:00")
    ap.add_argument("--group-mem", default="64G")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve() if str(args.outdir).strip() else (ROOT / "results" / "qe_vacancy_convergence_20260506").resolve()
    pseudo = Path(args.pseudo).expanduser().resolve()
    if not pseudo.exists():
        raise FileNotFoundError(f"Pseudo file not found: {pseudo}")
    if int(args.supercell_n) <= 0:
        raise ValueError("--supercell-n must be positive.")

    ecut_series = _parse_float_list(args.ecut_series)
    fixed_kmesh = _parse_kmesh_list(args.fixed_kmesh)[0]
    kmesh_series = _parse_kmesh_list(args.kmesh_series)

    pristine = bulk("Al", "fcc", a=float(args.a0), cubic=False).repeat((int(args.supercell_n), int(args.supercell_n), int(args.supercell_n)))
    vacancy_index, vacancy_site = _choose_vacancy_index(pristine)
    vacancy = _remove_atom(pristine, vacancy_index)

    psp_dir = outdir / "psp"
    psp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pseudo, psp_dir / pseudo.name)

    write(str(outdir / "pristine_start.xyz"), pristine)
    write(str(outdir / "pristine_start.vasp"), pristine, direct=True, vasp5=True)
    write(str(outdir / "vacancy_start.xyz"), vacancy)
    write(str(outdir / "vacancy_start.vasp"), vacancy, direct=True, vasp5=True)
    manifest = {
        "a0_A": float(args.a0),
        "cell_basis": "primitive",
        "supercell_n": int(args.supercell_n),
        "pristine_n_atoms": int(len(pristine)),
        "vacancy_n_atoms": int(len(vacancy)),
        "vacancy_site": vacancy_site,
        "force_conv_eV_per_A": float(args.force_conv),
    }
    _write_text(outdir / "manifest.json", json.dumps(manifest, indent=2))

    for ecut_ev in ecut_series:
        setting_dir = outdir / "ecut_scan" / f"ecut_{int(round(ecut_ev)):04d}eV"
        pristine_dir = setting_dir / "pristine_scf"
        vacancy_dir = setting_dir / "vacancy_relax"
        _write_input(
            pristine_dir / "scf.in",
            calculation="scf",
            prefix=f"Al_pristine_prim{args.supercell_n}_ecut_{int(round(ecut_ev)):04d}",
            atoms=pristine,
            ecut_ev=ecut_ev,
            kmesh=fixed_kmesh,
            pseudo_rel="../../../psp",
            pseudo_name=pseudo.name,
            force_conv_eva=None,
        )
        _write_job(
            pristine_dir / "job.sh",
            job_name=f"QVPE{int(round(ecut_ev)):04d}",
            input_name="scf.in",
            output_name="scf.out",
            partition=str(args.pristine_partition),
            ntasks=int(args.pristine_ntasks),
            time_limit=str(args.pristine_time_limit),
            mem=str(args.pristine_mem),
        )

        _write_input(
            vacancy_dir / "relax.in",
            calculation="relax",
            prefix=f"Al_vac_prim{args.supercell_n}_ecut_{int(round(ecut_ev)):04d}",
            atoms=vacancy,
            ecut_ev=ecut_ev,
            kmesh=fixed_kmesh,
            pseudo_rel="../../../psp",
            pseudo_name=pseudo.name,
            force_conv_eva=float(args.force_conv),
        )
        _write_job(
            vacancy_dir / "job.sh",
            job_name=f"QVVE{int(round(ecut_ev)):04d}",
            input_name="relax.in",
            output_name="relax.out",
            partition=str(args.vacancy_partition),
            ntasks=int(args.vacancy_ntasks),
            time_limit=str(args.vacancy_time_limit),
            mem=str(args.vacancy_mem),
        )
        _write_group_job(
            setting_dir / "group_job.sh",
            job_name=f"QVG_E{int(round(ecut_ev)):04d}",
            partition=str(args.group_partition),
            ntasks=int(args.group_ntasks),
            time_limit=str(args.group_time_limit),
            mem=str(args.group_mem),
        )

    for kmesh in kmesh_series:
        setting_dir = outdir / "kmesh_scan" / f"k_{_kmesh_tag(kmesh)}"
        pristine_dir = setting_dir / "pristine_scf"
        vacancy_dir = setting_dir / "vacancy_relax"
        _write_input(
            pristine_dir / "scf.in",
            calculation="scf",
            prefix=f"Al_pristine_prim{args.supercell_n}_k_{_kmesh_tag(kmesh)}",
            atoms=pristine,
            ecut_ev=float(args.fixed_ecut),
            kmesh=kmesh,
            pseudo_rel="../../../psp",
            pseudo_name=pseudo.name,
            force_conv_eva=None,
        )
        _write_job(
            pristine_dir / "job.sh",
            job_name=f"QVPK{_kmesh_tag(kmesh).replace('x', '')}",
            input_name="scf.in",
            output_name="scf.out",
            partition=str(args.pristine_partition),
            ntasks=int(args.pristine_ntasks),
            time_limit=str(args.pristine_time_limit),
            mem=str(args.pristine_mem),
        )

        _write_input(
            vacancy_dir / "relax.in",
            calculation="relax",
            prefix=f"Al_vac_prim{args.supercell_n}_k_{_kmesh_tag(kmesh)}",
            atoms=vacancy,
            ecut_ev=float(args.fixed_ecut),
            kmesh=kmesh,
            pseudo_rel="../../../psp",
            pseudo_name=pseudo.name,
            force_conv_eva=float(args.force_conv),
        )
        _write_job(
            vacancy_dir / "job.sh",
            job_name=f"QVVK{_kmesh_tag(kmesh).replace('x', '')}",
            input_name="relax.in",
            output_name="relax.out",
            partition=str(args.vacancy_partition),
            ntasks=int(args.vacancy_ntasks),
            time_limit=str(args.vacancy_time_limit),
            mem=str(args.vacancy_mem),
        )
        _write_group_job(
            setting_dir / "group_job.sh",
            job_name=f"QVG_K{_kmesh_tag(kmesh).replace('x', '')}",
            partition=str(args.group_partition),
            ntasks=int(args.group_ntasks),
            time_limit=str(args.group_time_limit),
            mem=str(args.group_mem),
        )

    _write_submit_script(outdir / "submit_ecut_scan.sh", "ecut_scan")
    _write_submit_script(outdir / "submit_kmesh_scan.sh", "kmesh_scan")
    _write_group_submit_script(outdir / "submit_grouped_ecut_scan.sh", "ecut_scan")
    _write_group_submit_script(outdir / "submit_grouped_kmesh_scan.sh", "kmesh_scan")
    _write_readme(
        outdir / "README.md",
        supercell_n=int(args.supercell_n),
        pristine_atoms=int(len(pristine)),
        ecut_series=ecut_series,
        fixed_kmesh=fixed_kmesh,
        fixed_ecut=float(args.fixed_ecut),
        kmesh_series=kmesh_series,
        force_conv_eva=float(args.force_conv),
    )

    print(f"[qe-vacancy] Prepared workflow under: {outdir}")
    print(f"[qe-vacancy] Primitive supercell : {args.supercell_n}x{args.supercell_n}x{args.supercell_n}")
    print(f"[qe-vacancy] Pristine atoms      : {len(pristine)}")
    print(f"[qe-vacancy] Vacancy atoms       : {len(vacancy)}")


if __name__ == "__main__":
    main()
