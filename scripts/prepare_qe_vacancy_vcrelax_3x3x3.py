from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import write


ROOT = Path(__file__).resolve().parents[1]
RY_TO_EV = 13.605693122994
BOHR_TO_ANG = 0.529177210903
RY_PER_BOHR_TO_EV_PER_ANG = RY_TO_EV / BOHR_TO_ANG
DEFAULT_A0_A = 4.039848


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_kmesh_list(text: str) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for token in str(text).split(","):
        parts = token.strip().lower().split("x")
        if len(parts) != 3:
            raise ValueError(f"Invalid kmesh token: {token}")
        out.append(tuple(int(x) for x in parts))
    return out


def parse_repeat(text: str) -> tuple[int, int, int]:
    parts = str(text).strip().lower().replace(",", "x").split("x")
    if len(parts) != 3:
        raise ValueError(f"Invalid repeat: {text}")
    repeat = tuple(int(x) for x in parts)
    if any(x <= 0 for x in repeat):
        raise ValueError(f"Repeat values must be positive: {repeat}")
    return repeat


def ktag(kmesh: tuple[int, int, int]) -> str:
    return f"{kmesh[0]:02d}x{kmesh[1]:02d}x{kmesh[2]:02d}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def cell_summary(atoms) -> dict[str, object]:
    cell = atoms.get_cell()
    return {
        "lengths_A": [float(x) for x in cell.lengths()],
        "angles_deg": [float(x) for x in cell.angles()],
        "volume_A3": float(atoms.get_volume()),
    }


def center_candidate_site(atoms, atom_index: int):
    centered = atoms.copy()
    scaled = centered.get_scaled_positions(wrap=True)
    target = np.array([0.5, 0.5, 0.5])
    shift_scaled = target - scaled[int(atom_index)]
    shift_cart = shift_scaled @ centered.get_cell().array
    centered.translate(shift_cart)
    centered.wrap()
    return centered, shift_scaled, shift_cart


def choose_center_vacancy(atoms) -> tuple[int, dict[str, object]]:
    scaled = atoms.get_scaled_positions(wrap=True)
    target = np.array([0.5, 0.5, 0.5])
    diff = scaled - target
    diff -= np.round(diff)
    cart = diff @ atoms.get_cell().array
    distances = np.linalg.norm(cart, axis=1)
    index = int(np.argmin(distances))
    return index, {
        "removed_atom_index": index,
        "removed_atom_scaled": [float(x) for x in scaled[index]],
        "removed_atom_cart_A": [float(x) for x in atoms.positions[index]],
        "distance_to_cell_center_A": float(distances[index]),
    }


def atoms_cards(atoms) -> tuple[str, str]:
    cell_lines = ["CELL_PARAMETERS angstrom"]
    for vec in atoms.get_cell().array:
        cell_lines.append(f"{vec[0]:18.10f} {vec[1]:18.10f} {vec[2]:18.10f}")
    pos_lines = ["ATOMIC_POSITIONS crystal"]
    for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_scaled_positions(wrap=True)):
        pos_lines.append(f"{sym:2s} {pos[0]:18.10f} {pos[1]:18.10f} {pos[2]:18.10f}")
    return "\n".join(cell_lines), "\n".join(pos_lines)


def write_vcrelax_input(
    path: Path,
    *,
    prefix: str,
    atoms,
    ecut_ev: float,
    kmesh: tuple[int, int, int],
    pseudo_name: str,
    force_conv_eva: float,
    press_conv_kbar: float,
) -> None:
    ecut_ry = float(ecut_ev) / RY_TO_EV
    force_conv_ry_bohr = float(force_conv_eva) / RY_PER_BOHR_TO_EV_PER_ANG
    cell_card, pos_card = atoms_cards(atoms)
    text = f"""&CONTROL
    calculation = 'vc-relax'
    prefix = '{prefix}'
    pseudo_dir = '../../../psp'
    outdir = './tmp'
    verbosity = 'high'
    tprnfor = .true.
    tstress = .true.
    forc_conv_thr = {force_conv_ry_bohr:.10f}
/

&SYSTEM
    ibrav = 0
    nat = {len(atoms)}
    ntyp = 1
    ecutwfc = {ecut_ry:.10f}
    ecutrho = {8.0 * ecut_ry:.10f}
    occupations = 'smearing'
    smearing = 'mp'
    degauss = 0.02
    nosym = .true.
    noinv = .true.
/

&ELECTRONS
    conv_thr = 1.0d-8
    electron_maxstep = 300
    mixing_beta = 0.2
/

&IONS
    ion_dynamics = 'bfgs'
/

&CELL
    cell_dynamics = 'bfgs'
    press = 0.0
    press_conv_thr = {float(press_conv_kbar):.6f}
    cell_dofree = 'all'
/

ATOMIC_SPECIES
Al 26.9815385 {pseudo_name}

{cell_card}

{pos_card}

K_POINTS automatic
{kmesh[0]} {kmesh[1]} {kmesh[2]} 0 0 0
"""
    write_text(path, text)


def write_group_job(path: Path, *, job_name: str, partition: str, ntasks: int, time_limit: str, mem: str) -> None:
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

run_qe () {{
  local subdir="$1"
  echo "============================================================"
  echo "[RUN] $subdir"
  if [ -f "$subdir/vc-relax.out" ] && grep -q "JOB DONE" "$subdir/vc-relax.out"; then
    echo "[SKIP] $subdir already completed"
    return 0
  fi
  (
    cd "$subdir"
    rm -f CRASH
    rm -rf tmp
    mkdir -p tmp
    mpirun -np "${{SLURM_NTASKS}}" "$PWX" -in vc-relax.in > vc-relax.out
  )
}}

echo "[INFO] Host    : $(hostname)"
echo "[INFO] Workdir : $(pwd)"
echo "[INFO] Job ID  : $SLURM_JOB_ID"
echo "[INFO] Start   : $(date)"

run_qe pristine_vcrelax
run_qe vacancy_vcrelax

echo "[INFO] End     : $(date)"
"""
    write_text(path, text)


def write_array(path: Path, *, settings: list[str], max_parallel: int, partition: str, ntasks: int, time_limit: str, mem: str) -> None:
    settings_file = path.with_suffix(".settings")
    write_text(settings_file, "\n".join(settings) + "\n")
    text = f"""#!/bin/bash
#SBATCH --job-name=QEVCR3
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --account=MST114175
#SBATCH --array=0-{len(settings) - 1}%{max_parallel}
#SBATCH --output=logs_submit/%x_%A_%a.out
#SBATCH --error=logs_submit/%x_%A_%a.err

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs_submit

SETTING=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{settings_file.name}")
echo "[INFO] SETTING=$SETTING"
cd "$SETTING"
bash group_job.sh
"""
    write_text(path, text)


def prepare_case(
    root: Path,
    rel_setting: str,
    *,
    pristine,
    vacancy,
    ecut_ev: float,
    kmesh: tuple[int, int, int],
    pseudo_name: str,
    force_conv_eva: float,
    press_conv_kbar: float,
    partition: str,
    ntasks: int,
    time_limit: str,
    mem: str,
) -> None:
    setting_dir = root / rel_setting
    write_vcrelax_input(
        setting_dir / "pristine_vcrelax" / "vc-relax.in",
        prefix=f"Al_pristine_vcr3_{rel_setting.replace('/', '_')}",
        atoms=pristine,
        ecut_ev=ecut_ev,
        kmesh=kmesh,
        pseudo_name=pseudo_name,
        force_conv_eva=force_conv_eva,
        press_conv_kbar=press_conv_kbar,
    )
    write_vcrelax_input(
        setting_dir / "vacancy_vcrelax" / "vc-relax.in",
        prefix=f"Al_vac_vcr3_{rel_setting.replace('/', '_')}",
        atoms=vacancy,
        ecut_ev=ecut_ev,
        kmesh=kmesh,
        pseudo_name=pseudo_name,
        force_conv_eva=force_conv_eva,
        press_conv_kbar=press_conv_kbar,
    )
    write_group_job(
        setting_dir / "group_job.sh",
        job_name=f"QVCR{rel_setting.split('/')[-1].replace('_', '')[:8]}",
        partition=partition,
        ntasks=ntasks,
        time_limit=time_limit,
        mem=mem,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare QE vc-relax vacancy benchmark with conventional fcc 3x3x3 cell.")
    ap.add_argument("--outdir", default=str(ROOT / "results" / "qe_vacancy_vcrelax_conv3x3x3_20260528"))
    ap.add_argument("--pseudo", default=str(ROOT / "results" / "qe_vacancy_convergence_20260506" / "psp" / "Al_PAW_PBE.UPF"))
    ap.add_argument("--a0", type=float, default=DEFAULT_A0_A)
    ap.add_argument("--repeat", default="3x3x3")
    ap.add_argument("--ecut-series", default="400,600,800")
    ap.add_argument("--fixed-kmesh", default="3x3x3")
    ap.add_argument("--fixed-ecut", type=float, default=800.0)
    ap.add_argument("--kmesh-series", default="2x2x2,3x3x3,4x4x4,5x5x5")
    ap.add_argument("--force-conv", type=float, default=0.002)
    ap.add_argument("--press-conv-kbar", type=float, default=0.5)
    ap.add_argument("--partition", default="ct56")
    ap.add_argument("--ntasks", type=int, default=28)
    ap.add_argument("--time-limit", default="4-00:00:00")
    ap.add_argument("--mem", default="128G")
    ap.add_argument("--max-parallel", type=int, default=4)
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    pseudo = Path(args.pseudo).expanduser().resolve()
    if not pseudo.exists():
        raise FileNotFoundError(f"Missing QE pseudopotential: {pseudo}")

    repeat = parse_repeat(args.repeat)
    pristine = bulk("Al", "fcc", a=float(args.a0), cubic=True).repeat(repeat)
    vacancy_idx, pre_shift_vacancy_info = choose_center_vacancy(pristine)
    pristine, shift_scaled, shift_cart = center_candidate_site(pristine, vacancy_idx)
    vacancy_idx, vacancy_info = choose_center_vacancy(pristine)
    vacancy = pristine.copy()
    del vacancy[vacancy_idx]

    if len(pristine) > 250:
        raise ValueError(f"QE pristine atom count exceeds requested limit: {len(pristine)}")

    outdir.mkdir(parents=True, exist_ok=True)
    psp = outdir / "psp"
    psp.mkdir(exist_ok=True)
    shutil.copy2(pseudo, psp / pseudo.name)

    write(str(outdir / "pristine_start.vasp"), pristine, direct=True, vasp5=True)
    write(str(outdir / "vacancy_start.vasp"), vacancy, direct=True, vasp5=True)
    write(str(outdir / "pristine_start.xyz"), pristine)
    write(str(outdir / "vacancy_start.xyz"), vacancy)

    fixed_kmesh = parse_kmesh_list(args.fixed_kmesh)[0]
    kmesh_series = parse_kmesh_list(args.kmesh_series)
    ecut_series = parse_float_list(args.ecut_series)
    settings: list[str] = []

    for ecut_ev in ecut_series:
        rel = f"ecut_scan/ecut_{int(round(ecut_ev)):04d}eV_k{ktag(fixed_kmesh)}"
        settings.append(rel)
        prepare_case(
            outdir,
            rel,
            pristine=pristine,
            vacancy=vacancy,
            ecut_ev=float(ecut_ev),
            kmesh=fixed_kmesh,
            pseudo_name=pseudo.name,
            force_conv_eva=float(args.force_conv),
            press_conv_kbar=float(args.press_conv_kbar),
            partition=str(args.partition),
            ntasks=int(args.ntasks),
            time_limit=str(args.time_limit),
            mem=str(args.mem),
        )

    for kmesh in kmesh_series:
        rel = f"kmesh_scan/k_{ktag(kmesh)}_ecut_{int(round(float(args.fixed_ecut))):04d}eV"
        settings.append(rel)
        prepare_case(
            outdir,
            rel,
            pristine=pristine,
            vacancy=vacancy,
            ecut_ev=float(args.fixed_ecut),
            kmesh=kmesh,
            pseudo_name=pseudo.name,
            force_conv_eva=float(args.force_conv),
            press_conv_kbar=float(args.press_conv_kbar),
            partition=str(args.partition),
            ntasks=int(args.ntasks),
            time_limit=str(args.time_limit),
            mem=str(args.mem),
        )

    write_array(
        outdir / "submit_vcrelax_ct56_array.sh",
        settings=settings,
        max_parallel=int(args.max_parallel),
        partition=str(args.partition),
        ntasks=int(args.ntasks),
        time_limit=str(args.time_limit),
        mem=str(args.mem),
    )
    scripts_dir = outdir / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), scripts_dir / Path(__file__).name)
    collector = ROOT / "scripts" / "collect_qe_vcrelax_vacancy.py"
    if collector.exists():
        shutil.copy2(collector, scripts_dir / collector.name)

    manifest = {
        "workflow": "qe_vacancy_vcrelax_conv3x3x3",
        "status": "new professor-requested replacement for rejected fixed-cell/scf vacancy data",
        "calculation": "vc-relax for both pristine and vacancy",
        "cell_basis": "conventional cubic fcc",
        "repeat": list(repeat),
        "pristine_n_atoms": int(len(pristine)),
        "vacancy_n_atoms": int(len(vacancy)),
        "vacancy_concentration_percent": 100.0 / float(len(pristine)),
        "a0_start_A": float(args.a0),
        "pristine_cell_start": cell_summary(pristine),
        "vacancy_cell_start": cell_summary(vacancy),
        "vacancy_site": vacancy_info,
        "pre_shift_vacancy_site": pre_shift_vacancy_info,
        "origin_shift_scaled": [float(x) for x in shift_scaled],
        "origin_shift_cart_A": [float(x) for x in shift_cart],
        "vesta_note": "The conventional 3x3x3 origin is shifted so the selected vacancy site is exactly at the visual cell center before removal.",
        "force_conv_eV_A": float(args.force_conv),
        "press_conv_kbar": float(args.press_conv_kbar),
        "ecut_scan_eV": ecut_series,
        "fixed_kmesh_for_ecut": list(fixed_kmesh),
        "fixed_ecut_for_kmesh_eV": float(args.fixed_ecut),
        "kmesh_scan": [list(k) for k in kmesh_series],
        "formation_energy_formula": "E_f^vac = E_vc-relax_vac^(N-1) - ((N-1)/N) E_vc-relax_pristine^N",
    }
    write_text(outdir / "manifest.json", json.dumps(manifest, indent=2))
    write_text(
        outdir / "README.md",
        "# QE vacancy vc-relax conventional 3x3x3 benchmark\n\n"
        "This workflow replaces the earlier 2x2x4 fixed-cell SCF/relax vacancy data.\n"
        "Both pristine and vacancy calculations use `calculation = 'vc-relax'`.\n",
    )

    print("============================================================")
    print("QE vc-relax conventional 3x3x3 vacancy workflow prepared")
    print("============================================================")
    print(f"Output        : {outdir}")
    print(f"Atoms         : {len(pristine)} -> {len(vacancy)}")
    print(f"Vacancy conc. : {100.0 / float(len(pristine)):.6f} %")
    print(f"Cell lengths  : {cell_summary(pristine)['lengths_A']}")
    print(f"Submit        : cd {outdir} && sbatch submit_vcrelax_ct56_array.sh")


if __name__ == "__main__":
    main()
