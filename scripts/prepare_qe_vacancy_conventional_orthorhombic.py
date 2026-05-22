from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import write


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BOHR_TO_ANG = 0.529177210903
RY_TO_EV = 13.605693122994
RY_PER_BOHR_TO_EV_PER_ANG = RY_TO_EV / BOHR_TO_ANG
DEFAULT_QE_A0_ANG = 4.039848


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


def _parse_repeat(text: str) -> tuple[int, int, int]:
    parts = str(text).lower().replace(",", "x").split("x")
    if len(parts) != 3:
        raise ValueError(f"Invalid repeat value: {text}")
    repeat = tuple(int(v.strip()) for v in parts)
    if any(v <= 0 for v in repeat):
        raise ValueError(f"Repeat values must be positive: {text}")
    return repeat


def _kmesh_tag(mesh: tuple[int, int, int]) -> str:
    return f"{mesh[0]:02d}x{mesh[1]:02d}x{mesh[2]:02d}"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _cell_summary(atoms) -> dict[str, object]:
    cell = atoms.get_cell()
    lengths = cell.lengths()
    angles = cell.angles()
    return {
        "lengths_A": [float(v) for v in lengths],
        "angles_deg": [float(v) for v in angles],
        "volume_A3": float(atoms.get_volume()),
    }


def _choose_vacancy_index(atoms) -> tuple[int, dict[str, object]]:
    scaled = atoms.get_scaled_positions(wrap=True)
    target = np.array([0.5, 0.5, 0.5])
    diff_scaled = scaled - target
    diff_scaled -= np.round(diff_scaled)
    cart_diff = diff_scaled @ atoms.get_cell().array
    dists = np.linalg.norm(cart_diff, axis=1)
    idx = int(np.argmin(dists))
    pos = atoms.get_positions()[idx]
    return idx, {
        "index": idx,
        "scaled_position": [float(v) for v in scaled[idx]],
        "cartesian_position_A": [float(v) for v in pos],
        "distance_to_cell_center_A": float(dists[idx]),
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

    lines = [
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
        lines.extend(
            [
                "",
                "&IONS",
                "    ion_dynamics = 'bfgs'",
                "/",
            ]
        )
    lines.extend(
        [
            "",
            "ATOMIC_SPECIES",
            f"Al 26.9815385 {pseudo_name}",
            "",
            cell_card,
            "",
            pos_card,
            "",
            "K_POINTS automatic",
            f"{kmesh[0]} {kmesh[1]} {kmesh[2]} 0 0 0",
            "",
        ]
    )
    _write_text(path, "\n".join(lines))


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

run_qe () {{
    local subdir="$1"
    local input="$2"
    local output="$3"

    echo "============================================================"
    echo "[RUN] $subdir / $input"
    if [ -f "$subdir/$output" ] && grep -q "JOB DONE" "$subdir/$output"; then
        echo "[SKIP] $subdir already completed."
        return 0
    fi

    (
      cd "$subdir"
      rm -f CRASH
      rm -rf tmp
      mkdir -p tmp
      mpirun -np ${{SLURM_NTASKS}} "$PWX" -in "$input" > "$output"
    )
}}

run_qe pristine_scf scf.in scf.out
run_qe vacancy_relax relax.in relax.out

echo "[INFO] End       : $(date)"
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


def _write_array_submit_script(
    path: Path,
    *,
    job_name: str,
    partition: str,
    ntasks: int,
    time_limit: str,
    mem: str,
    max_parallel: int,
    settings: list[str],
) -> None:
    settings_file = path.with_suffix(".settings")
    _write_text(settings_file, "\n".join(settings) + "\n")
    last_index = max(len(settings) - 1, 0)
    text = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --account=MST114175
#SBATCH --array=0-{last_index}%{max_parallel}
#SBATCH --output=logs_submit/%x_%A_%a.out
#SBATCH --error=logs_submit/%x_%A_%a.err

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs_submit

SETTING=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{settings_file.name}")
if [ -z "$SETTING" ]; then
  echo "[ERROR] Empty setting for task $SLURM_ARRAY_TASK_ID"
  exit 1
fi

echo "[INFO] Host      : $(hostname)"
echo "[INFO] Workdir   : $(pwd)"
echo "[INFO] Job ID    : $SLURM_JOB_ID"
echo "[INFO] Array ID  : $SLURM_ARRAY_TASK_ID"
echo "[INFO] Setting   : $SETTING"
echo "[INFO] Start     : $(date)"

cd "$SETTING"
bash group_job.sh

echo "[INFO] End       : $(date)"
"""
    _write_text(path, text)


def _prepare_case(
    *,
    setting_dir: Path,
    pristine,
    vacancy,
    ecut_ev: float,
    kmesh: tuple[int, int, int],
    pseudo_name: str,
    pristine_prefix: str,
    vacancy_prefix: str,
    job_name: str,
    force_conv_eva: float,
    partition: str,
    ntasks: int,
    time_limit: str,
    mem: str,
) -> None:
    pristine_dir = setting_dir / "pristine_scf"
    vacancy_dir = setting_dir / "vacancy_relax"
    _write_input(
        pristine_dir / "scf.in",
        calculation="scf",
        prefix=pristine_prefix,
        atoms=pristine,
        ecut_ev=ecut_ev,
        kmesh=kmesh,
        pseudo_rel="../../../psp",
        pseudo_name=pseudo_name,
        force_conv_eva=None,
    )
    _write_input(
        vacancy_dir / "relax.in",
        calculation="relax",
        prefix=vacancy_prefix,
        atoms=vacancy,
        ecut_ev=ecut_ev,
        kmesh=kmesh,
        pseudo_rel="../../../psp",
        pseudo_name=pseudo_name,
        force_conv_eva=force_conv_eva,
    )
    _write_group_job(
        setting_dir / "group_job.sh",
        job_name=job_name,
        partition=partition,
        ntasks=ntasks,
        time_limit=time_limit,
        mem=mem,
    )


def _write_readme(
    path: Path,
    *,
    a0: float,
    repeat: tuple[int, int, int],
    pristine_atoms: int,
    vacancy_atoms: int,
    fixed_kmesh: tuple[int, int, int],
    ecut_series: list[float],
    dense_kmesh: tuple[int, int, int],
    dense_ecut_series: list[float],
    fixed_ecut: float,
    kmesh_series: list[tuple[int, int, int]],
    force_conv_eva: float,
) -> None:
    text = f"""# QE conventional 2x2x4 vacancy convergence workflow

This workflow replaces the visually misleading primitive rhombohedral 4x4x4
cell used in the earlier slide draft.

Chosen QE defect cell:

- conventional fcc Al cell repeated `{repeat[0]}x{repeat[1]}x{repeat[2]}`
- orthorhombic 90-degree VESTA-friendly cell
- pristine atoms: `{pristine_atoms}`
- vacancy atoms: `{vacancy_atoms}`
- lattice constant: `{a0:.6f} A`
- central vacancy site removed

This keeps the original 64/63 atom scale but avoids the slanted primitive-cell
visualization that was rejected in the meeting. It also stays below the
practical QE limit of 250 atoms.

Prepared scans:

1. `ecut_scan`: fixed k-mesh `{_kmesh_tag(fixed_kmesh)}`, ecut series `{", ".join(f"{v:.0f}" for v in ecut_series)} eV`
2. `dense_k05_ecut_scan`: fixed k-mesh `{_kmesh_tag(dense_kmesh)}`, ecut series `{", ".join(f"{v:.0f}" for v in dense_ecut_series)} eV`
3. `kmesh_scan`: fixed cutoff `{fixed_ecut:.0f} eV`, k-mesh series `{", ".join(_kmesh_tag(v) for v in kmesh_series)}`

Vacancy runs use `relax` with force threshold `{force_conv_eva:.4f} eV/A`.

Submit on iservice. The split scripts avoid flooding `ct56`:

```bash
sbatch submit_ctest_short_array.sh
sbatch submit_ct56_long_array.sh
```

Collect after completion:

```bash
python scripts/collect_all_qe_vacancy_recursive.py --rootdir .
```
"""
    _write_text(path, text)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare VESTA-friendly conventional QE vacancy convergence inputs.")
    ap.add_argument("--outdir", default=str(ROOT / "results" / "qe_vacancy_conventional_2x2x4_20260522"))
    ap.add_argument("--pseudo", default=str(ROOT / "results" / "qe_vacancy_convergence_20260506" / "psp" / "Al_PAW_PBE.UPF"))
    ap.add_argument("--a0", type=float, default=DEFAULT_QE_A0_ANG)
    ap.add_argument("--repeat", default="2x2x4", help="Conventional fcc repeat, default gives 64 pristine atoms.")
    ap.add_argument("--ecut-series", default="300,400,500,600,800")
    ap.add_argument("--fixed-kmesh", default="2x2x2")
    ap.add_argument("--dense-kmesh", default="5x5x5")
    ap.add_argument("--dense-ecut-series", default="400,500,600,800")
    ap.add_argument("--fixed-ecut", type=float, default=600.0)
    ap.add_argument("--kmesh-series", default="1x1x1,2x2x2,3x3x3,4x4x4,5x5x5,6x6x6")
    ap.add_argument("--force-conv", type=float, default=0.002, help="Vacancy relaxation force threshold in eV/A.")
    ap.add_argument("--partition", default="ct56")
    ap.add_argument("--ntasks", type=int, default=28)
    ap.add_argument("--time-limit", default="1-00:00:00")
    ap.add_argument("--mem", default="96G")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    pseudo = Path(args.pseudo).expanduser().resolve()
    if not pseudo.exists():
        raise FileNotFoundError(f"Pseudo file not found: {pseudo}")

    repeat = _parse_repeat(args.repeat)
    ecut_series = _parse_float_list(args.ecut_series)
    fixed_kmesh = _parse_kmesh_list(args.fixed_kmesh)[0]
    dense_kmesh = _parse_kmesh_list(args.dense_kmesh)[0]
    dense_ecut_series = _parse_float_list(args.dense_ecut_series)
    kmesh_series = _parse_kmesh_list(args.kmesh_series)

    pristine = bulk("Al", "fcc", a=float(args.a0), cubic=True).repeat(repeat)
    vacancy_index, vacancy_site = _choose_vacancy_index(pristine)
    vacancy = _remove_atom(pristine, vacancy_index)

    if len(pristine) > 250:
        raise ValueError(f"QE atom-count limit exceeded: pristine has {len(pristine)} atoms.")

    psp_dir = outdir / "psp"
    psp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pseudo, psp_dir / pseudo.name)

    write(str(outdir / "pristine_start.xyz"), pristine)
    write(str(outdir / "pristine_start.vasp"), pristine, direct=True, vasp5=True)
    write(str(outdir / "vacancy_start.xyz"), vacancy)
    write(str(outdir / "vacancy_start.vasp"), vacancy, direct=True, vasp5=True)

    manifest = {
        "a0_A": float(args.a0),
        "cell_basis": "conventional_fcc_orthorhombic",
        "conventional_repeat": list(repeat),
        "pristine_n_atoms": int(len(pristine)),
        "vacancy_n_atoms": int(len(vacancy)),
        "pristine_cell": _cell_summary(pristine),
        "vacancy_cell": _cell_summary(vacancy),
        "vacancy_site": vacancy_site,
        "force_conv_eV_per_A": float(args.force_conv),
        "ecut_scan_eV": ecut_series,
        "fixed_kmesh_for_ecut": list(fixed_kmesh),
        "dense_kmesh_for_cutoff_check": list(dense_kmesh),
        "dense_kmesh_ecut_scan_eV": dense_ecut_series,
        "fixed_ecut_for_kmesh_eV": float(args.fixed_ecut),
        "kmesh_scan": [list(v) for v in kmesh_series],
    }
    _write_text(outdir / "manifest.json", json.dumps(manifest, indent=2))

    for ecut_ev in ecut_series:
        setting_name = f"ecut_{int(round(ecut_ev)):04d}eV"
        _prepare_case(
            setting_dir=outdir / "ecut_scan" / setting_name,
            pristine=pristine,
            vacancy=vacancy,
            ecut_ev=ecut_ev,
            kmesh=fixed_kmesh,
            pseudo_name=pseudo.name,
            pristine_prefix=f"Al_pristine_conv2x2x4_ecut_{int(round(ecut_ev)):04d}",
            vacancy_prefix=f"Al_vac_conv2x2x4_ecut_{int(round(ecut_ev)):04d}",
            job_name=f"QVCE{int(round(ecut_ev)):04d}",
            force_conv_eva=float(args.force_conv),
            partition=str(args.partition),
            ntasks=int(args.ntasks),
            time_limit=str(args.time_limit),
            mem=str(args.mem),
        )

    for ecut_ev in dense_ecut_series:
        setting_name = f"ecut_{int(round(ecut_ev)):04d}eV_k05"
        _prepare_case(
            setting_dir=outdir / "dense_k05_ecut_scan" / setting_name,
            pristine=pristine,
            vacancy=vacancy,
            ecut_ev=ecut_ev,
            kmesh=dense_kmesh,
            pseudo_name=pseudo.name,
            pristine_prefix=f"Al_pristine_conv2x2x4_k05_ecut_{int(round(ecut_ev)):04d}",
            vacancy_prefix=f"Al_vac_conv2x2x4_k05_ecut_{int(round(ecut_ev)):04d}",
            job_name=f"QVCD{int(round(ecut_ev)):04d}",
            force_conv_eva=float(args.force_conv),
            partition=str(args.partition),
            ntasks=int(args.ntasks),
            time_limit=str(args.time_limit),
            mem=str(args.mem),
        )

    for kmesh in kmesh_series:
        setting_name = f"k_{_kmesh_tag(kmesh)}"
        _prepare_case(
            setting_dir=outdir / "kmesh_scan" / setting_name,
            pristine=pristine,
            vacancy=vacancy,
            ecut_ev=float(args.fixed_ecut),
            kmesh=kmesh,
            pseudo_name=pseudo.name,
            pristine_prefix=f"Al_pristine_conv2x2x4_k_{_kmesh_tag(kmesh)}",
            vacancy_prefix=f"Al_vac_conv2x2x4_k_{_kmesh_tag(kmesh)}",
            job_name=f"QVCK{_kmesh_tag(kmesh).replace('x', '')}",
            force_conv_eva=float(args.force_conv),
            partition=str(args.partition),
            ntasks=int(args.ntasks),
            time_limit=str(args.time_limit),
            mem=str(args.mem),
        )

    _write_group_submit_script(outdir / "submit_grouped_ecut_scan.sh", "ecut_scan")
    _write_group_submit_script(outdir / "submit_grouped_dense_k05_ecut_scan.sh", "dense_k05_ecut_scan")
    _write_group_submit_script(outdir / "submit_grouped_kmesh_scan.sh", "kmesh_scan")
    ctest_short_settings = [
        "ecut_scan/ecut_0300eV",
        "ecut_scan/ecut_0400eV",
        "ecut_scan/ecut_0500eV",
        "ecut_scan/ecut_0600eV",
        "ecut_scan/ecut_0800eV",
        "kmesh_scan/k_01x01x01",
        "kmesh_scan/k_02x02x02",
        "kmesh_scan/k_03x03x03",
        "kmesh_scan/k_04x04x04",
    ]
    ct56_long_settings = [
        "dense_k05_ecut_scan/ecut_0400eV_k05",
        "dense_k05_ecut_scan/ecut_0500eV_k05",
        "dense_k05_ecut_scan/ecut_0600eV_k05",
        "dense_k05_ecut_scan/ecut_0800eV_k05",
        "kmesh_scan/k_05x05x05",
        "kmesh_scan/k_06x06x06",
    ]
    _write_array_submit_script(
        outdir / "submit_ctest_short_array.sh",
        job_name="QVCshort",
        partition="ctest",
        ntasks=int(args.ntasks),
        time_limit="02:00:00",
        mem=str(args.mem),
        max_parallel=2,
        settings=ctest_short_settings,
    )
    _write_array_submit_script(
        outdir / "submit_ct56_long_array.sh",
        job_name="QVClong",
        partition="ct56",
        ntasks=int(args.ntasks),
        time_limit=str(args.time_limit),
        mem=str(args.mem),
        max_parallel=2,
        settings=ct56_long_settings,
    )

    scripts_dir = outdir / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), scripts_dir / Path(__file__).name)
    collector = ROOT / "scripts" / "collect_all_qe_vacancy_recursive.py"
    if collector.exists():
        shutil.copy2(collector, scripts_dir / collector.name)

    _write_readme(
        outdir / "README.md",
        a0=float(args.a0),
        repeat=repeat,
        pristine_atoms=int(len(pristine)),
        vacancy_atoms=int(len(vacancy)),
        fixed_kmesh=fixed_kmesh,
        ecut_series=ecut_series,
        dense_kmesh=dense_kmesh,
        dense_ecut_series=dense_ecut_series,
        fixed_ecut=float(args.fixed_ecut),
        kmesh_series=kmesh_series,
        force_conv_eva=float(args.force_conv),
    )

    cell = _cell_summary(pristine)
    print("============================================================")
    print("QE conventional vacancy workflow prepared")
    print("============================================================")
    print(f"Output          : {outdir}")
    print(f"Cell basis      : conventional fcc orthorhombic {repeat[0]}x{repeat[1]}x{repeat[2]}")
    print(f"Pristine atoms  : {len(pristine)}")
    print(f"Vacancy atoms   : {len(vacancy)}")
    print(f"Cell lengths A  : {cell['lengths_A']}")
    print(f"Cell angles deg : {cell['angles_deg']}")
    print(f"Volume A^3      : {cell['volume_A3']:.6f}")
    print(f"Vacancy index   : {vacancy_site['index']}")
    print(f"Cutoff scan eV  : {ecut_series}")
    print(f"Dense k05 ecut  : {dense_ecut_series}")


if __name__ == "__main__":
    main()
