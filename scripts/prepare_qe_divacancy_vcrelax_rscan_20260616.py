from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import write


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_qe_vacancy_vcrelax_3x3x3 import (  # noqa: E402
    DEFAULT_A0_A,
    cell_summary,
    parse_kmesh_list,
    parse_repeat,
    write_array,
    write_group_job,
    write_vcrelax_input,
)


def safe_distance_label(distance_a: float) -> str:
    return f"r{distance_a:.4f}A".replace(".", "p")


def write_structure_pair(base: Path, atoms) -> None:
    write(str(base.with_suffix(".vasp")), atoms, direct=True, vasp5=True)
    write(str(base.with_suffix(".xyz")), atoms)


def build_centered_pristine(a0: float, repeat: tuple[int, int, int]):
    atoms = bulk("Al", "fcc", a=a0, cubic=True).repeat(repeat)
    scaled = atoms.get_scaled_positions(wrap=True)
    target = np.array([0.5, 0.5, 0.5])
    diff = scaled - target
    diff -= np.round(diff)
    distances = np.linalg.norm(diff @ atoms.get_cell().array, axis=1)
    center_index = int(np.argmin(distances))
    shift_scaled = target - scaled[center_index]
    atoms.set_scaled_positions((scaled + shift_scaled) % 1.0)
    atoms.wrap()
    return atoms, center_index, shift_scaled


def enumerate_same_height_pairs(pristine, center_index: int, z_tol: float, distance_tol: float):
    positions = pristine.get_positions()
    center = positions[center_index]
    candidates = []
    for idx, pos in enumerate(positions):
        if idx == center_index:
            continue
        delta = pos - center
        if abs(float(delta[2])) > z_tol:
            continue
        distance = float(np.linalg.norm(delta))
        if distance <= distance_tol:
            continue
        candidates.append((distance, idx, delta))

    grouped = []
    for distance, idx, delta in sorted(candidates, key=lambda item: item[0]):
        if grouped and abs(distance - grouped[-1][0]) <= distance_tol:
            continue
        grouped.append((distance, idx, delta))
    return grouped


def remove_two_atoms(atoms, first_index: int, second_index: int):
    divacancy = atoms.copy()
    for idx in sorted([first_index, second_index], reverse=True):
        del divacancy[idx]
    return divacancy


def prepare_case(
    root: Path,
    rel_setting: str,
    *,
    pristine,
    divacancy,
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
        prefix=f"Al_pristine_divac_{rel_setting.replace('/', '_')}",
        atoms=pristine,
        ecut_ev=ecut_ev,
        kmesh=kmesh,
        pseudo_name=pseudo_name,
        force_conv_eva=force_conv_eva,
        press_conv_kbar=press_conv_kbar,
    )
    write_vcrelax_input(
        setting_dir / "vacancy_vcrelax" / "vc-relax.in",
        prefix=f"Al_divac_{rel_setting.replace('/', '_')}",
        atoms=divacancy,
        ecut_ev=ecut_ev,
        kmesh=kmesh,
        pseudo_name=pseudo_name,
        force_conv_eva=force_conv_eva,
        press_conv_kbar=press_conv_kbar,
    )
    write_group_job(
        setting_dir / "group_job.sh",
        job_name=f"QDV{rel_setting.split('/')[-1].replace('_', '')[:8]}",
        partition=partition,
        ntasks=ntasks,
        time_limit=time_limit,
        mem=mem,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare QE/PBE vc-relax scan for same-height divacancy pairs in conventional fcc Al."
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--pseudo", required=True)
    parser.add_argument("--a0", type=float, default=DEFAULT_A0_A)
    parser.add_argument("--repeat", default="3x3x3")
    parser.add_argument("--ecut", type=float, default=800.0)
    parser.add_argument("--kmesh", default="3x3x3")
    parser.add_argument("--force-conv", type=float, default=0.002)
    parser.add_argument("--press-conv-kbar", type=float, default=0.5)
    parser.add_argument("--partition", default="ct56")
    parser.add_argument("--ntasks", type=int, default=28)
    parser.add_argument("--time-limit", default="4-00:00:00")
    parser.add_argument("--mem", default="128G")
    parser.add_argument("--max-parallel", type=int, default=5)
    parser.add_argument("--z-tol", type=float, default=1.0e-6)
    parser.add_argument("--distance-tol", type=float, default=1.0e-4)
    parser.add_argument("--max-pairs", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    pseudo = Path(args.pseudo).expanduser().resolve()
    if not pseudo.exists():
        raise FileNotFoundError(f"Missing pseudo: {pseudo}")
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)

    repeat = parse_repeat(args.repeat)
    kmesh = parse_kmesh_list(args.kmesh)[0]
    pristine, center_index, shift_scaled = build_centered_pristine(args.a0, repeat)
    pairs = enumerate_same_height_pairs(pristine, center_index, args.z_tol, args.distance_tol)
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    psp_dir = outdir / "psp"
    psp_dir.mkdir()
    pseudo_name = pseudo.name
    shutil.copy2(pseudo, psp_dir / pseudo_name)
    write_structure_pair(outdir / "pristine_start", pristine)

    settings: list[str] = []
    rows: list[dict[str, object]] = []
    pair_root = outdir / "pair_scan"
    pair_root.mkdir()
    n_pristine = len(pristine)

    for case_idx, (distance, second_index, delta) in enumerate(pairs, start=1):
        setting_name = f"pair_{case_idx:02d}_{safe_distance_label(distance)}"
        rel_setting = f"pair_scan/{setting_name}"
        case_dir = outdir / rel_setting
        divacancy = remove_two_atoms(pristine, center_index, second_index)
        prepare_case(
            outdir,
            rel_setting,
            pristine=pristine,
            divacancy=divacancy,
            ecut_ev=args.ecut,
            kmesh=kmesh,
            pseudo_name=pseudo_name,
            force_conv_eva=args.force_conv,
            press_conv_kbar=args.press_conv_kbar,
            partition=args.partition,
            ntasks=args.ntasks,
            time_limit=args.time_limit,
            mem=args.mem,
        )
        write_structure_pair(case_dir / "pristine_start", pristine)
        write_structure_pair(case_dir / "divacancy_start", divacancy)
        manifest = {
            "setting": setting_name,
            "scan_type": "pair",
            "code": "Quantum ESPRESSO",
            "functional": "PBE",
            "pseudo": pseudo_name,
            "relaxation": "vc-relax for pristine and divacancy",
            "a0_start_A": args.a0,
            "conventional_repeat": list(repeat),
            "cell_summary": cell_summary(pristine),
            "all_cell_lengths_exceed_10_A": all(float(x) > 10.0 for x in pristine.cell.lengths()),
            "pristine_n_atoms": n_pristine,
            "vacancy_n_atoms": len(divacancy),
            "vacancy_count": n_pristine - len(divacancy),
            "vacancy_concentration_percent": 100.0 * (n_pristine - len(divacancy)) / n_pristine,
            "first_vacancy_index": center_index,
            "second_vacancy_index": second_index,
            "pair_distance_A": distance,
            "pair_delta_A": [float(x) for x in delta],
            "ecut_eV": args.ecut,
            "kmesh": list(kmesh),
            "force_conv_eV_A": args.force_conv,
            "press_conv_kbar": args.press_conv_kbar,
            "formation_energy_formula": "E_f^2vac(r) = E_divac^(N-2,r) - ((N-2)/N) E_pristine^N",
            "per_vacancy_formula": "E_f^2vac(r) / 2",
        }
        (case_dir / "pair_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        settings.append(rel_setting)
        rows.append(
            {
                "case": setting_name,
                "r_A": f"{distance:.8f}",
                "dx_A": f"{float(delta[0]):.8f}",
                "dy_A": f"{float(delta[1]):.8f}",
                "dz_A": f"{float(delta[2]):.8f}",
                "N_pristine": n_pristine,
                "N_divacancy": len(divacancy),
                "vacancy_count": n_pristine - len(divacancy),
                "vacancy_concentration_percent": f"{100.0 * (n_pristine - len(divacancy)) / n_pristine:.8f}",
                "source_dir": str(case_dir),
                "input_files": "pristine_vcrelax/vc-relax.in; vacancy_vcrelax/vc-relax.in; pair_manifest.json",
                "expected_output_files": "pristine_vcrelax/vc-relax.out; vacancy_vcrelax/vc-relax.out",
                "ecut_eV": args.ecut,
                "kmesh": "x".join(str(x) for x in kmesh),
            }
        )

    write_array(
        outdir / "submit_qe_divacancy_pair_array.sh",
        settings=settings,
        max_parallel=args.max_parallel,
        partition=args.partition,
        ntasks=args.ntasks,
        time_limit=args.time_limit,
        mem=args.mem,
    )
    with (outdir / "qe_divacancy_pair_plan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    top_manifest = {
        "workflow": "qe_pbe_al_divacancy_pair_rscan",
        "root": str(outdir),
        "pair_count": len(settings),
        "cell_lengths_A": [float(x) for x in pristine.cell.lengths()],
        "all_cell_lengths_exceed_10_A": all(float(x) > 10.0 for x in pristine.cell.lengths()),
        "settings": settings,
        "submit_command": f"cd {outdir} && sbatch submit_qe_divacancy_pair_array.sh",
    }
    (outdir / "manifest.json").write_text(json.dumps(top_manifest, indent=2), encoding="utf-8")
    print(json.dumps(top_manifest, indent=2))


if __name__ == "__main__":
    main()
