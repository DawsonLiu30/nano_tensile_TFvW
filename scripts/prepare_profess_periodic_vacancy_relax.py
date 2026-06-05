#!/usr/bin/env python3
"""Prepare PROFESS fixed-cell ion-relax cases for periodic vacancy columns.

The generated structures are axially periodic.  ``circle`` is used for
nanocolumns and polygonal cross sections are used for faceted nanocolumns.
Vacancies are placed at reproducible radial positions: inner, middle, outer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ase_nanocrystal import build_periodic_prism


HA_PER_BOHR_TO_EV_PER_A = 51.4220674763


KEDF_TEMPLATES: dict[str, list[str]] = {
    "TFPLUS_DEFAULT": ["KINE TF+"],
    "TFVW_L1_M1": ["KINE TF+", "PARA LAMB 1", "PARA MU 1"],
    "CAT": ["KINE CAT"],
    "HC": ["KINE HC", "PARA BETA 0.460655337083368"],
    "WT": ["KINE WT"],
    "WGC": [
        "KINE WGC",
        "WGCT 2",
        "PARA LAMB 1",
        "PARA MU 1",
        "PARA ALPHA 1.206011329583298",
        "PARA BETA 0.460655337083368",
        "PARA GAMMA 2.7",
    ],
}


def parse_csv_list(text: str) -> list[str]:
    values = [x.strip() for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError(f"Empty comma-separated list: {text!r}")
    return values


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in parse_csv_list(text)]


def normalize_position(value: str) -> str:
    key = str(value).strip().lower()
    aliases = {
        "inner": "inner",
        "core": "inner",
        "center": "inner",
        "centre": "inner",
        "middle": "middle",
        "mid": "middle",
        "outer": "outer",
        "surface": "outer",
        "edge": "outer",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported vacancy position {value!r}")
    return aliases[key]


def family(shape: str) -> str:
    return "nanocolumn" if shape == "circle" else "faceted_nanocolumn"


def radius_tag(diameter_nm: float) -> str:
    return f"{float(diameter_nm):.2f}".replace(".", "p")


def build_short_periodic_structure(
    *,
    a0: float,
    diameter_nm: float,
    min_lz_A: float,
    vacuum_A: float,
    orientation: str,
    shape: str,
    shape_rotation_deg: float,
):
    base = build_periodic_prism(
        a0=float(a0),
        diameter_nm=float(diameter_nm),
        length_z=1.0,
        vacuum=float(vacuum_A),
        orientation=str(orientation),
        cross_section_shape=str(shape),
        shape_rotation_deg=float(shape_rotation_deg),
    )
    base_lz = float(base.cell.lengths()[2])
    repeat_z = max(1, int(math.ceil(float(min_lz_A) / max(base_lz, 1e-12))))
    return base.repeat((1, 1, repeat_z)), repeat_z


def choose_vacancy_index(atoms, radial_position: str, z_window_fraction: float) -> tuple[int, dict[str, object]]:
    radial_position = normalize_position(radial_position)
    pos = atoms.get_positions()
    cell = atoms.get_cell().array
    cx = 0.5 * float(cell[0, 0])
    cy = 0.5 * float(cell[1, 1])
    z_center = float(np.mean(pos[:, 2]))
    radial = np.hypot(pos[:, 0] - cx, pos[:, 1] - cy)
    z_offset = np.abs(pos[:, 2] - z_center)
    z_window = max(float(z_window_fraction) * float(atoms.cell.lengths()[2]), 1e-6)
    candidates = np.where(z_offset <= z_window)[0]
    if len(candidates) == 0:
        candidates = np.arange(len(atoms), dtype=int)
    candidate_radial = radial[candidates]
    outer = float(np.max(candidate_radial))

    if radial_position == "inner":
        chosen = int(candidates[int(np.argmin(candidate_radial))])
        rule = "minimum radial distance in central z window"
        target = float(np.min(candidate_radial))
    elif radial_position == "middle":
        target = 0.5 * outer
        chosen = int(candidates[int(np.argmin(np.abs(candidate_radial - target)))])
        rule = "closest to half outer radial distance in central z window"
    else:
        chosen = int(candidates[int(np.argmax(candidate_radial))])
        rule = "maximum radial distance in central z window"
        target = outer

    site = {
        "vacancy_index": chosen,
        "vacancy_position": radial_position,
        "selection_rule": rule,
        "vacancy_x_A": float(pos[chosen, 0]),
        "vacancy_y_A": float(pos[chosen, 1]),
        "vacancy_z_A": float(pos[chosen, 2]),
        "vacancy_radial_A": float(radial[chosen]),
        "vacancy_radial_fraction": float(radial[chosen] / outer) if outer > 0 else math.nan,
        "target_radial_A": target,
        "outer_candidate_radial_A": outer,
        "z_window_A": z_window,
    }
    return chosen, site


def remove_atom(atoms, index: int):
    defect = atoms.copy()
    del defect[int(index)]
    return defect


def vasp_to_profess_ion(vasp_path: Path, ion_path: Path, pseudo_name: str) -> dict[str, float]:
    atoms = read(str(vasp_path), format="vasp")
    cell = atoms.cell.array
    scaled = atoms.get_scaled_positions(wrap=False)
    symbols = atoms.get_chemical_symbols()
    with ion_path.open("w", encoding="utf-8") as fh:
        fh.write("%BLOCK LATTICE_CART\n")
        for vec in cell:
            fh.write(f"{vec[0]:18.10f} {vec[1]:18.10f} {vec[2]:18.10f}\n")
        fh.write("%END BLOCK LATTICE_CART\n")
        fh.write("%BLOCK POSITIONS_FRAC\n")
        for sym, frac in zip(symbols, scaled):
            fh.write(f"{sym:2s} {frac[0]:18.10f} {frac[1]:18.10f} {frac[2]:18.10f}\n")
        fh.write("%END BLOCK POSITIONS_FRAC\n")
        fh.write("%BLOCK SPECIES_POT\n")
        fh.write(f"    Al {pseudo_name}\n")
        fh.write("%END BLOCK SPECIES_POT\n")
    lengths = atoms.cell.lengths()
    angles = atoms.cell.angles()
    return {
        "natoms": len(atoms),
        "volume_A3": float(atoms.get_volume()),
        "a_A": float(lengths[0]),
        "b_A": float(lengths[1]),
        "c_A": float(lengths[2]),
        "alpha_deg": float(angles[0]),
        "beta_deg": float(angles[1]),
        "gamma_deg": float(angles[2]),
    }


def write_profess_inpt(
    path: Path,
    *,
    stem: str,
    ecut: float,
    kedf: str,
    mode: str,
    ion_method: str,
    ion_tolf_ev_a: float,
) -> None:
    if kedf not in KEDF_TEMPLATES:
        raise ValueError(f"Unsupported KEDF {kedf!r}; supported: {sorted(KEDF_TEMPLATES)}")
    lines = [f"ecut {float(ecut):g}"]
    if mode == "relax":
        lines.extend(
            [
                "MINI ion",
                "method ntn",
                f"method ion {ion_method}",
                f"TOLF {float(ion_tolf_ev_a) / HA_PER_BOHR_TO_EV_PER_A:.12g}",
            ]
        )
    else:
        lines.append("method ntn")
    lines.extend(KEDF_TEMPLATES[kedf])
    lines.extend(
        [
            "exch lda",
            f"geometryfile {stem}.ion",
            "",
            "print minimizer density 2",
            "print minimizer geom 2",
            "calculate stresses",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_runner_scripts(outdir: Path, settings_file: Path) -> None:
    (outdir / "run_one_profess_relax_case.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$1"
PROFESS_BIN="${PROFESS_BIN:-/gpfs-work/dawson666/profess3_build_20260605/bin/PROFESS}"

cd "${CASE_DIR}"
echo "[INFO] host=$(hostname)"
echo "[INFO] case_dir=${CASE_DIR}"
echo "[INFO] profess=${PROFESS_BIN}"

for stem in pristine_relax vacancy_relax; do
  echo "============================================================"
  echo "[RUN] ${stem}"
  if [[ -s "${stem}.out" ]] && grep -q "END OF PROFESS\\|Total Run Time" "${stem}.out"; then
    echo "[SKIP] ${stem}.out already has an end marker"
    continue
  fi
  "${PROFESS_BIN}" "${stem}" > "${stem}.stdout" 2> "${stem}.stderr"
done
""",
        encoding="utf-8",
    )
    (outdir / "submit_profess_relax_array.sh").write_text(
        f"""#!/usr/bin/env bash
#SBATCH -J PROFvac
#SBATCH -A MST114175
#SBATCH -p ctest
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SETTINGS="${{SETTINGS:-{settings_file.name}}}"
TASK_ID="${{SLURM_ARRAY_TASK_ID:-0}}"
CASE_REL="$(sed -n "$((TASK_ID + 1))p" "${{ROOT}}/${{SETTINGS}}" | cut -f1)"
if [[ -z "${{CASE_REL}}" ]]; then
  echo "[ERROR] No setting for task ${{TASK_ID}}" >&2
  exit 2
fi

mkdir -p "${{ROOT}}/logs"
bash "${{ROOT}}/run_one_profess_relax_case.sh" "${{ROOT}}/${{CASE_REL}}"
""",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--a0", type=float, default=4.039848)
    ap.add_argument("--diameters", default="1.0,1.5,2.0")
    ap.add_argument("--shapes", default="circle,hexagon")
    ap.add_argument("--orientation", default="111")
    ap.add_argument("--positions", default="inner,middle,outer")
    ap.add_argument("--kedfs", default="TFPLUS_DEFAULT,CAT")
    ap.add_argument("--vacuum", type=float, default=10.0)
    ap.add_argument("--min-lz", type=float, default=10.0)
    ap.add_argument("--shape-rotation-deg", type=float, default=0.0)
    ap.add_argument("--z-window-fraction", type=float, default=0.25)
    ap.add_argument("--ecut", type=float, default=1600.0)
    ap.add_argument("--ion-method", default="cg2")
    ap.add_argument("--ion-tolf-ev-a", type=float, default=0.002)
    ap.add_argument(
        "--pseudo",
        default="/gpfs-work/dawson666/profess3_build_20260605/src/test/optCell/al_HC.lda.recpot",
        type=Path,
    )
    args = ap.parse_args()

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    pseudo_src = args.pseudo
    if not pseudo_src.exists():
        raise FileNotFoundError(f"Pseudo not found: {pseudo_src}")
    pseudo_name = pseudo_src.name
    pseudo_dst = outdir / pseudo_name
    pseudo_dst.write_bytes(pseudo_src.read_bytes())

    diameters = parse_float_list(args.diameters)
    shapes = [s.lower() for s in parse_csv_list(args.shapes)]
    positions = [normalize_position(p) for p in parse_csv_list(args.positions)]
    kedfs = parse_csv_list(args.kedfs)

    structure_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    settings_rows: list[dict[str, object]] = []

    for shape in shapes:
        for diameter_nm in diameters:
            pristine, repeat_z = build_short_periodic_structure(
                a0=args.a0,
                diameter_nm=diameter_nm,
                min_lz_A=args.min_lz,
                vacuum_A=args.vacuum,
                orientation=args.orientation,
                shape=shape,
                shape_rotation_deg=args.shape_rotation_deg,
            )
            for vacancy_position in positions:
                vac_index, site = choose_vacancy_index(
                    pristine,
                    radial_position=vacancy_position,
                    z_window_fraction=args.z_window_fraction,
                )
                vacancy = remove_atom(pristine, vac_index)
                base_name = (
                    f"{family(shape)}_{shape}_periodic_{args.orientation}_"
                    f"{radius_tag(diameter_nm)}nm_vac_{vacancy_position}"
                )

                # Keep one canonical structure copy so DFTpy and PROFESS can
                # later be generated from exactly the same VASP files.
                structure_rel = Path("structures") / base_name
                structure_dir = outdir / structure_rel
                structure_dir.mkdir(parents=True, exist_ok=True)
                pristine_vasp = structure_dir / "pristine_start.vasp"
                vacancy_vasp = structure_dir / "vacancy_start.vasp"
                write(str(pristine_vasp), pristine, vasp5=True, direct=True)
                write(str(vacancy_vasp), vacancy, vasp5=True, direct=True)
                structure_manifest = {
                    "case": base_name,
                    "family": family(shape),
                    "shape": shape,
                    "orientation": args.orientation,
                    "diameter_nm": diameter_nm,
                    "a0_A": args.a0,
                    "vacuum_A": args.vacuum,
                    "min_lz_A": args.min_lz,
                    "repeat_z": repeat_z,
                    "N_pristine": len(pristine),
                    "N_vacancy": len(vacancy),
                    "vacancy_concentration_percent": 100.0 / len(pristine),
                    "structure_rel": str(structure_rel).replace("\\", "/"),
                    "pristine_volume_A3": float(pristine.get_volume()),
                    "vacancy_volume_A3": float(vacancy.get_volume()),
                    **site,
                }
                (structure_dir / "structure_metadata.json").write_text(
                    json.dumps(structure_manifest, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                structure_rows.append(structure_manifest)

                for kedf in kedfs:
                    case_rel = Path("profess") / kedf / base_name
                    case_dir = outdir / case_rel
                    case_dir.mkdir(parents=True, exist_ok=True)
                    (case_dir / pseudo_name).write_bytes(pseudo_dst.read_bytes())
                    (case_dir / "pristine_start.vasp").write_bytes(pristine_vasp.read_bytes())
                    (case_dir / "vacancy_start.vasp").write_bytes(vacancy_vasp.read_bytes())
                    pristine_meta = vasp_to_profess_ion(
                        case_dir / "pristine_start.vasp",
                        case_dir / "pristine_relax.ion",
                        pseudo_name,
                    )
                    vacancy_meta = vasp_to_profess_ion(
                        case_dir / "vacancy_start.vasp",
                        case_dir / "vacancy_relax.ion",
                        pseudo_name,
                    )
                    write_profess_inpt(
                        case_dir / "pristine_relax.inpt",
                        stem="pristine_relax",
                        ecut=args.ecut,
                        kedf=kedf,
                        mode="relax",
                        ion_method=args.ion_method,
                        ion_tolf_ev_a=args.ion_tolf_ev_a,
                    )
                    write_profess_inpt(
                        case_dir / "vacancy_relax.inpt",
                        stem="vacancy_relax",
                        ecut=args.ecut,
                        kedf=kedf,
                        mode="relax",
                        ion_method=args.ion_method,
                        ion_tolf_ev_a=args.ion_tolf_ev_a,
                    )
                    manifest = {
                        "case": base_name,
                        "kedf": kedf,
                        "family": family(shape),
                        "shape": shape,
                        "orientation": args.orientation,
                        "diameter_nm": diameter_nm,
                        "a0_A": args.a0,
                        "vacuum_A": args.vacuum,
                        "min_lz_A": args.min_lz,
                        "repeat_z": repeat_z,
                        "N_pristine": len(pristine),
                        "N_vacancy": len(vacancy),
                        "vacancy_concentration_percent": 100.0 / len(pristine),
                        "ecut_eV": args.ecut,
                        "ion_method": args.ion_method,
                        "ion_tolf_eV_A": args.ion_tolf_ev_a,
                        "pseudo": pseudo_name,
                        "structure_rel": str(structure_rel).replace("\\", "/"),
                        **site,
                        "pristine_volume_A3": pristine_meta["volume_A3"],
                        "vacancy_volume_A3": vacancy_meta["volume_A3"],
                    }
                    (case_dir / "manifest.json").write_text(
                        json.dumps(manifest, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                    summary_rows.append({**manifest, "case_rel": str(case_rel).replace("\\", "/")})
                    settings_rows.append({"case_rel": str(case_rel).replace("\\", "/")})

    write_csv(outdir / "shared_structure_manifest.csv", structure_rows)
    write_csv(outdir / "profess_case_manifest.csv", summary_rows)
    # Backward-compatible name for scripts/notebooks that already look for it.
    write_csv(outdir / "structure_manifest.csv", summary_rows)

    (outdir / "dftpy").mkdir(exist_ok=True)
    (outdir / "dftpy" / "README_SAME_STRUCTURES.txt").write_text(
        "DFTpy cases should be generated from ../structures/<case>/pristine_start.vasp "
        "and ../structures/<case>/vacancy_start.vasp so DFTpy and PROFESS use the "
        "same structures.\n",
        encoding="utf-8",
    )
    (outdir / "README_LAYOUT.txt").write_text(
        "\n".join(
            [
                "Periodic vacancy benchmark package layout",
                "",
                "structures/<case>/",
                "  Canonical VASP structures shared by all calculators.",
                "",
                "profess/<KEDF>/<case>/",
                "  PROFESS input/output generated from structures/<case>/.",
                "",
                "dftpy/",
                "  Reserved for DFTpy input/output generated from the same structures.",
                "",
                "This separation avoids mixing calculator outputs while preserving a single",
                "source of truth for geometry.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    settings_file = outdir / "settings.tsv"
    with settings_file.open("w", encoding="utf-8") as fh:
        for row in settings_rows:
            fh.write(f"{row['case_rel']}\n")
    write_runner_scripts(outdir, settings_file)
    for script in ["run_one_profess_relax_case.sh", "submit_profess_relax_array.sh"]:
        (outdir / script).chmod(0o755)

    print("============================================================")
    print("Prepared PROFESS periodic vacancy relax package")
    print("============================================================")
    print(f"Outdir      : {outdir}")
    print(f"Cases       : {len(settings_rows)}")
    print(f"Manifest    : {outdir / 'structure_manifest.csv'}")
    print(f"Settings    : {settings_file}")
    print("Submit hint :")
    last = len(settings_rows) - 1
    print(f"  cd {outdir} && sbatch --array=0-{last}%2 submit_profess_relax_array.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
