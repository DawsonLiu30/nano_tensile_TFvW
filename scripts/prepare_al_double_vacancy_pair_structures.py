from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import write


DEFAULT_A0_A = 4.039848


def parse_repeat(text: str) -> tuple[int, int, int]:
    parts = str(text).strip().lower().replace(",", "x").split("x")
    if len(parts) != 3:
        raise ValueError(f"Invalid repeat: {text}")
    repeat = tuple(int(part) for part in parts)
    if any(value <= 0 for value in repeat):
        raise ValueError(f"Repeat values must be positive: {repeat}")
    return repeat


def token(value: float) -> str:
    return f"{float(value):.4f}".replace(".", "p")


def center_on_site(atoms, target_scaled: np.ndarray = np.array([0.5, 0.5, 0.5])):
    scaled = atoms.get_scaled_positions(wrap=True)
    diff = scaled - target_scaled[None, :]
    diff -= np.round(diff)
    idx = int(np.argmin(np.sum(diff * diff, axis=1)))
    shift_scaled = target_scaled - scaled[idx]
    atoms = atoms.copy()
    atoms.translate(shift_scaled @ atoms.cell.array)
    atoms.wrap()
    scaled = atoms.get_scaled_positions(wrap=True)
    diff = scaled - target_scaled[None, :]
    diff -= np.round(diff)
    idx = int(np.argmin(np.sum(diff * diff, axis=1)))
    return atoms, idx, shift_scaled


def min_image_vector(atoms, i: int, j: int) -> np.ndarray:
    scaled = atoms.get_scaled_positions(wrap=True)
    ds = scaled[j] - scaled[i]
    ds -= np.round(ds)
    return ds @ atoms.cell.array


def same_height_candidates(atoms, center_idx: int, z_tol_A: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scaled = atoms.get_scaled_positions(wrap=True)
    for idx in range(len(atoms)):
        if idx == center_idx:
            continue
        dr = min_image_vector(atoms, center_idx, idx)
        r = float(np.linalg.norm(dr))
        if abs(float(dr[2])) <= float(z_tol_A):
            rows.append(
                {
                    "second_index": int(idx),
                    "r_A": r,
                    "dx_A": float(dr[0]),
                    "dy_A": float(dr[1]),
                    "dz_A": float(dr[2]),
                    "second_scaled": [float(v) for v in scaled[idx]],
                    "second_cart_A": [float(v) for v in atoms.positions[idx]],
                }
            )
    rows.sort(key=lambda row: float(row["r_A"]))
    return rows


def unique_by_distance(candidates: list[dict[str, object]], tol_A: float) -> list[dict[str, object]]:
    unique: list[dict[str, object]] = []
    for row in candidates:
        r = float(row["r_A"])
        if not any(abs(r - float(existing["r_A"])) <= float(tol_A) for existing in unique):
            unique.append(row)
    return unique


def write_structure(base: Path, atoms) -> None:
    write(str(base.with_suffix(".vasp")), atoms, direct=True, vasp5=True)
    write(str(base.with_suffix(".xyz")), atoms)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare conventional fcc Al double-vacancy pair structures with "
            "two vacancies on the same z-height plane."
        )
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--a0", type=float, default=DEFAULT_A0_A)
    parser.add_argument("--repeat", default="3x3x3")
    parser.add_argument("--z-tol", type=float, default=1.0e-6)
    parser.add_argument("--distance-tol", type=float, default=1.0e-4)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 means keep all unique distances.")
    args = parser.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    repeat = parse_repeat(args.repeat)
    pristine = bulk("Al", "fcc", a=float(args.a0), cubic=True).repeat(repeat)
    pristine, center_idx, shift_scaled = center_on_site(pristine)
    scaled = pristine.get_scaled_positions(wrap=True)
    candidates = same_height_candidates(pristine, center_idx, float(args.z_tol))
    unique = unique_by_distance(candidates, float(args.distance_tol))
    if int(args.max_pairs) > 0:
        unique = unique[: int(args.max_pairs)]

    write_structure(outdir / "pristine_start", pristine)

    rows: list[dict[str, object]] = []
    center_info = {
        "center_index": int(center_idx),
        "center_scaled": [float(v) for v in scaled[center_idx]],
        "center_cart_A": [float(v) for v in pristine.positions[center_idx]],
    }
    for pair_id, row in enumerate(unique, start=1):
        second_idx = int(row["second_index"])
        divac = pristine.copy()
        for idx in sorted([center_idx, second_idx], reverse=True):
            del divac[idx]
        divac.wrap()

        case = f"pair_{pair_id:02d}_r{token(float(row['r_A']))}A"
        case_dir = outdir / "pair_scan" / case
        case_dir.mkdir(parents=True, exist_ok=True)
        write_structure(case_dir / "divacancy_start", divac)
        write_structure(case_dir / "pristine_start", pristine)

        manifest = {
            "case": case,
            "cell_basis": "conventional cubic fcc",
            "a0_start_A": float(args.a0),
            "repeat": list(repeat),
            "pristine_n_atoms": int(len(pristine)),
            "divacancy_n_atoms": int(len(divac)),
            "vacancy_count": 2,
            "vacancy_concentration_percent": 200.0 / float(len(pristine)),
            "first_vacancy": center_info,
            "second_vacancy": row,
            "pair_distance_min_image_A": float(row["r_A"]),
            "same_height_condition": f"|dz| <= {float(args.z_tol):.3e} A",
            "formation_energy_formula": (
                "E_f^2vac(r) = E_divac^(N-2,r) - ((N-2)/N) E_pristine^N; "
                "per-vacancy value = E_f^2vac / 2"
            ),
        }
        (case_dir / "pair_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        rows.append(
            {
                "case": case,
                "r_A": f"{float(row['r_A']):.8f}",
                "dx_A": f"{float(row['dx_A']):.8f}",
                "dy_A": f"{float(row['dy_A']):.8f}",
                "dz_A": f"{float(row['dz_A']):.8f}",
                "pristine_n_atoms": len(pristine),
                "divacancy_n_atoms": len(divac),
                "vacancy_concentration_percent": f"{200.0 / float(len(pristine)):.8f}",
                "case_dir": str(case_dir),
            }
        )

    with (outdir / "double_vacancy_pair_plan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "workflow": "al_double_vacancy_pair_structure_plan",
        "a0_start_A": float(args.a0),
        "repeat": list(repeat),
        "cell_lengths_A": [float(v) for v in pristine.cell.lengths()],
        "cell_angles_deg": [float(v) for v in pristine.cell.angles()],
        "cell_volume_A3": float(pristine.get_volume()),
        "all_cell_lengths_exceed_10_A": all(float(v) > 10.0 for v in pristine.cell.lengths()),
        "pristine_n_atoms": int(len(pristine)),
        "divacancy_n_atoms": int(len(pristine) - 2),
        "origin_shift_scaled": [float(v) for v in shift_scaled],
        "first_vacancy": center_info,
        "unique_same_height_pair_count": len(rows),
        "pair_distances_A": [float(row["r_A"]) for row in rows],
    }
    (outdir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Plan CSV: {outdir / 'double_vacancy_pair_plan.csv'}")


if __name__ == "__main__":
    main()
