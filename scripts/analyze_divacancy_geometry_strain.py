from __future__ import annotations

import argparse
import csv
import json
import math
from fractions import Fraction
from functools import reduce
from pathlib import Path

import numpy as np
from ase.io import read


def last_bfgs_fmax(path: Path) -> float:
    value = math.nan
    if not path.exists():
        return value
    for line in path.read_text(errors="ignore").splitlines():
        if line.strip().startswith("BFGS:"):
            try:
                value = float(line.split()[-1])
            except (ValueError, IndexError):
                pass
    return value


def minimum_image_vectors(frac_delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    wrapped = frac_delta - np.round(frac_delta)
    return wrapped @ cell


def crystallographic_direction(frac_delta: np.ndarray) -> str:
    wrapped = frac_delta - np.round(frac_delta)
    fractions = [Fraction(abs(float(value))).limit_denominator(48) for value in wrapped]
    denominators = [value.denominator for value in fractions]
    common = math.lcm(*denominators)
    integers = [int(value * common) for value in fractions]
    nonzero = [value for value in integers if value]
    divisor = reduce(math.gcd, nonzero) if nonzero else 1
    reduced = [value // divisor for value in integers]
    return "[" + "".join(str(value) for value in reduced) + "]"


def local_bond_strain(start, final, cutoff_a: float) -> tuple[np.ndarray, np.ndarray]:
    n_atoms = len(start)
    start_scaled = start.get_scaled_positions(wrap=True)
    final_scaled = final.get_scaled_positions(wrap=True)
    start_cell = start.cell.array
    final_cell = final.cell.array
    total_values: list[list[float]] = [[] for _ in range(n_atoms)]
    nonaffine_values: list[list[float]] = [[] for _ in range(n_atoms)]

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            d0 = np.linalg.norm(minimum_image_vectors(start_scaled[j] - start_scaled[i], start_cell))
            if d0 <= 1.0e-10 or d0 > cutoff_a:
                continue
            d_affine = np.linalg.norm(
                minimum_image_vectors(start_scaled[j] - start_scaled[i], final_cell)
            )
            d1 = np.linalg.norm(minimum_image_vectors(final_scaled[j] - final_scaled[i], final_cell))
            total_strain = float((d1 - d0) / d0)
            nonaffine_strain = float((d1 - d_affine) / d_affine)
            total_values[i].append(total_strain)
            total_values[j].append(total_strain)
            nonaffine_values[i].append(nonaffine_strain)
            nonaffine_values[j].append(nonaffine_strain)

    total = np.array([float(np.mean(row)) if row else math.nan for row in total_values])
    nonaffine = np.array([float(np.mean(row)) if row else math.nan for row in nonaffine_values])
    return total, nonaffine


def as_float(value, default=math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root: Path, output: Path, cutoff_a: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    case_rows: list[dict[str, object]] = []
    atom_rows: list[dict[str, object]] = []

    for manifest_path in sorted((root / "pair_scan").glob("*/point_manifest.json")):
        case_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result_path = case_dir / "result.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))

        start_path = case_dir / "divacancy_start.vasp"
        if not start_path.exists():
            start_path = case_dir / "vacancy_start.vasp"
        final_path = case_dir / "divacancy_vc_relaxed.vasp"
        if not final_path.exists():
            final_path = case_dir / "vacancy_vc_relaxed.vasp"

        pristine = read(case_dir / "pristine_raw.vasp")
        start = read(start_path)
        final = read(final_path)
        if len(start) != len(final):
            raise ValueError(f"Atom-count mismatch: {case_dir}")

        first_index = int(manifest["first_vacancy_index"])
        second_index = int(manifest["second_vacancy_index"])
        pristine_scaled = pristine.get_scaled_positions(wrap=True)
        vacancy_scaled = pristine_scaled[[first_index, second_index]]
        pair_frac = vacancy_scaled[1] - vacancy_scaled[0]
        pair_frac -= np.round(pair_frac)
        pair_vector = minimum_image_vectors(pair_frac, pristine.cell.array)
        pair_vector_final_affine = minimum_image_vectors(pair_frac, final.cell.array)

        start_scaled = start.get_scaled_positions(wrap=True)
        final_scaled = final.get_scaled_positions(wrap=True)
        non_affine = minimum_image_vectors(final_scaled - start_scaled, final.cell.array)
        displacement = np.linalg.norm(non_affine, axis=1)
        bond_strain, nonaffine_bond_strain = local_bond_strain(start, final, cutoff_a)

        distances_to_sites = []
        for atom_scaled in start_scaled:
            site_distances = [
                np.linalg.norm(minimum_image_vectors(atom_scaled - site, pristine.cell.array))
                for site in vacancy_scaled
            ]
            distances_to_sites.append(min(site_distances))

        start_cell = pristine.cell.array
        final_cell = final.cell.array
        deformation_gradient = np.linalg.solve(start_cell, final_cell)
        green_strain = 0.5 * (deformation_gradient.T @ deformation_gradient - np.eye(3))
        lengths = pristine.cell.lengths()
        max_minimum_image_distance = float(np.linalg.norm(lengths / 2.0))

        defect_log = case_dir / "divacancy_relax.log"
        if not defect_log.exists():
            defect_log = case_dir / "vacancy_relax.log"

        case_rows.append(
            {
                "case": case_dir.name,
                "initial_pair_distance_A": float(np.linalg.norm(pair_vector)),
                "crystallographic_direction_family": crystallographic_direction(pair_frac),
                "initial_pair_dx_A": float(pair_vector[0]),
                "initial_pair_dy_A": float(pair_vector[1]),
                "initial_pair_dz_A": float(pair_vector[2]),
                "affine_final_pair_distance_A": float(np.linalg.norm(pair_vector_final_affine)),
                "cell_a_A": float(lengths[0]),
                "cell_b_A": float(lengths[1]),
                "cell_c_A": float(lengths[2]),
                "maximum_minimum_image_distance_A": max_minimum_image_distance,
                "E_2vac_eV": as_float(result.get("vacancy_formation_energy_eV")),
                "pristine_final_fmax_eV_A": as_float(
                    result.get("pristine_final_fmax_eV_A"), last_bfgs_fmax(case_dir / "pristine_relax.log")
                ),
                "divacancy_final_fmax_eV_A": as_float(
                    result.get("divacancy_final_fmax_eV_A"), last_bfgs_fmax(defect_log)
                ),
                "mean_nonaffine_displacement_A": float(np.mean(displacement)),
                "max_nonaffine_displacement_A": float(np.max(displacement)),
                "mean_abs_local_bond_strain": float(np.nanmean(np.abs(bond_strain))),
                "max_abs_local_bond_strain": float(np.nanmax(np.abs(bond_strain))),
                "mean_abs_nonaffine_bond_strain": float(np.nanmean(np.abs(nonaffine_bond_strain))),
                "max_abs_nonaffine_bond_strain": float(np.nanmax(np.abs(nonaffine_bond_strain))),
                "global_green_strain_xx": float(green_strain[0, 0]),
                "global_green_strain_yy": float(green_strain[1, 1]),
                "global_green_strain_zz": float(green_strain[2, 2]),
                "source_dir": str(case_dir),
            }
        )

        for atom_index, (distance, disp, strain, nonaffine_strain) in enumerate(
            zip(distances_to_sites, displacement, bond_strain, nonaffine_bond_strain)
        ):
            atom_rows.append(
                {
                    "case": case_dir.name,
                    "atom_index": atom_index,
                    "initial_distance_to_nearest_vacancy_A": float(distance),
                    "nonaffine_displacement_A": float(disp),
                    "local_mean_bond_strain": float(strain),
                    "local_mean_nonaffine_bond_strain": float(nonaffine_strain),
                }
            )

    case_rows.sort(key=lambda row: float(row["initial_pair_distance_A"]))
    write_csv(output / "divacancy_geometry_strain_summary.csv", case_rows)
    write_csv(output / "divacancy_atom_displacement_strain.csv", atom_rows)

    trend_rows: list[dict[str, object]] = []
    for previous, current in zip(case_rows, case_rows[1:]):
        trend_rows.append(
            {
                "from_case": previous["case"],
                "to_case": current["case"],
                "from_r_A": previous["initial_pair_distance_A"],
                "to_r_A": current["initial_pair_distance_A"],
                "delta_r_A": float(current["initial_pair_distance_A"]) - float(previous["initial_pair_distance_A"]),
                "delta_E_2vac_eV": float(current["E_2vac_eV"]) - float(previous["E_2vac_eV"]),
            }
        )
    write_csv(output / "divacancy_energy_trend_deltas.csv", trend_rows)
    return case_rows, atom_rows


def make_plots(
    output: Path,
    case_rows: list[dict[str, object]],
    atom_rows: list[dict[str, object]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    cases = sorted({str(row["case"]) for row in atom_rows})
    for case in cases:
        selected = [row for row in atom_rows if row["case"] == case]
        distance = [float(row["initial_distance_to_nearest_vacancy_A"]) for row in selected]
        displacement = [float(row["nonaffine_displacement_A"]) for row in selected]
        strain = [float(row["local_mean_nonaffine_bond_strain"]) for row in selected]
        axes[0].scatter(distance, displacement, s=8, alpha=0.55, label=case)
        axes[1].scatter(distance, strain, s=8, alpha=0.55)

    axes[0].set(xlabel="Distance to nearest vacancy (A)", ylabel="Non-affine displacement (A)")
    axes[1].set(xlabel="Distance to nearest vacancy (A)", ylabel="Local non-affine bond-strain proxy")
    axes[0].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "divacancy_displacement_and_strain_proxy.png", dpi=300)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.2, 3.6))
    directions = sorted({str(row["crystallographic_direction_family"]) for row in case_rows})
    for direction in directions:
        selected = [row for row in case_rows if row["crystallographic_direction_family"] == direction]
        selected.sort(key=lambda row: float(row["initial_pair_distance_A"]))
        distance = [float(row["initial_pair_distance_A"]) for row in selected]
        energy = [float(row["E_2vac_eV"]) for row in selected]
        if len(selected) > 1:
            axis.plot(distance, energy, "o-", label=direction)
        else:
            axis.scatter(distance, energy, label=direction)
    axis.set(
        xlabel="Initial minimum-image vacancy distance (A)",
        ylabel="Two-vacancy formation energy (eV)",
    )
    axis.legend(title="Direction", frameon=False)
    figure.tight_layout()
    figure.savefig(output / "divacancy_E2vac_by_direction.png", dpi=300)
    figure.savefig(output / "divacancy_E2vac_by_direction.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze relaxed divacancy geometry and strain proxies.")
    parser.add_argument("--rootdir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--neighbor-cutoff", type=float, default=3.3)
    args = parser.parse_args()

    root = Path(args.rootdir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    case_rows, atom_rows = analyze(root, output, args.neighbor_cutoff)
    make_plots(output, case_rows, atom_rows)
    print(f"cases={len(case_rows)}")
    print(f"atoms={len(atom_rows)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
