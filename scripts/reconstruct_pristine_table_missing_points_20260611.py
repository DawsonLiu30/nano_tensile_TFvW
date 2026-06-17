#!/usr/bin/env python3
"""Reconstruct pristine-only professor-table values for timed-out vacancy cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

from ase.io import read


MISSING_SETTINGS = (
    "tfvw_lam0p1_mu0p1",
    "tfvw_lam0p1_mu0p2",
    "tfvw_lam0p2_mu0p1",
    "tfvw_lam0p3_mu0p1",
)

MATRIX_FILES = {
    "total": "matrix_pristine_total_energy_eV_per_atom.csv",
    "kedf": "matrix_pristine_kedf_energy_eV_per_atom.csv",
    "lattice": "matrix_pristine_lattice_constant_A.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", required=True, type=Path)
    parser.add_argument("--pp", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    return parser.parse_args()


def read_matrix(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    return rows[0], rows[1:]


def write_matrix(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def matrix_set(
    header: list[str],
    rows: list[list[str]],
    lambda_value: float,
    mu_value: float,
    value: float,
) -> None:
    col = next(
        index
        for index, label in enumerate(header)
        if index > 0 and math.isclose(float(label), mu_value, abs_tol=1e-12)
    )
    row = next(
        item for item in rows if math.isclose(float(item[0]), lambda_value, abs_tol=1e-12)
    )
    row[col] = f"{value:.15g}"


def read_final_log_energy(log_path: Path) -> float:
    final_energy = math.nan
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().startswith(("LBFGS:", "BFGS:")):
            parts = line.split()
            final_energy = float(parts[3])
    if not math.isfinite(final_energy):
        raise RuntimeError(f"No final optimizer energy found in {log_path}")
    return final_energy


def main() -> None:
    args = parse_args()
    rootdir = args.rootdir.resolve()
    outdir = args.outdir.resolve()
    tables_dir = rootdir / "analysis" / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "scf_reconstruction").mkdir(exist_ok=True)

    sys.path.insert(0, str(args.repo.resolve()))
    from app.dft_engine import evaluate_atoms_with_energy_components

    matrices: dict[str, tuple[list[str], list[list[str]]]] = {}
    for key, filename in MATRIX_FILES.items():
        matrices[key] = read_matrix(tables_dir / filename)

    audit_rows: list[dict[str, object]] = []
    for setting in MISSING_SETTINGS:
        case_dir = rootdir / "weight_scan" / setting
        manifest = json.loads((case_dir / "point_manifest.json").read_text(encoding="utf-8"))
        trajectory = case_dir / "pristine_relax_atom_lbfgs.traj"
        atoms = read(str(trajectory), index=-1)
        output_file = outdir / "scf_reconstruction" / f"{setting}_pristine_final_scf.out"

        _, total_energy, _, terms = evaluate_atoms_with_energy_components(
            atoms,
            pp_file=args.pp.resolve(),
            spacing=float(manifest["spacing_A"]),
            kedf=str(manifest["kedf"]),
            xc=str(manifest["xc"]),
            kedf_x=float(manifest["kedf_x"]),
            kedf_y=float(manifest["kedf_y"]),
            dftpy_outfile=str(output_file),
        )

        n_atoms = len(atoms)
        kinetic_energy = float(terms["KEDF"])
        repeat = manifest["conventional_repeat"]
        repeat_count = int(repeat[0]) * int(repeat[1]) * int(repeat[2])
        lattice_constant = (float(atoms.get_volume()) / repeat_count) ** (1.0 / 3.0)
        log_energy = read_final_log_energy(case_dir / "pristine_relax_atom_lbfgs.log")

        values = {
            "total": total_energy / n_atoms,
            "kedf": kinetic_energy / n_atoms,
            "lattice": lattice_constant,
        }
        for key, value in values.items():
            header, rows = matrices[key]
            matrix_set(
                header,
                rows,
                float(manifest["lambda"]),
                float(manifest["mu"]),
                value,
            )

        audit_rows.append(
            {
                "setting": setting,
                "lambda": manifest["lambda"],
                "mu": manifest["mu"],
                "n_atoms": n_atoms,
                "volume_A3": atoms.get_volume(),
                "total_energy_eV_scf": total_energy,
                "total_energy_eV_optimizer_log": log_energy,
                "scf_minus_log_eV": total_energy - log_energy,
                "total_energy_eV_per_atom": values["total"],
                "kedf_energy_eV": kinetic_energy,
                "kedf_energy_eV_per_atom": values["kedf"],
                "lattice_constant_A": lattice_constant,
                "trajectory": str(trajectory),
                "scf_output": str(output_file),
            }
        )
        print(
            f"{setting}: E/atom={values['total']:.9f} eV, "
            f"KEDF/atom={values['kedf']:.9f} eV, a0={lattice_constant:.9f} A"
        )

    output_names = {
        "total": "matrix_total_energy_eV_per_atom_100pt.csv",
        "kedf": "matrix_kedf_energy_eV_per_atom_100pt.csv",
        "lattice": "matrix_lattice_constant_A_100pt.csv",
    }
    for key, filename in output_names.items():
        header, rows = matrices[key]
        write_matrix(outdir / filename, header, rows)

    combined_rows: list[list[str]] = []
    combined_headers: list[str] = []
    ordered = ("total", "kedf", "lattice")
    for panel_index, key in enumerate(ordered):
        header, _ = matrices[key]
        if panel_index:
            combined_headers.append("")
        combined_headers.extend(header)
    combined_rows.append(combined_headers)
    for row_index in range(10):
        merged: list[str] = []
        for panel_index, key in enumerate(ordered):
            _, rows = matrices[key]
            if panel_index:
                merged.append("")
            merged.extend(rows[row_index])
        combined_rows.append(merged)

    combined_path = outdir / "professor_three_panel_100pt.csv"
    with combined_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(combined_rows)

    audit_path = outdir / "pristine_missing_point_reconstruction_audit.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    shutil.copy2(
        tables_dir / "matrix_quality_status.csv",
        outdir / "source_matrix_quality_status_96pt.csv",
    )

    for key, (_, rows) in matrices.items():
        numeric_count = sum(
            1
            for row in rows
            for value in row[1:]
            if value.strip() and value.strip().lower() != "nan"
        )
        if numeric_count != 100:
            raise RuntimeError(f"{key} matrix contains only {numeric_count}/100 values")

    print(f"Wrote complete 100-point tables to: {outdir}")


if __name__ == "__main__":
    main()
