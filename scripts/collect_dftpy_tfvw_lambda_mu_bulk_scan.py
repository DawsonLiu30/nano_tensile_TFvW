from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MATRIX_FIELDS = {
    "total_energy_eV_per_atom": "Total energy (eV/atom)",
    "kinetic_energy_eV_per_atom": "KEDF kinetic energy (eV/atom)",
    "lattice_constant_A": "Equilibrium lattice constant (A)",
    "tf_energy_eV_per_atom": "Weighted TF energy (eV/atom)",
    "vw_energy_eV_per_atom": "Weighted vW energy (eV/atom)",
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def matrix_rows(
    rows: list[dict[str, object]],
    lambda_values: list[float],
    mu_values: list[float],
    field: str,
) -> list[dict[str, object]]:
    lookup = {
        (float(row["lambda_tf"]), float(row["mu_vw"])): row.get(field, math.nan)
        for row in rows
    }
    output = []
    for lambda_tf in lambda_values:
        matrix_row: dict[str, object] = {"lambda/mu": lambda_tf}
        for mu_vw in mu_values:
            matrix_row[f"{mu_vw:.1f}"] = lookup.get((lambda_tf, mu_vw), math.nan)
        output.append(matrix_row)
    return output


def write_three_panel_table(
    path: Path,
    matrices: dict[str, list[dict[str, object]]],
    mu_values: list[float],
) -> None:
    selected = [
        ("Total energy (eV/atom)", matrices["total_energy_eV_per_atom"]),
        ("KEDF energy (eV/atom)", matrices["kinetic_energy_eV_per_atom"]),
        ("Lattice constant (A)", matrices["lattice_constant_A"]),
    ]
    width = len(mu_values) + 1
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        title_row: list[object] = []
        header_row: list[object] = []
        for index, (title, _) in enumerate(selected):
            if index:
                title_row.append("")
                header_row.append("")
            title_row.extend([title] + [""] * (width - 1))
            header_row.extend(["lambda/mu"] + [f"{value:.1f}" for value in mu_values])
        writer.writerow(title_row)
        writer.writerow(header_row)
        for row_index in range(len(selected[0][1])):
            combined: list[object] = []
            for index, (_, matrix) in enumerate(selected):
                if index:
                    combined.append("")
                row = matrix[row_index]
                combined.extend([row["lambda/mu"]] + [row[f"{value:.1f}"] for value in mu_values])
            writer.writerow(combined)


def plot_heatmap(
    path: Path,
    rows: list[dict[str, object]],
    lambda_values: list[float],
    mu_values: list[float],
    field: str,
    title: str,
) -> None:
    lookup = {
        (float(row["lambda_tf"]), float(row["mu_vw"])): float(row.get(field, math.nan))
        for row in rows
    }
    matrix = np.asarray(
        [[lookup.get((lambda_tf, mu_vw), math.nan) for mu_vw in mu_values] for lambda_tf in lambda_values],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(8.2, 6.6), constrained_layout=True)
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(mu_values)), [f"{value:.1f}" for value in mu_values])
    ax.set_yticks(range(len(lambda_values)), [f"{value:.1f}" for value in lambda_values])
    ax.set_xlabel("mu_vW")
    ax.set_ylabel("lambda_TF")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect the DFTpy TF+vW lambda-mu bulk scan.")
    parser.add_argument("--rootdir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rootdir = Path(args.rootdir).expanduser().resolve()
    top_manifest = json.loads((rootdir / "manifest.json").read_text(encoding="utf-8"))
    lambda_values = [float(value) for value in top_manifest["lambda_tf_values"]]
    mu_values = [float(value) for value in top_manifest["mu_vw_values"]]

    rows: list[dict[str, object]] = []
    for setting in (rootdir / "settings_lambda_mu_scan.txt").read_text(encoding="utf-8").splitlines():
        setting = setting.strip()
        if not setting:
            continue
        case_dir = rootdir / "lambda_mu_scan" / setting
        manifest = json.loads((case_dir / "point_manifest.json").read_text(encoding="utf-8"))
        result_path = case_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        rows.append(
            {
                "setting": setting,
                "done": bool(result.get("done", False)),
                "lambda_tf": float(manifest["lambda_tf"]),
                "mu_vw": float(manifest["mu_vw"]),
                "total_energy_eV": result.get("equilibrium_total_energy_eV", math.nan),
                "total_energy_eV_per_atom": result.get(
                    "equilibrium_total_energy_eV_per_atom", math.nan
                ),
                "kinetic_energy_eV": result.get("equilibrium_kinetic_energy_eV", math.nan),
                "kinetic_energy_eV_per_atom": result.get(
                    "equilibrium_kinetic_energy_eV_per_atom", math.nan
                ),
                "tf_energy_eV_per_atom": result.get(
                    "equilibrium_tf_energy_eV_per_atom", math.nan
                ),
                "vw_energy_eV_per_atom": result.get(
                    "equilibrium_vw_energy_eV_per_atom", math.nan
                ),
                "lattice_constant_A": result.get("equilibrium_a0_A", math.nan),
                "bulk_modulus_GPa": result.get("bulk_modulus_GPa", math.nan),
                "fit_method": result.get("fit_method", ""),
                "fit_rmse_eV_per_atom": result.get("fit_rmse_eV_per_atom", math.nan),
                "fit_reliable": result.get("fit_reliable", False),
                "boundary_minimum": result.get("boundary_minimum", False),
                "equilibrium_hydrostatic_stress_GPa": result.get(
                    "equilibrium_hydrostatic_stress_GPa", math.nan
                ),
                "equilibrium_refinement_steps": result.get(
                    "equilibrium_refinement_steps", 0
                ),
                "case_dir": str(case_dir),
            }
        )

    rows.sort(key=lambda row: (float(row["lambda_tf"]), float(row["mu_vw"])))
    tables_dir = rootdir / "tables"
    figures_dir = rootdir / "figures"
    tables_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    write_csv(tables_dir / "lambda_mu_bulk_long_summary.csv", rows)

    matrices = {}
    for field, title in MATRIX_FIELDS.items():
        matrix = matrix_rows(rows, lambda_values, mu_values, field)
        matrices[field] = matrix
        write_csv(tables_dir / f"matrix_{field}.csv", matrix)
        plot_heatmap(
            figures_dir / f"heatmap_{field}.png",
            rows,
            lambda_values,
            mu_values,
            field,
            title,
        )

    write_three_panel_table(
        tables_dir / "professor_three_panel_lambda_mu_table.csv",
        matrices,
        mu_values,
    )

    completed = sum(bool(row["done"]) for row in rows)
    reliable = sum(bool(row["done"]) and bool(row["fit_reliable"]) for row in rows)
    note = [
        "# DFTpy TF+vW lambda-mu bulk scan",
        "",
        "Definition:",
        "",
        "`T_s[n] = lambda_TF T_TF[n] + mu_vW T_vW[n]`",
        "",
        "The two coefficients are independent. No lambda+mu=1 constraint is applied.",
        "",
        f"- Completed points: {completed}/{len(rows)}",
        f"- Reliable interior EOS fits: {reliable}/{len(rows)}",
        "- Main table values are reported per atom.",
        "- The long summary also preserves whole-cell energies and fit diagnostics.",
    ]
    (rootdir / "COLLECTION_NOTE.md").write_text("\n".join(note) + "\n", encoding="utf-8")

    print("============================================================")
    print("DFTpy TF+vW lambda-mu bulk scan collected")
    print("============================================================")
    print(f"Root      : {rootdir}")
    print(f"Completed : {completed}/{len(rows)}")
    print(f"Reliable  : {reliable}/{len(rows)}")
    print(f"Main table: {tables_dir / 'professor_three_panel_lambda_mu_table.csv'}")


if __name__ == "__main__":
    main()
