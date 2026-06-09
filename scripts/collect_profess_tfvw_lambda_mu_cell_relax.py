from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIELDS = {
    "total_energy_eV_per_atom": "Total energy (eV/atom)",
    "kinetic_energy_eV_per_atom": "KEDF kinetic energy (eV/atom)",
    "lattice_constant_A": "Lattice constant (A)",
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", required=True)
    args = parser.parse_args()
    rootdir = Path(args.rootdir).resolve()
    manifest = json.loads((rootdir / "manifest.json").read_text())
    lambdas = [float(value) for value in manifest["lambda_tf_values"]]
    mus = [float(value) for value in manifest["mu_vw_values"]]
    rows = []
    for setting in (rootdir / "settings_lambda_mu_scan.txt").read_text().splitlines():
        result_path = rootdir / "lambda_mu_scan" / setting / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        rows.append(
            {
                "setting": setting,
                "done": result.get("done", False),
                "status": result.get("status", "MISSING"),
                "calculation_completed": result.get("calculation_completed", False),
                "cell_relax_converged": result.get("cell_relax_converged", False),
                "stable_fcc_equilibrium": result.get(
                    "stable_fcc_equilibrium", False
                ),
                "restart_used": result.get("restart_used", False),
                "restart_reason": result.get("restart_reason", ""),
                "original_attempt_status": result.get(
                    "original_attempt_status", ""
                ),
                "lambda_tf": result.get("lambda_tf", math.nan),
                "mu_vw": result.get("mu_vw", math.nan),
                "total_energy_eV_per_atom": result.get("total_energy_eV_per_atom", math.nan),
                "kinetic_energy_eV_per_atom": result.get(
                    "kinetic_energy_eV_per_atom", math.nan
                ),
                "lattice_constant_A": result.get("lattice_constant_A", math.nan),
                "final_max_abs_stress_GPa": result.get(
                    "final_max_abs_stress_GPa", math.nan
                ),
                "case_dir": str(result_path.parent),
            }
        )
    tables = rootdir / "tables"
    figures = rootdir / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    write_csv(tables / "profess_lambda_mu_long_summary.csv", rows)

    lookup = {(float(r["lambda_tf"]), float(r["mu_vw"])): r for r in rows if r["done"]}
    matrices = {}
    for field, title in FIELDS.items():
        matrix_rows = []
        array = []
        for lambda_tf in lambdas:
            row = {"lambda/mu": lambda_tf}
            values = []
            for mu_vw in mus:
                value = lookup.get((lambda_tf, mu_vw), {}).get(field, math.nan)
                row[f"{mu_vw:.1f}"] = value
                values.append(float(value))
            matrix_rows.append(row)
            array.append(values)
        matrices[field] = matrix_rows
        write_csv(tables / f"matrix_{field}.csv", matrix_rows)
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        image = ax.imshow(np.asarray(array), origin="lower", aspect="auto")
        ax.set_xticks(range(len(mus)), [f"{value:.1f}" for value in mus])
        ax.set_yticks(range(len(lambdas)), [f"{value:.1f}" for value in lambdas])
        ax.set_xlabel("mu_vW")
        ax.set_ylabel("lambda_TF")
        ax.set_title(f"PROFESS: {title}")
        fig.colorbar(image, ax=ax)
        fig.savefig(figures / f"heatmap_{field}.png", dpi=300)
        plt.close(fig)

    panel = tables / "professor_three_panel_lambda_mu_table.csv"
    with panel.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["Total energy (eV/atom)"]
            + [""] * 10
            + [""]
            + ["KEDF energy (eV/atom)"]
            + [""] * 10
            + [""]
            + ["Lattice constant (A)"]
            + [""] * 10
        )
        header = ["lambda/mu"] + [f"{value:.1f}" for value in mus]
        writer.writerow(header + [""] + header + [""] + header)
        for index in range(len(lambdas)):
            combined = []
            for field in FIELDS:
                row = matrices[field][index]
                combined.extend([row["lambda/mu"]] + [row[f"{mu:.1f}"] for mu in mus])
                if field != list(FIELDS)[-1]:
                    combined.append("")
            writer.writerow(combined)
    completed = sum(bool(row["done"]) for row in rows)
    calculations = sum(bool(row["calculation_completed"]) for row in rows)
    print(f"Completed PROFESS calculations: {calculations}/{len(rows)}")
    print(f"Stable fcc equilibria         : {completed}/{len(rows)}")
    print(panel)


if __name__ == "__main__":
    main()
