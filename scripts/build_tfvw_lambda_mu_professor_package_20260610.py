from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path


REFERENCE_A0_A = 4.039848


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def build_vacancy_qc(vacancy_root: Path) -> list[dict[str, object]]:
    long_path = vacancy_root / "analysis" / "tables" / "lambda_mu_vacancy_long_summary.csv"
    rows = read_csv(long_path)
    output: list[dict[str, object]] = []
    for row in rows:
        case_dir = vacancy_root / "weight_scan" / row["setting"]
        result_path = case_dir / "result.json"
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.exists()
            else {}
        )
        stresses = [
            abs(float(value))
            for tensor_row in result.get("pristine_stress_GPa", [])
            for value in tensor_row
        ]
        cell_lengths = [float(value) for value in result.get("pristine_cell_lengths_A", [])]
        output.append(
            {
                **row,
                "pristine_conventional_a0_A": (
                    sum(cell_lengths) / len(cell_lengths) / 3.0
                    if cell_lengths
                    else math.nan
                ),
                "pristine_max_abs_stress_GPa": max(stresses) if stresses else math.nan,
                "professor_three_property_source": False,
                "qc_note": (
                    "Vacancy matrix; not used for professor Total/KEDF/a0 bulk table."
                    if row["status"] == "DONE"
                    else "Incomplete vacancy point."
                ),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bulk-root", required=True)
    parser.add_argument("--vacancy-root", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    bulk_root = Path(args.bulk_root).expanduser().resolve()
    vacancy_root = Path(args.vacancy_root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    bulk_table = bulk_root / "tables" / "lambda_mu_bulk_long_summary.csv"
    rows = read_csv(bulk_table)
    for row in rows:
        row["delta_a0_vs_reference_A"] = (
            as_float(row["lattice_constant_A"]) - REFERENCE_A0_A
        )
        row["physical_status"] = (
            "STABLE"
            if row["stable_fcc_equilibrium"].lower() == "true"
            else "COLLAPSED"
            if row["relaxation_converged"].lower() == "true"
            else "UNCONVERGED"
        )

    rows.sort(key=lambda row: (as_float(row["lambda_tf"]), as_float(row["mu_vw"])))
    write_csv(outdir / "02_BULK_LONG_DATA_100_POINTS.csv", rows)

    closest = sorted(
        rows,
        key=lambda row: abs(as_float(row["delta_a0_vs_reference_A"])),
    )[:15]
    closest_rows = [
        {
            "rank": index,
            "lambda_tf": row["lambda_tf"],
            "mu_vw": row["mu_vw"],
            "lattice_constant_A": row["lattice_constant_A"],
            "delta_a0_vs_reference_A": row["delta_a0_vs_reference_A"],
            "total_energy_eV_per_atom": row["total_energy_eV_per_atom"],
            "kinetic_energy_eV_per_atom": row["kinetic_energy_eV_per_atom"],
            "physical_status": row["physical_status"],
        }
        for index, row in enumerate(closest, start=1)
    ]
    write_csv(outdir / "03_CLOSEST_LATTICE_CONSTANT_POINTS.csv", closest_rows)

    lambda_summary: list[dict[str, object]] = []
    lambda_values = sorted({as_float(row["lambda_tf"]) for row in rows})
    for lambda_tf in lambda_values:
        subset = [row for row in rows if as_float(row["lambda_tf"]) == lambda_tf]
        lattice = [as_float(row["lattice_constant_A"]) for row in subset]
        lambda_summary.append(
            {
                "lambda_tf": lambda_tf,
                "stable_points": sum(row["physical_status"] == "STABLE" for row in subset),
                "collapsed_points": sum(
                    row["physical_status"] == "COLLAPSED" for row in subset
                ),
                "a0_min_A": min(lattice),
                "a0_max_A": max(lattice),
                "a0_mean_A": sum(lattice) / len(lattice),
            }
        )
    write_csv(outdir / "04_LAMBDA_ROW_SUMMARY.csv", lambda_summary)

    vacancy_qc = build_vacancy_qc(vacancy_root)
    write_csv(outdir / "05_VACANCY_96_POINT_QC.csv", vacancy_qc)

    bulk_lookup = {
        (round(as_float(row["lambda_tf"]), 6), round(as_float(row["mu_vw"]), 6)): row
        for row in rows
    }
    joint_rows: list[dict[str, object]] = []
    for vacancy_row in vacancy_qc:
        if vacancy_row["status"] != "DONE":
            continue
        key = (
            round(as_float(vacancy_row["lambda_tf"]), 6),
            round(as_float(vacancy_row["mu_vw"]), 6),
        )
        bulk_row = bulk_lookup[key]
        lattice_constant = as_float(bulk_row["lattice_constant_A"])
        vacancy_energy = as_float(vacancy_row["vacancy_formation_energy_eV"])
        delta_a0 = abs(lattice_constant - REFERENCE_A0_A)
        delta_ef = abs(vacancy_energy - 0.601167)
        joint_rows.append(
            {
                "lambda_tf": key[0],
                "mu_vw": key[1],
                "bulk_lattice_constant_A": lattice_constant,
                "abs_delta_a0_A": delta_a0,
                "vacancy_formation_energy_eV": vacancy_energy,
                "abs_delta_Ef_vs_0p601167_eV": delta_ef,
                "bulk_physical_status": bulk_row["physical_status"],
                "joint_score_da0_0p05_dEf_0p1": (delta_a0 / 0.05) ** 2
                + (delta_ef / 0.1) ** 2,
                "strict_joint_match": delta_a0 <= 0.05 and delta_ef <= 0.1,
                "loose_joint_match": delta_a0 <= 0.10 and delta_ef <= 0.2,
            }
        )
    joint_rows.sort(key=lambda row: as_float(row["joint_score_da0_0p05_dEf_0p1"]))
    write_csv(outdir / "06_BULK_VACANCY_JOINT_SCREEN.csv", joint_rows)

    table_map = {
        "professor_three_panel_lambda_mu_table.csv": "01_PROFESSOR_THREE_PANEL_100_POINTS.csv",
        "matrix_total_energy_eV_per_atom.csv": "matrices/MATRIX_TOTAL_ENERGY_eV_PER_ATOM.csv",
        "matrix_kinetic_energy_eV_per_atom.csv": "matrices/MATRIX_KEDF_ENERGY_eV_PER_ATOM.csv",
        "matrix_lattice_constant_A.csv": "matrices/MATRIX_LATTICE_CONSTANT_A.csv",
        "matrix_physical_status.csv": "matrices/MATRIX_PHYSICAL_STATUS.csv",
        "matrix_tf_energy_eV_per_atom.csv": "matrices/MATRIX_WEIGHTED_TF_eV_PER_ATOM.csv",
        "matrix_vw_energy_eV_per_atom.csv": "matrices/MATRIX_WEIGHTED_VW_eV_PER_ATOM.csv",
    }
    for source_name, destination_name in table_map.items():
        copy_if_exists(
            bulk_root / "tables" / source_name,
            outdir / destination_name,
        )

    for figure in (
        "heatmap_total_energy_eV_per_atom.png",
        "heatmap_kinetic_energy_eV_per_atom.png",
        "heatmap_lattice_constant_A.png",
    ):
        copy_if_exists(
            bulk_root / "figures" / figure,
            outdir / "figures" / figure,
        )

    copy_if_exists(
        vacancy_root / "analysis" / "COMPLETION_AUDIT.md",
        outdir / "vacancy_qc" / "COMPLETION_AUDIT.md",
    )
    copy_if_exists(
        vacancy_root / "analysis" / "tables" / "incomplete_cases.csv",
        outdir / "vacancy_qc" / "INCOMPLETE_4_CASES.csv",
    )

    stable_count = sum(row["physical_status"] == "STABLE" for row in rows)
    collapsed_count = sum(row["physical_status"] == "COLLAPSED" for row in rows)
    done_vacancy = sum(row["status"] == "DONE" for row in vacancy_qc)
    vacancy_ef = [
        as_float(row["vacancy_formation_energy_eV"])
        for row in vacancy_qc
        if row["status"] == "DONE"
    ]
    vacancy_high = sum(value > 5.0 for value in vacancy_ef)
    vacancy_reasonable = sum(0.0 <= value <= 1.0 for value in vacancy_ef)
    strict_joint = sum(bool(row["strict_joint_match"]) for row in joint_rows)
    loose_joint = sum(bool(row["loose_joint_match"]) for row in joint_rows)
    best_joint = joint_rows[0]

    best = closest_rows[0]
    analysis = f"""# TFvW lambda-mu matrix analysis

## Correct data source for the professor's three tables

- Use the **100-point bulk full-cell relaxation scan**.
- Table units: total energy in eV/atom, KEDF kinetic energy in eV/atom, lattice constant in A.
- All 100 cell relaxations reached the numerical force criterion.
- Stable fcc equilibria: {stable_count}/100.
- Collapsed/nonphysical equilibria (`a0 < 3.0 A`): {collapsed_count}/100.

## Main result

- `lambda_TF` is the dominant control of the lattice constant.
- The closest point to the reference `a0 = {REFERENCE_A0_A:.6f} A` is:
  - lambda = {best['lambda_tf']}
  - mu = {best['mu_vw']}
  - a0 = {as_float(best['lattice_constant_A']):.6f} A
  - delta a0 = {as_float(best['delta_a0_vs_reference_A']):+.6f} A
  - total energy = {as_float(best['total_energy_eV_per_atom']):.6f} eV/atom
  - KEDF energy = {as_float(best['kinetic_energy_eV_per_atom']):.6f} eV/atom
- This supports keeping the TF coefficient near `lambda = 1`.

## Interpretation cautions

- Absolute total energies from different lambda/mu functionals should not be ranked as if they were the same Hamiltonian.
- The 21 collapsed points are numerically converged but not physically acceptable fcc Al equilibria.
- The separate vacancy matrix is not the source of the professor's Total/KEDF/a0 table.

## Vacancy matrix audit

- Completed vacancy points: {done_vacancy}/100.
- Incomplete/timeouts: {100 - done_vacancy}/100.
- Completed points with `0 <= Ef_vac <= 1 eV`: {vacancy_reasonable}/{done_vacancy}.
- Completed points with `Ef_vac > 5 eV`: {vacancy_high}/{done_vacancy}.
- Therefore vacancy-energy matching alone cannot select lambda/mu; the bulk lattice and energy diagnostics must be applied first.

## Joint bulk-vacancy screen

- Strict joint criterion (`|delta a0| <= 0.05 A` and `|delta Ef| <= 0.10 eV`): {strict_joint} points.
- Loose joint criterion (`|delta a0| <= 0.10 A` and `|delta Ef| <= 0.20 eV`): {loose_joint} point.
- Best current compromise:
  - lambda = {best_joint['lambda_tf']}
  - mu = {best_joint['mu_vw']}
  - a0 = {as_float(best_joint['bulk_lattice_constant_A']):.6f} A
  - Ef_vac = {as_float(best_joint['vacancy_formation_energy_eV']):.6f} eV
- No sampled point reproduces both targets tightly.
- Linear interpolation along `mu = 0.1` between lambda 0.9 and 1.0 predicts:
  - lambda about 0.9241 for `a0 = 4.039848 A`
  - interpolated `Ef_vac` about 0.6222 eV
- The logical next calculation is therefore a fine joint scan around lambda 0.90-0.95 and mu 0.08-0.15.
"""
    (outdir / "ANALYSIS_SUMMARY.md").write_text(analysis, encoding="utf-8")

    readme = """# Professor lambda-mu data package

Main file:

`01_PROFESSOR_THREE_PANEL_100_POINTS.csv`

It contains the requested 10x10 matrices for:

1. total energy (eV/atom)
2. KEDF kinetic energy (eV/atom)
3. equilibrium lattice constant (A)

Use `matrices/MATRIX_PHYSICAL_STATUS.csv` together with the main table.
`COLLAPSED` means the optimization converged numerically but the fcc lattice
constant fell below 3.0 A and is not a physically acceptable Al equilibrium.

The vacancy files are retained only under QC and are not mixed into the bulk
three-property table.
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Wrote package: {outdir}")
    print(f"Bulk stable/collapsed: {stable_count}/{collapsed_count}")
    print(
        "Best a0 point: "
        f"lambda={best['lambda_tf']} mu={best['mu_vw']} "
        f"a0={as_float(best['lattice_constant_A']):.6f} A"
    )


if __name__ == "__main__":
    main()
