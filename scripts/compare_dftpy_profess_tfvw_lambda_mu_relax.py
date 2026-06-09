from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_rows(path: Path) -> dict[tuple[float, float], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (float(row["lambda_tf"]), float(row["mu_vw"])): row
            for row in csv.DictReader(handle)
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dftpy-root", required=True)
    parser.add_argument("--profess-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    dftpy = read_rows(Path(args.dftpy_root) / "tables" / "lambda_mu_bulk_long_summary.csv")
    profess = read_rows(
        Path(args.profess_root) / "tables" / "profess_lambda_mu_long_summary.csv"
    )
    rows = []
    for key in sorted(set(dftpy) | set(profess)):
        d = dftpy.get(key, {})
        p = profess.get(key, {})
        d_stable = d.get("done", "").lower() == "true"
        p_stable = p.get("done", "").lower() == "true"
        row = {
            "lambda_tf": key[0],
            "mu_vw": key[1],
            "dftpy_status": d.get("status", "MISSING"),
            "profess_status": p.get("status", "MISSING"),
            "dftpy_stable_fcc": d_stable,
            "profess_stable_fcc": p_stable,
            "both_stable_fcc": d_stable and p_stable,
        }
        for field in (
            "total_energy_eV_per_atom",
            "kinetic_energy_eV_per_atom",
            "lattice_constant_A",
        ):
            dv = float(d.get(field, math.nan))
            pv = float(p.get(field, math.nan))
            row[f"dftpy_{field}"] = dv
            row[f"profess_{field}"] = pv
            row[f"profess_minus_dftpy_{field}"] = (
                pv - dv if d_stable and p_stable else math.nan
            )
        rows.append(row)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
