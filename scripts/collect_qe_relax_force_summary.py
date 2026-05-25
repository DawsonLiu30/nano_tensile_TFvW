from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


RY_BOHR_TO_EV_A = 13.605693122994 / 0.529177210903


TOTAL_FORCE_RE = re.compile(r"Total force\s*=\s*([0-9.Ee+-]+)")
ENERGY_RE = re.compile(r"!\s+total energy\s+=\s+([-0-9.Ee+]+)")
FINAL_ENERGY_RE = re.compile(r"Final energy\s*=\s*([-0-9.Ee+]+)")
def read_text(path: Path) -> str:
    return path.read_text(errors="ignore")


def parse_relax_out(path: Path) -> dict[str, object]:
    text = read_text(path)
    total_forces = [float(m.group(1)) for m in TOTAL_FORCE_RE.finditer(text)]
    final_energy_match = None
    for final_energy_match in FINAL_ENERGY_RE.finditer(text):
        pass
    bang_energies = [float(m.group(1)) for m in ENERGY_RE.finditer(text)]

    final_energy_ry = math.nan
    if final_energy_match is not None:
        final_energy_ry = float(final_energy_match.group(1))
    elif bang_energies:
        final_energy_ry = bang_energies[-1]

    last_total_force_ry_bohr = total_forces[-1] if total_forces else math.nan

    return {
        "job_done": "JOB DONE" in text,
        "bfgs_converged": "bfgs converged" in text.lower(),
        "time_limit": "DUE TO TIME LIMIT" in text or "TIME LIMIT" in text,
        "final_energy_Ry": final_energy_ry,
        "last_total_force_Ry_Bohr": last_total_force_ry_bohr,
        "last_total_force_eV_A": last_total_force_ry_bohr * RY_BOHR_TO_EV_A
        if not math.isnan(last_total_force_ry_bohr)
        else math.nan,
    }


def infer_case(path: Path, root: Path) -> str:
    try:
        return str(path.parent.parent.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path.parent.parent)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect final force diagnostics from QE vacancy relax.out files."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="QE vacancy run root containing ecut_scan/kmesh_scan folders.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Defaults to processed_vacancy_convergence/qe_relax_force_summary.csv.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = (
        Path(args.out).resolve()
        if args.out
        else root / "processed_vacancy_convergence" / "qe_relax_force_summary.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for relax_out in sorted(root.rglob("vacancy_relax/relax.out")):
        row = {
            "case": infer_case(relax_out, root),
            "relax_out": str(relax_out),
        }
        row.update(parse_relax_out(relax_out))
        rows.append(row)

    fieldnames = [
        "case",
        "job_done",
        "bfgs_converged",
        "time_limit",
        "final_energy_Ry",
        "last_total_force_Ry_Bohr",
        "last_total_force_eV_A",
        "relax_out",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("============================================================")
    print("QE relax force collection completed")
    print("============================================================")
    print(f"Root : {root}")
    print(f"Output: {out}")
    print()
    for row in rows:
        status = "DONE" if row["job_done"] else "PENDING"
        print(
            f"{status:7s} {row['case']:45s} "
            f"QE_Total_force={row['last_total_force_eV_A']:.6g} eV/A"
        )


if __name__ == "__main__":
    main()
