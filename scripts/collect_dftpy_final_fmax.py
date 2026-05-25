from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


BFGS_RE = re.compile(
    r"^(?P<optimizer>BFGS|LBFGS|FIRE):\s+"
    r"(?P<step>\d+)\s+\S+\s+"
    r"(?P<energy>[-0-9.Ee+]+)\s+"
    r"(?P<fmax>[0-9.Ee+-]+)"
)


def parse_log(path: Path) -> dict[str, object]:
    final: dict[str, object] = {
        "optimizer": "",
        "final_step": math.nan,
        "final_energy_eV_from_log": math.nan,
        "actual_final_fmax_eV_A_from_ASE_log": math.nan,
        "converged_keyword_in_log": False,
    }
    if not path.exists():
        return final

    text = path.read_text(errors="ignore")
    final["converged_keyword_in_log"] = "converged" in text.lower()
    for line in text.splitlines():
        match = BFGS_RE.search(line.strip())
        if not match:
            continue
        final = {
            **final,
            "optimizer": match.group("optimizer"),
            "final_step": int(match.group("step")),
            "final_energy_eV_from_log": float(match.group("energy")),
            "actual_final_fmax_eV_A_from_ASE_log": float(match.group("fmax")),
        }
    return final


def manifest_value(manifest: dict[str, object], key: str, default: object = "") -> object:
    return manifest.get(key, default)


def collect_series(series_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case_dir in sorted(series_root.glob("*_scan/*")):
        if not case_dir.is_dir():
            continue
        manifest_path = case_dir / "point_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        log_path = case_dir / "vacancy_relax.log"
        parsed = parse_log(log_path)
        target = float(manifest_value(manifest, "fmax_eV_per_A", math.nan))
        actual = float(parsed["actual_final_fmax_eV_A_from_ASE_log"])
        rows.append(
            {
                "series": series_root.name,
                "scan": case_dir.parent.name,
                "setting": case_dir.name,
                "case_dir": str(case_dir),
                "cell_basis": manifest_value(manifest, "cell_basis"),
                "conventional_repeat_label": manifest_value(manifest, "conventional_repeat_label"),
                "N_pristine": manifest_value(manifest, "pristine_n_atoms"),
                "N_vacancy": manifest_value(manifest, "vacancy_n_atoms"),
                "spacing_A": manifest_value(manifest, "spacing_A"),
                "optimizer": parsed["optimizer"],
                "final_step": parsed["final_step"],
                "final_energy_eV_from_log": parsed["final_energy_eV_from_log"],
                "actual_final_fmax_eV_A_from_ASE_log": actual,
                "target_fmax_eV_A": target,
                "meets_target_fmax": bool(not math.isnan(actual) and actual <= target),
                "converged_keyword_in_log": parsed["converged_keyword_in_log"],
                "log_path": str(log_path),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect actual final fmax values from DFTpy/ASE vacancy_relax.log files."
    )
    parser.add_argument(
        "roots",
        nargs="+",
        help="One or more DFTpy conventional result roots.",
    )
    parser.add_argument(
        "--out",
        default="dftpy_conventional_actual_final_fmax_summary.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for root_text in args.roots:
        rows.extend(collect_series(Path(root_text).expanduser().resolve()))

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "series",
        "scan",
        "setting",
        "case_dir",
        "cell_basis",
        "conventional_repeat_label",
        "N_pristine",
        "N_vacancy",
        "spacing_A",
        "optimizer",
        "final_step",
        "final_energy_eV_from_log",
        "actual_final_fmax_eV_A_from_ASE_log",
        "target_fmax_eV_A",
        "meets_target_fmax",
        "converged_keyword_in_log",
        "log_path",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {out}")
    print(f"Rows : {len(rows)}")
    for row in rows:
        print(
            row["series"],
            row["scan"],
            row["setting"],
            f"fmax={float(row['actual_final_fmax_eV_A_from_ASE_log']):.6g}",
            f"meets? {row['meets_target_fmax']}",
        )


if __name__ == "__main__":
    main()
