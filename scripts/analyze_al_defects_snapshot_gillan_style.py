#!/usr/bin/env python3
"""Analyze a pulled Al_defects snapshot in Gillan-style vacancy notation.

The script is intentionally local/offline: pull the focused iservice snapshot
first, then run this against the local `dftpy45_snapshot` folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_json_error": repr(exc)}


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "ok"}
    return False


def classify_row(d: dict[str, Any]) -> str:
    if d.get("_json_error"):
        return "bad_result_json"

    # Prefer explicit qualification written by production/collector scripts.
    for key in ("qualified", "quality_qualified", "is_qualified"):
        if key in d:
            return "qualified" if is_truthy(d.get(key)) else "unqualified"

    pf = as_float(d.get("pristine_final_fmax_eV_A"))
    vf = as_float(d.get("vacancy_final_fmax_eV_A"))
    target = as_float(d.get("fmax_target_eV_A") or d.get("fmax_eV_A")) or 0.01
    if pf is not None and vf is not None:
        return "qualified" if pf <= target and vf <= target else "unqualified"

    ef = as_float(d.get("vacancy_formation_energy_eV"))
    if ef is not None:
        return "completed_unclassified"
    return "incomplete_result"


def latest_status(case_dir: Path) -> dict[str, Any]:
    paths = sorted(case_dir.glob("run_status_job*_task*.json"))
    if not paths:
        return {}
    return read_json(paths[-1])


def detect_matrix_root(snapshot_root: Path) -> Path:
    candidates = list(
        snapshot_root.glob(
            "results/Al_defects/01_calibration/single_vacancy/"
            "dftpy_tfvw_lambda_mu/coarse_10x10_vacancy_formation"
        )
    )
    if candidates:
        return candidates[0]
    matches = list(snapshot_root.rglob("coarse_10x10_vacancy_formation"))
    if not matches:
        raise FileNotFoundError(
            "Cannot find coarse_10x10_vacancy_formation under snapshot root"
        )
    return matches[0]


def collect_rows(snapshot_root: Path, matrix_root: Path) -> list[dict[str, Any]]:
    settings_file = matrix_root / "01_settings" / "settings_lambda_mu_10x10.txt"
    settings: list[tuple[str, str, str]] = []
    if settings_file.exists():
        for line in settings_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            setting = parts[0]
            lam = parts[1] if len(parts) > 1 else ""
            mu = parts[2] if len(parts) > 2 else ""
            settings.append((setting, lam, mu))
    else:
        for case_dir in sorted((matrix_root / "03_runs").glob("tfvw_*")):
            manifest = read_json(case_dir / "point_manifest.json")
            settings.append(
                (
                    case_dir.name,
                    str(manifest.get("lambda", manifest.get("kedf_x", ""))),
                    str(manifest.get("mu", manifest.get("kedf_y", ""))),
                )
            )

    rows: list[dict[str, Any]] = []
    for index, (setting, lam, mu) in enumerate(settings):
        case_dir = matrix_root / "03_runs" / setting
        result_path = case_dir / "result.json"
        manifest = read_json(case_dir / "point_manifest.json")
        data = read_json(result_path) if result_path.exists() else {}
        status_data = latest_status(case_dir)
        status_from_result = classify_row(data) if result_path.exists() else ""
        if result_path.exists():
            status = status_from_result
        elif status_data.get("status"):
            status = str(status_data.get("status"))
        elif (case_dir / "RUNNING_LOCK").exists():
            status = "running_or_stale_lock"
        else:
            status = "no_result_json"

        row: dict[str, Any] = {
            "index": index,
            "setting": setting,
            "lambda": data.get("lambda", data.get("kedf_x", manifest.get("lambda", lam))),
            "mu": data.get("mu", data.get("kedf_y", manifest.get("mu", mu))),
            "status": status,
            "result_status": status_from_result,
            "run_status_status": status_data.get("status", ""),
            "run_status_rc": status_data.get("python_or_timeout_rc", ""),
            "run_status_has_result_json": status_data.get("has_result_json", ""),
            "has_result_json": result_path.exists(),
            "source_dir": str(case_dir),
            "source_dir_relative": str(case_dir.relative_to(snapshot_root)),
            "result_json": str(result_path) if result_path.exists() else "",
            "pristine_input": str(case_dir / "dftpy_pristine_input.ini"),
            "vacancy_input": str(case_dir / "dftpy_vacancy_input.ini"),
            "pristine_structure": str(case_dir / "pristine_raw.vasp"),
            "vacancy_structure": str(case_dir / "vacancy_start.vasp"),
            "pristine_relax_log": str(case_dir / "pristine_relax.log"),
            "vacancy_relax_log": str(case_dir / "vacancy_relax.log"),
            "pristine_n_atoms": data.get("pristine_n_atoms", manifest.get("pristine_n_atoms")),
            "vacancy_n_atoms": data.get("vacancy_n_atoms", manifest.get("vacancy_n_atoms")),
            "pristine_energy_eV": data.get("pristine_energy_eV"),
            "vacancy_energy_eV": data.get("vacancy_energy_eV"),
            "vacancy_formation_energy_eV": data.get("vacancy_formation_energy_eV"),
            "pristine_kedf_energy_eV": data.get("pristine_kedf_energy_eV"),
            "vacancy_kedf_energy_eV": data.get("vacancy_kedf_energy_eV"),
            "vacancy_formation_kedf_energy_eV": data.get(
                "vacancy_formation_kedf_energy_eV"
            ),
            "lattice_constant_A": data.get("lattice_constant_A"),
            "pristine_final_fmax_eV_A": data.get("pristine_final_fmax_eV_A"),
            "vacancy_final_fmax_eV_A": data.get("vacancy_final_fmax_eV_A"),
            "fmax_target_eV_A": data.get(
                "fmax_target_eV_A", data.get("fmax_eV_A", manifest.get("fmax_eV_per_A"))
            ),
            "formula_total": data.get(
                "formation_energy_formula_total",
                manifest.get(
                    "formation_energy_formula_total",
                    "E_f^vac = E_vac(Al107) - (107/108) E_pristine(Al108)",
                ),
            ),
            "formula_kedf": data.get(
                "formation_energy_formula_kedf",
                manifest.get(
                    "formation_energy_formula_kedf",
                    "KEDF_f^vac = KEDF_vac(Al107) - (107/108) KEDF_pristine(Al108)",
                ),
            ),
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(
    path: Path, rows: list[dict[str, Any]], value_key: str, qualified_only: bool
) -> None:
    filtered = [
        r
        for r in rows
        if (not qualified_only or r.get("status") == "qualified")
        and as_float(r.get(value_key)) is not None
    ]
    lambdas = sorted({as_float(r.get("lambda")) for r in filtered if as_float(r.get("lambda")) is not None})
    mus = sorted({as_float(r.get("mu")) for r in filtered if as_float(r.get("mu")) is not None})
    lookup = {
        (as_float(r.get("lambda")), as_float(r.get("mu"))): as_float(r.get(value_key))
        for r in filtered
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lambda/mu", *[f"{m:g}" for m in mus]])
        for lam in lambdas:
            writer.writerow(
                [f"{lam:g}", *["" if lookup.get((lam, mu)) is None else lookup[(lam, mu)] for mu in mus]]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-root",
        required=True,
        type=Path,
        help="Local dftpy45_snapshot folder produced by pull_iservice_al_defects_snapshot.sh",
    )
    parser.add_argument("--matrix-root", type=Path)
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()

    snapshot_root = args.snapshot_root.resolve()
    matrix_root = args.matrix_root.resolve() if args.matrix_root else detect_matrix_root(snapshot_root)
    outdir = args.outdir or snapshot_root / "_analysis_gillan_style"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(snapshot_root, matrix_root)
    counts = Counter(r["status"] for r in rows)
    has_result = sum(1 for r in rows if r["has_result_json"])
    qualified = sum(1 for r in rows if r["status"] == "qualified")

    fieldnames = [
        "index",
        "setting",
        "lambda",
        "mu",
        "status",
        "result_status",
        "run_status_status",
        "run_status_rc",
        "run_status_has_result_json",
        "has_result_json",
        "pristine_n_atoms",
        "vacancy_n_atoms",
        "pristine_energy_eV",
        "vacancy_energy_eV",
        "vacancy_formation_energy_eV",
        "pristine_kedf_energy_eV",
        "vacancy_kedf_energy_eV",
        "vacancy_formation_kedf_energy_eV",
        "lattice_constant_A",
        "pristine_final_fmax_eV_A",
        "vacancy_final_fmax_eV_A",
        "fmax_target_eV_A",
        "formula_total",
        "formula_kedf",
        "source_dir",
        "source_dir_relative",
        "result_json",
        "pristine_input",
        "vacancy_input",
        "pristine_structure",
        "vacancy_structure",
        "pristine_relax_log",
        "vacancy_relax_log",
    ]
    write_csv(outdir / "vacancy_lambda_mu_gillan_long_summary.csv", rows, fieldnames)

    professor_rows = [
        {
            "lambda": r["lambda"],
            "mu": r["mu"],
            "status": r["status"],
            "vacancy_formation_energy_eV": r["vacancy_formation_energy_eV"],
            "vacancy_formation_kedf_energy_eV": r["vacancy_formation_kedf_energy_eV"],
            "lattice_constant_A": r["lattice_constant_A"],
            "source_dir_relative": r["source_dir_relative"],
        }
        for r in rows
    ]
    write_csv(
        outdir / "professor_three_metrics_long.csv",
        professor_rows,
        [
            "lambda",
            "mu",
            "status",
            "vacancy_formation_energy_eV",
            "vacancy_formation_kedf_energy_eV",
            "lattice_constant_A",
            "source_dir_relative",
        ],
    )

    for qualified_only in (False, True):
        suffix = "qualified_only" if qualified_only else "all_completed"
        write_matrix(
            outdir / f"matrix_vacancy_formation_energy_eV_{suffix}.csv",
            rows,
            "vacancy_formation_energy_eV",
            qualified_only,
        )
        write_matrix(
            outdir / f"matrix_kedf_formation_energy_eV_{suffix}.csv",
            rows,
            "vacancy_formation_kedf_energy_eV",
            qualified_only,
        )
        write_matrix(
            outdir / f"matrix_lattice_constant_A_{suffix}.csv",
            rows,
            "lattice_constant_A",
            qualified_only,
        )

    report = [
        "# Gillan-Style Vacancy Lambda-Mu Snapshot Analysis",
        "",
        f"snapshot_root: `{snapshot_root}`",
        f"matrix_root: `{matrix_root}`",
        "",
        "## Counts",
        "",
        f"- total points: {len(rows)}",
        f"- has result.json: {has_result}",
        f"- qualified: {qualified}",
        f"- not qualified / incomplete: {len(rows) - qualified}",
        "",
        "## Status Counts",
        "",
    ]
    report.extend(f"- {k}: {v}" for k, v in sorted(counts.items()))
    report.extend(
        [
            "",
            "## Formula Check",
            "",
            "- Perfect system is already included as `pristine_raw.vasp` / Al108.",
            "- Defective system is already included as `vacancy_start.vasp` / Al107.",
            "- Main total-energy formula: `E_f^vac = E_vac(Al107) - (107/108) E_pristine(Al108)`.",
            "- Main KEDF formula: `KEDF_f^vac = KEDF_vac(Al107) - (107/108) KEDF_pristine(Al108)`.",
            "- Lattice constant is the relaxed pristine 3x3x3 cell length divided by 3.",
            "",
            "## Outputs",
            "",
            "- `vacancy_lambda_mu_gillan_long_summary.csv`",
            "- `professor_three_metrics_long.csv`",
            "- `matrix_vacancy_formation_energy_eV_all_completed.csv`",
            "- `matrix_kedf_formation_energy_eV_all_completed.csv`",
            "- `matrix_lattice_constant_A_all_completed.csv`",
            "- `matrix_*_qualified_only.csv` variants",
        ]
    )
    (outdir / "GILLAN_STYLE_ANALYSIS.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "total_points": len(rows),
                "has_result_json": has_result,
                "qualified": qualified,
                "status_counts": dict(counts),
                "outdir": str(outdir),
                "formula_total": "E_f^vac = E_vac(Al107) - (107/108) E_pristine(Al108)",
                "formula_kedf": "KEDF_f^vac = KEDF_vac(Al107) - (107/108) KEDF_pristine(Al108)",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
