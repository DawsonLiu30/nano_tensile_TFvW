from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from ase.io import read


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _latest_result_dir(case_dir: Path) -> Path:
    candidates = sorted(
        [p for p in (case_dir / "results").glob("*") if (p / "summary.csv").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No result folder with summary.csv found under {case_dir / 'results'}")
    return candidates[0]


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _case_dir_from_results(result_dir: Path) -> Path | None:
    try:
        if result_dir.parent.name == "results":
            return result_dir.parent.parent
    except IndexError:
        return None
    return None


def _d111_from_case(case_dir: Path | None, fallback_a0: float | None) -> float:
    if fallback_a0 is not None:
        return float(fallback_a0) / math.sqrt(3.0)
    if case_dir is not None:
        manifest = _read_json(case_dir / "inputs" / "grip_vacancy_manifest.json")
        a0 = manifest.get("geometry", {}).get("a0_input_A")
        if a0 is not None:
            return float(a0) / math.sqrt(3.0)
    raise ValueError("Could not infer a0. Pass --a0 explicitly.")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze finite-grip tensile fracture using a z-gap criterion.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--case-dir", help="Case directory; latest results/*/summary.csv is used.")
    src.add_argument("--results-dir", help="Specific results directory containing summary.csv.")
    ap.add_argument("--a0", type=float, default=None, help="Lattice constant in Angstrom. Defaults to manifest a0.")
    ap.add_argument("--gap-factor", type=float, default=3.0, help="Fracture threshold multiplier, default 3*d111.")
    ap.add_argument("--out", default="", help="Output CSV path. Defaults beside summary.csv.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.case_dir:
        case_dir = _resolve(args.case_dir)
        result_dir = _latest_result_dir(case_dir)
    else:
        result_dir = _resolve(args.results_dir)
        case_dir = _case_dir_from_results(result_dir)

    summary = result_dir / "summary.csv"
    if not summary.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary}")

    metadata = _read_json(case_dir / "inputs" / "grip_metadata.json") if case_dir else {}
    fixed = np.asarray(metadata.get("fixed_indices", []), dtype=int)
    d111 = _d111_from_case(case_dir, args.a0)
    threshold = float(args.gap_factor) * d111
    out = _resolve(args.out) if args.out else result_dir / "fracture_analysis.csv"

    with open(summary, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    analysis_rows: list[dict[str, object]] = []
    first_fracture = None
    for row in rows:
        cycle = int(row["cycle"])
        xyz = result_dir / f"cycle_{cycle:03d}_relaxed.xyz"
        if not xyz.exists():
            continue
        atoms = read(str(xyz))
        z = atoms.get_positions()[:, 2]
        all_gap = float(np.max(np.diff(np.sort(z)))) if len(z) > 1 else 0.0
        if fixed.size:
            free = np.setdiff1d(np.arange(len(atoms)), fixed)
            z_free = np.sort(z[free])
        else:
            z_free = np.sort(z)
        free_gap = float(np.max(np.diff(z_free))) if len(z_free) > 1 else 0.0
        free_span = float(z_free.max() - z_free.min()) if len(z_free) else 0.0
        fractured = bool(free_gap > threshold)
        if fractured and first_fracture is None:
            first_fracture = cycle
        analysis_rows.append(
            {
                "cycle": cycle,
                "strain": f"{float(row['strain']):.12f}",
                "strain_pct": f"{float(row['strain']) * 100.0:.6f}",
                "grip_stress_avg_GPa": f"{float(row['grip_stress_avg_GPa']):.12f}",
                "max_gap_all_A": f"{all_gap:.12f}",
                "max_gap_free_A": f"{free_gap:.12f}",
                "free_z_span_A": f"{free_span:.12f}",
                "d111_A": f"{d111:.12f}",
                "gap_threshold_A": f"{threshold:.12f}",
                "fracture_by_gap": str(fractured),
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cycle",
                "strain",
                "strain_pct",
                "grip_stress_avg_GPa",
                "max_gap_all_A",
                "max_gap_free_A",
                "free_z_span_A",
                "d111_A",
                "gap_threshold_A",
                "fracture_by_gap",
            ],
        )
        writer.writeheader()
        writer.writerows(analysis_rows)

    print(f"[fracture] summary      : {summary}")
    print(f"[fracture] output       : {out}")
    print(f"[fracture] d111         : {d111:.6f} A")
    print(f"[fracture] threshold    : {threshold:.6f} A ({float(args.gap_factor):.2f} * d111)")
    if first_fracture is None:
        print("[fracture] result       : no fracture by z-gap criterion")
    else:
        print(f"[fracture] result       : first fracture at cycle {first_fracture}")


if __name__ == "__main__":
    main()
