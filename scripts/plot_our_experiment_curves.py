#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.geometry_metrics import projected_xy_geometry

DEFAULT_OUT = ROOT / "results" / "our_experiment_strain_stress_curve.png"
DEFAULT_TABLE = ROOT / "results" / "our_experiment_curve_summary.csv"

ORIENTATION_COLORS = {
    "111": "#1f77b4",
    "100": "#d95f02",
    "110": "#1b9e77",
}

DIAMETER_STYLES = {
    1.0: "-",
    2.0: "--",
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot strain-stress curves for our finite-length nanocrystal experiments."
    )
    ap.add_argument("--pattern", default="paperfinite_*", help="Case directory glob under cases/")
    ap.add_argument("--include-smoke", action="store_true", help="Include *_smoke_* runs")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output PNG path")
    ap.add_argument("--table", default=str(DEFAULT_TABLE), help="Output CSV summary path")
    return ap.parse_args()


def _read_manifest(case_dir: Path) -> dict:
    manifest_path = case_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_summary_rows(summary_csv: Path) -> list[dict[str, str]]:
    with summary_csv.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _pick_stress_column(rows: list[dict[str, str]]) -> str | None:
    for key in ("eng_stress_top_GPa", "sigma_zz_GPa"):
        for row in rows:
            value = (row.get(key) or "").strip()
            if value:
                return key
    return None


def _series_from_rows(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, str] | None:
    stress_key = _pick_stress_column(rows)
    if stress_key is None:
        return None

    strain_vals: list[float] = []
    stress_vals: list[float] = []
    for row in rows:
        strain_raw = (row.get("strain") or "").strip()
        stress_raw = (row.get(stress_key) or "").strip()
        if not strain_raw or not stress_raw:
            continue
        try:
            strain_vals.append(float(strain_raw))
            stress_vals.append(float(stress_raw))
        except ValueError:
            continue

    if not strain_vals:
        return None

    strain = np.asarray(strain_vals, dtype=float)
    stress = np.asarray(stress_vals, dtype=float)
    order = np.argsort(strain)
    strain = strain[order]
    stress = stress[order]

    # Grip force signs can differ by orientation; plot the tensile branch as positive.
    if np.mean(stress < 0.0) > 0.5:
        stress = -stress

    return strain, stress, stress_key


def _latest_completed_result(case_dir: Path, include_smoke: bool) -> tuple[Path, list[dict[str, str]]] | None:
    results_dir = case_dir / "results"
    if not results_dir.exists():
        return None

    candidates = []
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if not include_smoke and "_smoke_" in run_dir.name:
            continue
        summary_csv = run_dir / "summary.csv"
        if not summary_csv.exists():
            continue
        rows = _read_summary_rows(summary_csv)
        if rows:
            candidates.append((run_dir, rows))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0].stat().st_mtime, reverse=True)
    return candidates[0]


def _actual_run_geometry(case_dir: Path, run_dir: Path) -> dict[str, float]:
    candidates = [
        run_dir / "cycle_000_relaxed.xyz",
        run_dir / "cycle_000_raw.xyz",
        case_dir / "inputs" / "init.vasp",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            return projected_xy_geometry(read(str(path)))
        except Exception:
            continue
    return {
        "span_x_A": float("nan"),
        "span_y_A": float("nan"),
        "area_bbox_xy_A2": float("nan"),
        "area_hull_xy_A2": float("nan"),
        "equiv_diameter_hull_nm": float("nan"),
        "n_hull_vertices": float("nan"),
    }


def _label_from_manifest(case_dir: Path, run_dir: Path, manifest: dict) -> tuple[str, str, float, dict[str, float]]:
    geometry = manifest.get("geometry", {})
    vacancy = manifest.get("vacancy", {})
    orientation = str(geometry.get("orientation", "unknown"))
    geom_actual = _actual_run_geometry(case_dir, run_dir)
    diameter_nm = float(geom_actual.get("equiv_diameter_hull_nm", float("nan")))
    vac_pct = float(vacancy.get("conc_pct", 0.0) or 0.0)
    if np.isfinite(diameter_nm) and diameter_nm > 0.0:
        label = f"{orientation} | d_eq {diameter_nm:.2f} nm | vac {vac_pct:.1f}%"
    else:
        label = f"{orientation} | vac {vac_pct:.1f}%"
    return label, orientation, diameter_nm, geom_actual


def main() -> None:
    args = _parse_args()
    cases_root = ROOT / "cases"
    out_png = Path(args.out).resolve()
    out_csv = Path(args.table).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    curves = []
    table_rows: list[dict[str, str | float | int]] = []

    for case_dir in sorted(cases_root.glob(str(args.pattern))):
        if not case_dir.is_dir():
            continue

        latest = _latest_completed_result(case_dir, include_smoke=bool(args.include_smoke))
        if latest is None:
            continue

        run_dir, rows = latest
        parsed = _series_from_rows(rows)
        if parsed is None:
            continue

        strain, stress, stress_key = parsed
        manifest = _read_manifest(case_dir)
        label, orientation, diameter_nm, geom_actual = _label_from_manifest(case_dir, run_dir, manifest)
        peak_idx = int(np.argmax(stress))
        final_idx = len(strain) - 1

        curves.append(
            {
                "case": case_dir.name,
                "run_dir": run_dir,
                "label": label,
                "orientation": orientation,
                "diameter_nm": diameter_nm,
                "strain": strain,
                "stress": stress,
                "stress_key": stress_key,
                "peak_idx": peak_idx,
                "geom_actual": geom_actual,
            }
        )
        table_rows.append(
            {
                "case": case_dir.name,
                "result_dir": str(run_dir),
                "orientation": orientation,
                "diameter_nm": diameter_nm,
                "span_x_A": float(geom_actual["span_x_A"]),
                "span_y_A": float(geom_actual["span_y_A"]),
                "area_hull_xy_A2": float(geom_actual["area_hull_xy_A2"]),
                "n_points": len(strain),
                "peak_tensile_stress_GPa": float(stress[peak_idx]),
                "peak_strain_pct": float(strain[peak_idx] * 100.0),
                "final_strain_pct": float(strain[final_idx] * 100.0),
                "stress_source": stress_key,
            }
        )

    if not curves:
        print("[plot] No completed curves found for the requested cases.")
        return

    curves.sort(key=lambda item: (item["diameter_nm"], item["orientation"], item["case"]))

    fig, ax = plt.subplots(figsize=(9, 6))
    for item in curves:
        color = ORIENTATION_COLORS.get(str(item["orientation"]), "#444444")
        linestyle = DIAMETER_STYLES.get(round(float(item["diameter_nm"]), 1), "-.")
        strain_pct = np.asarray(item["strain"], dtype=float) * 100.0
        stress_gpa = np.asarray(item["stress"], dtype=float)
        ax.plot(
            strain_pct,
            stress_gpa,
            linestyle=linestyle,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            label=str(item["label"]),
        )
        peak_idx = int(item["peak_idx"])
        ax.scatter(
            [strain_pct[peak_idx]],
            [stress_gpa[peak_idx]],
            color=color,
            s=28,
            zorder=5,
        )

    ax.set_title("Our Finite-Wire Al Nanocrystal Strain-Stress Curves")
    ax.set_xlabel("Engineering strain (%)")
    ax.set_ylabel("Tensile stress from grip force (GPa)")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=180)
    plt.close(fig)

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "case",
                "result_dir",
                "orientation",
                "diameter_nm",
                "span_x_A",
                "span_y_A",
                "area_hull_xy_A2",
                "n_points",
                "peak_tensile_stress_GPa",
                "peak_strain_pct",
                "final_strain_pct",
                "stress_source",
            ],
        )
        writer.writeheader()
        writer.writerows(table_rows)

    print(f"[plot] Wrote curve figure: {out_png}")
    print(f"[plot] Wrote curve table : {out_csv}")


if __name__ == "__main__":
    main()
