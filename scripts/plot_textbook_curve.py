from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _pick_column(columns: list[str], preferred: list[str]) -> str:
    for key in preferred:
        if key in columns:
            return key
    raise KeyError(f"None of the preferred columns exist: {preferred}")


def main():
    result_dirs = glob.glob("results/*") + glob.glob("cases/*/results/*")
    if not result_dirs:
        print("[plot] No result directories found under results/")
        return

    summary_candidates = [Path(result_dir) / "summary.csv" for result_dir in result_dirs]
    summary_candidates = [path for path in summary_candidates if path.exists()]
    if not summary_candidates:
        print("[plot] No summary.csv files found under results/ or cases/*/results/")
        return
    summary_path = max(summary_candidates, key=os.path.getmtime)
    latest_dir = summary_path.parent

    print(f"[plot] Reading: {summary_path}")
    df = pd.read_csv(summary_path)
    if df.empty:
        print("[plot] Empty summary.csv")
        return

    columns = df.columns.tolist()
    strain_key = _pick_column(columns, ["free_region_strain", "strain"])
    stress_key = _pick_column(
        columns,
        [
            "wire_stress_free_current_GPa",
            "eng_stress_top_GPa",
            "wire_stress_free_ref_GPa",
            "sigma_zz_GPa",
        ],
    )

    df = df.dropna(subset=[strain_key, stress_key]).copy()
    if df.empty:
        print("[plot] No valid stress-strain points after dropping NaN rows.")
        return

    strain_pct = df[strain_key].astype(float) * 100.0
    stress_gpa = df[stress_key].astype(float)
    if (stress_gpa < 0.0).mean() > 0.5:
        stress_gpa = -stress_gpa
        stress_label = f"-{stress_key} (GPa)"
    else:
        stress_label = f"{stress_key} (GPa)"

    max_idx = int(stress_gpa.idxmax())
    yield_strain = float(strain_pct.loc[max_idx])
    yield_stress = float(stress_gpa.loc[max_idx])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        strain_pct,
        stress_gpa,
        marker="o",
        markersize=5,
        linestyle="-",
        linewidth=2.0,
        color="#1f77b4",
        label="Free-region response",
    )
    ax.scatter([yield_strain], [yield_stress], color="red", s=80, zorder=5)
    ax.annotate(
        f"Peak\n({yield_strain:.2f}%, {yield_stress:.2f} GPa)",
        xy=(yield_strain, yield_stress),
        xytext=(yield_strain + 0.8, yield_stress),
        arrowprops=dict(facecolor="red", shrink=0.05, width=1.5, headwidth=7),
        fontsize=11,
        color="red",
    )

    ax.set_title("Stress-Strain Curve of [111] Al Nanowire")
    ax.set_xlabel("Free-region engineering strain (%)")
    ax.set_ylabel(stress_label)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    plt.tight_layout()

    output_png = Path(latest_dir) / "stress_strain_curve.png"
    plt.savefig(output_png, dpi=300)
    print(f"[plot] Saved: {output_png}")


if __name__ == "__main__":
    main()
