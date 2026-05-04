from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CASES_DIR = ROOT / "cases"


def _read_key_value_summary(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def _diameter_from_case(case_name: str) -> float:
    match = re.search(r"_(\d+(?:\.\d+)?)nm_", case_name)
    if not match:
        raise ValueError(f"Could not parse diameter from case name: {case_name}")
    return float(match.group(1))


def _read_summary_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _latest_bulk_result() -> dict[str, object]:
    comparison_dir = RESULTS_DIR / "bulk_DFT_OFDFT_QE_DFTpy_20260416"
    comparison_summary = comparison_dir / "qe_dftpy_bulk_summary.txt"
    comparison_table = comparison_dir / "bulk_benchmark_comparison.csv"
    comparison_png = comparison_dir / "bulk_compare_FINAL.png"
    if comparison_summary.exists() and comparison_table.exists() and comparison_png.exists():
        rows = _read_summary_rows(comparison_table)
        benchmark = next((row for row in rows if row["method"].startswith("Standard") or row["method"].startswith("Benchmark")), rows[0])
        qe_row = next((row for row in rows if "QE" in row["method"]), None)
        ofdft_row = next((row for row in rows if "OF-DFT" in row["method"]), None)
        return {
            "bulk_dir": comparison_dir,
            "summary_txt": comparison_summary,
            "summary": _read_key_value_summary(comparison_summary),
            "comparison_csv": comparison_table,
            "comparison_png": comparison_png,
            "benchmark_row": benchmark,
            "qe_row": qe_row,
            "ofdft_row": ofdft_row,
        }

    candidates = sorted(
        RESULTS_DIR.glob("bulk_Al_fcc_TFVW*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for bulk_dir in candidates:
        summary_txt = bulk_dir / "summary.txt"
        a0_scan_csv = bulk_dir / "a0_scan.csv"
        a0_scan_png = bulk_dir / "a0_scan.png"
        bulk_validation_png = bulk_dir / "bulk_validation.png"
        if not all(path.exists() for path in [summary_txt, a0_scan_csv, a0_scan_png, bulk_validation_png]):
            continue
        rows = _read_summary_rows(a0_scan_csv)
        sampled_min = min(rows, key=lambda row: float(row["energy_per_atom_eV"]))
        return {
            "bulk_dir": bulk_dir,
            "summary_txt": summary_txt,
            "summary": _read_key_value_summary(summary_txt),
            "a0_scan_csv": a0_scan_csv,
            "a0_scan_png": a0_scan_png,
            "bulk_validation_png": bulk_validation_png,
            "sampled_min": sampled_min,
        }
    raise RuntimeError("No completed bulk TFVW validation directory was found.")


def _latest_completed_short_runs() -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for case_dir in sorted(CASES_DIR.glob("paper_periodic_111_*nm_tfvw")):
        results_dir = case_dir / "results"
        if not results_dir.exists():
            continue
        candidates = sorted(
            results_dir.glob("paper_r*_short_tfvw_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for run_dir in candidates:
            summary_csv = run_dir / "summary.csv"
            stress_png = run_dir / "stress_strain.png"
            if not summary_csv.exists() or not stress_png.exists():
                continue
            rows = _read_summary_rows(summary_csv)
            if not rows:
                continue
            last_cycle = int(rows[-1]["cycle"])
            if last_cycle < 20:
                continue
            diameter_nm = _diameter_from_case(case_dir.name)
            runs.append(
                {
                    "diameter_nm": diameter_nm,
                    "case_dir": case_dir,
                    "run_dir": run_dir,
                    "summary_csv": summary_csv,
                    "stress_png": stress_png,
                    "rows": rows,
                }
            )
            break
    return sorted(runs, key=lambda item: float(item["diameter_nm"]))


def _peak_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return max(rows, key=lambda row: float(row["wire_stress_ref_GPa"]))


def _write_summary_csv(path: Path, runs: list[dict[str, object]]) -> None:
    fieldnames = [
        "diameter_nm",
        "case_dir",
        "run_dir",
        "peak_cycle",
        "peak_strain",
        "peak_wire_stress_ref_GPa",
        "final_cycle",
        "final_strain",
        "final_wire_stress_ref_GPa",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            rows = run["rows"]
            peak = _peak_row(rows)
            last = rows[-1]
            writer.writerow(
                {
                    "diameter_nm": f"{float(run['diameter_nm']):.1f}",
                    "case_dir": str(run["case_dir"]),
                    "run_dir": str(run["run_dir"]),
                    "peak_cycle": peak["cycle"],
                    "peak_strain": peak["strain"],
                    "peak_wire_stress_ref_GPa": peak["wire_stress_ref_GPa"],
                    "final_cycle": last["cycle"],
                    "final_strain": last["strain"],
                    "final_wire_stress_ref_GPa": last["wire_stress_ref_GPa"],
                }
            )


def _plot_overlay(path: Path, runs: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#0b5d7a", "#c84c09", "#4f772d", "#7c3aed", "#a21caf"]
    for idx, run in enumerate(runs):
        rows = run["rows"]
        strain = np.array([float(row["strain"]) * 100.0 for row in rows], dtype=float)
        stress = np.array([float(row["wire_stress_ref_GPa"]) for row in rows], dtype=float)
        peak = _peak_row(rows)
        peak_x = float(peak["strain"]) * 100.0
        peak_y = float(peak["wire_stress_ref_GPa"])
        label = f"{float(run['diameter_nm']):.1f} nm"
        color = colors[idx % len(colors)]
        ax.plot(strain, stress, marker="o", markersize=4, linewidth=2.0, color=color, label=label)
        ax.scatter([peak_x], [peak_y], color=color, s=60, zorder=5)

    ax.set_title("[111] Al Short-Wire Stress-Strain Comparison (TFVW)")
    ax.set_xlabel("Engineering strain (%)")
    ax.set_ylabel("Wire stress (GPa)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_grid(path: Path, runs: list[dict[str, object]]) -> None:
    n = len(runs)
    cols = min(3, max(1, n))
    rows_n = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(5.5 * cols, 4.0 * rows_n), squeeze=False)
    colors = ["#0b5d7a", "#c84c09", "#4f772d", "#7c3aed", "#a21caf"]

    for ax in axes.ravel():
        ax.set_visible(False)

    for idx, run in enumerate(runs):
        ax = axes[idx // cols][idx % cols]
        ax.set_visible(True)
        rows = run["rows"]
        strain = np.array([float(row["strain"]) * 100.0 for row in rows], dtype=float)
        stress = np.array([float(row["wire_stress_ref_GPa"]) for row in rows], dtype=float)
        peak = _peak_row(rows)
        peak_x = float(peak["strain"]) * 100.0
        peak_y = float(peak["wire_stress_ref_GPa"])
        color = colors[idx % len(colors)]
        ax.plot(strain, stress, marker="o", markersize=3.5, linewidth=2.0, color=color)
        ax.scatter([peak_x], [peak_y], color=color, s=50, zorder=5)
        ax.set_title(f"{float(run['diameter_nm']):.1f} nm")
        ax.set_xlabel("Strain (%)")
        ax.set_ylabel("Wire stress (GPa)")
        ax.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("[111] Al Short-Wire Stress-Strain Curves", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_strength_vs_diameter(path: Path, runs: list[dict[str, object]]) -> None:
    diameters = np.array([float(run["diameter_nm"]) for run in runs], dtype=float)
    peaks = np.array([float(_peak_row(run["rows"])["wire_stress_ref_GPa"]) for run in runs], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(diameters, peaks, marker="o", markersize=6, linewidth=2.0, color="#0b5d7a")
    for x, y in zip(diameters, peaks):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_title("Peak Short-Wire Strength vs Diameter")
    ax.set_xlabel("Diameter (nm)")
    ax.set_ylabel("Peak wire stress (GPa)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _copy_key_images(outdir: Path, bulk: dict[str, object], runs: list[dict[str, object]]) -> None:
    for stale in outdir.glob("[0-9][0-9]_stress_strain_*.png"):
        stale.unlink()
    if "comparison_png" in bulk:
        shutil.copy2(Path(bulk["comparison_png"]), outdir / "00_bulk_DFT_vs_OFDFT_QE_DFTpy.png")
    else:
        shutil.copy2(Path(bulk["a0_scan_png"]), outdir / "01_bulk_a0_scan.png")
        shutil.copy2(Path(bulk["bulk_validation_png"]), outdir / "02_bulk_validation.png")
    for idx, run in enumerate(runs, start=3):
        diameter = float(run["diameter_nm"])
        shutil.copy2(run["stress_png"], outdir / f"{idx:02d}_stress_strain_{diameter:.1f}nm.png")


def _write_summary_md(path: Path, bulk: dict[str, object], runs: list[dict[str, object]]) -> None:
    summary = bulk["summary"]
    lines = [
        "# Professor Review",
        "",
        "Completed short-wire results are summarized here. The bulk reference shown below uses the current EOS-style comparison requested by the professor.",
        "",
        "## Figures",
        "",
        "- `00_bulk_DFT_vs_OFDFT_QE_DFTpy.png`",
        "- `10_completed_short_wire_overlay.png`",
        "- `11_completed_short_wire_grid.png`",
        "- `12_peak_strength_vs_diameter.png`",
        "",
        "## Bulk reference",
        "",
        f"- Bulk directory: `{Path(bulk['bulk_dir']).name}`",
    ]
    if "benchmark_row" in bulk and bulk.get("qe_row") and bulk.get("ofdft_row"):
        benchmark_row = bulk["benchmark_row"]
        qe_row = bulk["qe_row"]
        ofdft_row = bulk["ofdft_row"]
        lines.extend(
            [
                f"- Benchmark `a0`: `{float(benchmark_row['a0_A']):.4f} A`",
                f"- Benchmark `B0`: `{float(benchmark_row['bulk_modulus_GPa']):.1f} GPa`",
                f"- QE / KS-DFT `a0`: `{float(qe_row['a0_A']):.6f} A`",
                f"- QE / KS-DFT `B0`: `{float(qe_row['bulk_modulus_GPa']):.2f} GPa`",
                f"- OF-DFT / TFvW `a0`: `{float(ofdft_row['a0_A']):.6f} A`",
                f"- OF-DFT / TFvW `B0`: `{float(ofdft_row['bulk_modulus_GPa']):.2f} GPa`",
                "",
            ]
        )
    else:
        sampled_min = bulk["sampled_min"]
        lines.extend(
            [
                f"- Sampled minimum in `a0 scan`: `{float(sampled_min['a0_A']):.3f} A`",
                f"- Quadratic-fit minimum in `a0 scan`: `{float(summary['a0_ref_A']):.6f} A`",
                f"- EOS bulk modulus: `{float(summary['eos_bulk_modulus_GPa']):.6f} GPa`",
                f"- Small-strain axial slope: `{float(summary['stress_slope_GPa']):.6f} GPa`",
                "",
            ]
        )
    lines.extend(
        [
        "## Individual stress-strain figures",
        "",
        ]
    )
    for idx, run in enumerate(runs, start=3):
        lines.append(f"- `{idx:02d}_stress_strain_{float(run['diameter_nm']):.1f}nm.png`")

    lines.extend(
        [
            "",
        "## Completed cases",
        "",
        ]
    )
    for run in runs:
        peak = _peak_row(run["rows"])
        lines.extend(
            [
                f"- `{float(run['diameter_nm']):.1f} nm`: peak `{float(peak['wire_stress_ref_GPa']):.6f} GPa` "
                f"at `{float(peak['strain']) * 100.0:.3f}%` strain",
            ]
        )

    lines.extend(
        [
            "",
            "## Table",
            "",
            "- `completed_short_wire_summary.csv`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build professor-facing summary figures for completed short-wire runs.")
    ap.add_argument("--outdir", default="results/professor_review")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    bulk = _latest_bulk_result()
    runs = _latest_completed_short_runs()
    if not runs:
        raise RuntimeError("No completed short-wire runs were found.")

    _copy_key_images(outdir, bulk, runs)
    _write_summary_csv(outdir / "completed_short_wire_summary.csv", runs)
    _plot_overlay(outdir / "10_completed_short_wire_overlay.png", runs)
    _plot_grid(outdir / "11_completed_short_wire_grid.png", runs)
    _plot_strength_vs_diameter(outdir / "12_peak_strength_vs_diameter.png", runs)
    _write_summary_md(outdir / "SUMMARY.md", bulk, runs)

    print(f"[professor-review] Wrote: {outdir}")


if __name__ == "__main__":
    main()
