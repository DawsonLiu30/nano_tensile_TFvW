#!/usr/bin/env python3
"""Draw simple professor-style lambda/mu maps for the local VACLM rerun.

The professor requested three very simple maps using lambda and mu as axes:

1. relaxed pristine lattice constant a0
2. vacancy formation energy using the Gillan-style supercell formula
3. pristine KEDF / kinetic energy from the raw DFTpy output

This script intentionally keeps the plots minimal: contour lines, labels,
computed points, and blank space where the local run failed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


LAMBDA_VALUES = [round(0.1 * i, 1) for i in range(1, 11)]
MU_VALUES = [round(0.1 * i, 1) for i in range(1, 11)]
QE_EF_REF_EV = 0.601167
AL_LATTICE_REF_A = 4.05
GILLAN_CALC_EF_EV = 0.56
GILLAN_EXP_EF_EV = 0.66


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rootdir",
        default=r"C:\Users\dawso\Desktop\LOCAL_DFTPY_VACLM_10X10_FULL_RERUN_LBFGS_GUARDED_20260626",
        help="Local VACLM rerun root.",
    )
    parser.add_argument(
        "--outdir",
        default=r"C:\Users\dawso\Desktop\LOCAL_DFTPY_VACLM_SIMPLE_MAPS_20260629",
        help="Output folder for figures and plot data.",
    )
    return parser.parse_args()


def read_flat_rows(root: Path) -> list[dict[str, str]]:
    flat = root / "10_professor_raw_log_table" / "professor_raw_log_flat_results.csv"
    with flat.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def empty_grid() -> np.ndarray:
    return np.full((len(MU_VALUES), len(LAMBDA_VALUES)), np.nan, dtype=float)


def grid_set(grid: np.ndarray, lam: float, mu: float, value: float | None) -> None:
    if value is None:
        return
    try:
        i = MU_VALUES.index(round(mu, 1))
        j = LAMBDA_VALUES.index(round(lam, 1))
    except ValueError:
        return
    grid[i, j] = value


def build_grids(root: Path, rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    a0_grid = empty_grid()
    kedf_grid = empty_grid()
    ef_grid = empty_grid()
    long_rows: list[dict[str, object]] = []

    for row in rows:
        lam = float(row["lambda"])
        mu = float(row["mu"])
        setting = row["setting"]
        case_dir = Path(row["source_dir"])

        a0 = safe_float(row.get("pristine_lattice_constant_A"))
        kedf = safe_float(row.get("pristine_kedf_energy_eV"))
        ef = None
        result = case_dir / "result.json"
        if result.exists():
            try:
                data = json.loads(result.read_text(encoding="utf-8", errors="replace"))
                ef = safe_float(str(data.get("vacancy_formation_energy_eV", "")))
            except Exception:
                ef = None

        grid_set(a0_grid, lam, mu, a0)
        grid_set(kedf_grid, lam, mu, kedf)
        grid_set(ef_grid, lam, mu, ef)

        long_rows.append(
            {
                "setting": setting,
                "lambda": lam,
                "mu": mu,
                "status": row.get("raw_table_status", ""),
                "pristine_lattice_constant_A": a0,
                "pristine_kedf_energy_eV": kedf,
                "vacancy_formation_energy_eV": ef,
                "source_dir": str(case_dir),
                "result_json": str(result),
            }
        )

    return a0_grid, kedf_grid, ef_grid, long_rows


def write_plot_data(outdir: Path, rows: Iterable[dict[str, object]]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "setting",
        "lambda",
        "mu",
        "status",
        "pristine_lattice_constant_A",
        "pristine_kedf_energy_eV",
        "vacancy_formation_energy_eV",
        "source_dir",
        "result_json",
    ]
    with (outdir / "professor_simple_map_plot_data.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def finite_points(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(np.isfinite(grid))
    return np.array([LAMBDA_VALUES[x] for x in xs]), np.array([MU_VALUES[y] for y in ys])


def contour_levels(grid: np.ndarray, n: int = 6) -> np.ndarray:
    vals = grid[np.isfinite(grid)]
    if len(vals) == 0:
        return np.array([])
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if abs(hi - lo) < 1e-12:
        return np.array([lo])
    return np.linspace(lo, hi, n)


def plot_panel(
    ax: plt.Axes,
    grid: np.ndarray,
    title: str,
    label: str,
    levels: list[float] | None = None,
    ref: float | None = None,
    ref_label: str | None = None,
    extra_refs: list[tuple[float, str]] | None = None,
    note: str | None = None,
    fmt: str = "%.3g",
) -> None:
    x, y = np.meshgrid(LAMBDA_VALUES, MU_VALUES)
    masked = np.ma.masked_invalid(grid)
    valid_x, valid_y = finite_points(grid)

    if levels is None:
        levels_arr = contour_levels(grid, n=6)
    else:
        levels_arr = np.array(levels, dtype=float)

    ax.set_facecolor("#f7f7f2")
    ax.scatter(valid_x, valid_y, s=8, color="#333333", alpha=0.34, zorder=3)

    if len(levels_arr) > 0 and np.isfinite(grid).sum() >= 4:
        cs = ax.contour(x, y, masked, levels=levels_arr, colors="#1f4d78", linewidths=1.05)
        ax.clabel(cs, inline=True, fontsize=7, fmt=fmt)

    if ref is not None and np.nanmin(grid) <= ref <= np.nanmax(grid):
        ref_cs = ax.contour(x, y, masked, levels=[ref], colors="#b41f2a", linewidths=1.6)
        ax.clabel(ref_cs, inline=True, fontsize=8, fmt={ref: ref_label or f"{ref:.3g}"})

    for extra_ref, extra_label in extra_refs or []:
        if np.nanmin(grid) <= extra_ref <= np.nanmax(grid):
            ref_cs = ax.contour(x, y, masked, levels=[extra_ref], colors="#b41f2a", linewidths=1.35)
            ax.clabel(ref_cs, inline=True, fontsize=8, fmt={extra_ref: extra_label})

    ax.set_title(title, fontsize=11, pad=7)
    ax.set_xlabel("lambda")
    ax.set_ylabel("mu")
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(0.1, 1.0)
    ax.set_xticks([0.1, 1.0])
    ax.set_yticks([0.1, 1.0])
    ax.tick_params(labelsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.text(0.03, 0.97, label, transform=ax.transAxes, va="top", ha="left", fontsize=8, color="#444444")
    if note:
        ax.text(0.03, 0.04, note, transform=ax.transAxes, va="bottom", ha="left", fontsize=7.5, color="#8d1f24")


def main() -> int:
    args = parse_args()
    root = Path(args.rootdir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_flat_rows(root)
    a0_grid, kedf_grid, ef_grid, long_rows = build_grids(root, rows)
    write_plot_data(outdir, long_rows)

    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.15), dpi=300, constrained_layout=True)
    plot_panel(
        axes[0],
        a0_grid,
        "lattice constant",
        "a0 (A)",
        levels=[3.4, 3.6, 3.8, 4.0, 4.2],
        ref=AL_LATTICE_REF_A,
        ref_label="Al 4.05",
        note="red: Al 4.05 A",
        fmt="%.2f",
    )
    plot_panel(
        axes[1],
        ef_grid,
        "vacancy formation energy",
        "Ef (eV)",
        levels=[1.0, 2.0, 3.0, 4.0, 5.0],
        ref=GILLAN_CALC_EF_EV,
        ref_label="0.56",
        extra_refs=[(GILLAN_EXP_EF_EV, "0.66")],
        note="red: 0.56 / 0.66 eV",
        fmt="%.1f",
    )
    plot_panel(
        axes[2],
        kedf_grid,
        "KEDF",
        "KEDF (eV)",
        levels=[2000, 2200, 2400, 2500],
        fmt="%.0f",
    )

    fig.suptitle("DFTpy TFvW lambda-mu scan", fontsize=12)
    combined_png = outdir / "professor_three_maps_minimal.png"
    combined_pdf = outdir / "professor_three_maps_minimal.pdf"
    fig.savefig(combined_png, bbox_inches="tight")
    fig.savefig(combined_pdf, bbox_inches="tight")
    plt.close(fig)

    single_specs = [
        ("professor_map_lattice_constant.png", a0_grid, "lattice constant", "a0 (A)", [3.4, 3.6, 3.8, 4.0, 4.2], AL_LATTICE_REF_A, "Al 4.05", [], "red: Al 4.05 A", "%.2f"),
        ("professor_map_vacancy_formation_energy.png", ef_grid, "vacancy formation energy", "Ef (eV)", [1.0, 2.0, 3.0, 4.0, 5.0], GILLAN_CALC_EF_EV, "0.56", [(GILLAN_EXP_EF_EV, "0.66")], "red: 0.56 / 0.66 eV", "%.1f"),
        ("professor_map_kedf.png", kedf_grid, "KEDF", "KEDF (eV)", [2000, 2200, 2400, 2500], None, None, [], None, "%.0f"),
    ]
    for filename, grid, title, label, levels, ref, ref_label, extra_refs, note, fmt in single_specs:
        f, ax = plt.subplots(figsize=(3.2, 3.2), dpi=300, constrained_layout=True)
        plot_panel(ax, grid, title, label, levels=levels, ref=ref, ref_label=ref_label, extra_refs=extra_refs, note=note, fmt=fmt)
        f.savefig(outdir / filename, bbox_inches="tight")
        plt.close(f)

    readme = f"""# Simple professor-style lambda/mu maps

These maps use the local DFTpy TFvW 10x10 rerun:

```text
{root}
```

## Definitions

- x-axis: lambda
- y-axis: mu
- lattice constant: relaxed pristine conventional-cell value, `mean(|a|, |b|, |c|) / 3`
- KEDF: pristine raw `KEDF` value parsed from `pristine_dftpy.out`
- vacancy formation energy: Gillan-style supercell formula,
  `E_vac(Al107) - (107/108) E_pristine(Al108)`, read from `result.json`
- red reference line in lattice panel: Al lattice constant 4.05 A
- red reference lines in vacancy panel: Gillan calculated 0.56 eV and
  experimental 0.66 eV reference values for Al vacancy formation energy

## Missing regions

Blank regions are not interpolated. They correspond to failed/pathological local
relaxations or missing complete vacancy results. This is intentional.

## Outputs

- `professor_three_maps_minimal.png`
- `professor_three_maps_minimal.pdf`
- `professor_map_lattice_constant.png`
- `professor_map_vacancy_formation_energy.png`
- `professor_map_kedf.png`
- `professor_simple_map_plot_data.csv`
"""
    (outdir / "README_SIMPLE_MAPS.md").write_text(readme, encoding="utf-8")

    package = outdir.with_suffix(".zip")
    if package.exists():
        package.unlink()
    shutil.make_archive(str(outdir), "zip", outdir)

    summary = {
        "outdir": str(outdir),
        "package": str(package),
        "combined_png": str(combined_png),
        "combined_pdf": str(combined_pdf),
        "a0_points": int(np.isfinite(a0_grid).sum()),
        "kedf_points": int(np.isfinite(kedf_grid).sum()),
        "ef_points": int(np.isfinite(ef_grid).sum()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
