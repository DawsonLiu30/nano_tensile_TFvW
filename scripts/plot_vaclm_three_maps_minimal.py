#!/usr/bin/env python3
"""Make very simple professor-style lambda/mu contour sketches.

This is intentionally less detailed than the QC plots:
white background, contour lines only, minimal labels.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


N_PRISTINE = 108
N_VACANCY = 107
LAMBDA_VALUES = np.array([round(0.1 * i, 1) for i in range(1, 11)])
MU_VALUES = np.array([round(0.1 * i, 1) for i in range(1, 11)])


def safe_path(path: str | Path) -> Path:
    p = Path(path)
    if os.name == "nt":
        s = str(p.resolve(strict=False))
        if not s.startswith("\\\\?\\"):
            return Path("\\\\?\\" + s)
    return p


def default_root() -> Path:
    candidates = [
        "iservice_packages/VACLM_PROFESSOR_RAW_IO_PACKAGE_20260624_FIXED/results/Al_defects/01_calibration/single_vacancy/dftpy_tfvw_lambda_mu/coarse_10x10_vacancy_formation",
        "iservice_packages/results/Al_defects/01_calibration/single_vacancy/dftpy_tfvw_lambda_mu/coarse_10x10_vacancy_formation",
    ]
    for c in candidates:
        p = safe_path(c)
        if p.exists():
            return p
    raise FileNotFoundError("Cannot find VACLM package root.")


def load_data(root: Path, qualified_only: bool) -> pd.DataFrame:
    flat = root / "10_professor_raw_log_table" / "professor_raw_output_flat_pristine_and_vacancy.csv"
    audit = root / "07_audit" / "vaclm_matrix_status_live.csv"
    raw = pd.read_csv(flat)
    aud = pd.read_csv(audit) if audit.exists() else pd.DataFrame()

    for c in ["lambda", "mu", "raw_total_energy", "raw_kinetic_energy", "lattice_constant_a0_A"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    p = raw[raw["role"].str.lower() == "pristine"][
        ["setting", "lambda", "mu", "raw_total_energy", "raw_kinetic_energy", "lattice_constant_a0_A"]
    ].rename(
        columns={
            "raw_total_energy": "E_pristine",
            "raw_kinetic_energy": "K_pristine",
            "lattice_constant_a0_A": "a0",
        }
    )
    v = raw[raw["role"].str.lower() == "vacancy"][
        ["setting", "raw_total_energy", "raw_kinetic_energy"]
    ].rename(columns={"raw_total_energy": "E_vacancy", "raw_kinetic_energy": "K_vacancy"})

    df = p.merge(v, on="setting", how="inner")
    factor = N_VACANCY / N_PRISTINE
    df["Ef"] = df["E_vacancy"] - factor * df["E_pristine"]
    df["dKEDF"] = df["K_vacancy"] - factor * df["K_pristine"]

    if not aud.empty and "qualified" in aud.columns:
        df = df.merge(aud[["setting", "qualified", "latest_status"]], on="setting", how="left")
        df["qualified_bool"] = df["qualified"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        df["qualified_bool"] = True
        df["latest_status"] = ""

    df = df.dropna(subset=["lambda", "mu", "a0", "Ef", "dKEDF"])
    if qualified_only:
        df = df[df["qualified_bool"]].copy()
    return df


def interpolate(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df["mu"].to_numpy(float)
    y = df["lambda"].to_numpy(float)
    z = df[value_col].to_numpy(float)
    gx, gy = np.meshgrid(np.linspace(0.1, 1.0, 220), np.linspace(0.1, 1.0, 220))
    gz = griddata((x, y), z, (gx, gy), method="linear")
    return gx, gy, gz


def nice_levels(values: np.ndarray, n: int = 7) -> np.ndarray:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return np.array([])
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    return np.linspace(lo, hi, n)


def panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    title: str,
    label_fmt: str,
    *,
    blue_levels: np.ndarray | None = None,
    red_level: float | None = None,
    red_point: tuple[float, float] | None = None,
) -> None:
    gx, gy, gz = interpolate(df, value_col)
    levels = blue_levels if blue_levels is not None else nice_levels(gz, 7)
    if levels.size:
        cs = ax.contour(gx, gy, gz, levels=levels, colors="#164a9b", linewidths=1.05)
        ax.clabel(cs, inline=True, fontsize=7, fmt=label_fmt)

    if red_level is not None:
        vals = gz[np.isfinite(gz)]
        if vals.size and vals.min() <= red_level <= vals.max():
            rs = ax.contour(gx, gy, gz, levels=[red_level], colors="#d62728", linewidths=1.6)
            ax.clabel(rs, inline=True, fontsize=8, fmt=label_fmt)

    if red_point is not None:
        mu, lam = red_point
        ax.scatter([mu], [lam], s=95, facecolors="none", edgecolors="#d62728", linewidths=1.8)

    ax.set_title(title, fontsize=11)
    ax.set_xlim(0.08, 1.02)
    ax.set_ylim(0.08, 1.02)
    ax.set_xticks([0.1, 1.0])
    ax.set_yticks([0.1, 1.0])
    ax.set_xlabel("mu", fontsize=10)
    ax.set_ylabel("lambda", fontsize=10)
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rootdir", default=None)
    ap.add_argument("--outdir", default="C:/Users/dawso/Desktop/VACLM_THREE_MAPS_MINIMAL_20260625")
    ap.add_argument("--qe-ref", type=float, default=0.60)
    ap.add_argument("--qualified-only", action="store_true", default=False)
    args = ap.parse_args()

    root = safe_path(args.rootdir) if args.rootdir else default_root()
    outdir = safe_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(root, qualified_only=args.qualified_only)
    tag = "qualified_only" if args.qualified_only else "all_numeric"
    df.to_csv(outdir / f"minimal_plot_data_{tag}.csv", index=False)

    closest = df.assign(diff=(df["Ef"] - args.qe_ref).abs()).sort_values("diff").iloc[0]
    red_point = (float(closest["mu"]), float(closest["lambda"]))

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.05), dpi=260)
    panel(axes[0], df, "a0", "a0", "%.2f", blue_levels=np.array([3.25, 3.50, 3.75, 4.00]), red_level=4.04)
    panel(axes[1], df, "Ef", "Ef", "%.1f", blue_levels=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), red_point=red_point)
    panel(axes[2], df, "dKEDF", "KEDF", "%.1f", blue_levels=np.array([-5.0, -4.0, -3.0, -2.0]))
    fig.tight_layout(w_pad=2.0)

    combined_png = outdir / f"professor_three_maps_minimal_{tag}.png"
    combined_pdf = outdir / f"professor_three_maps_minimal_{tag}.pdf"
    fig.savefig(combined_png, bbox_inches="tight")
    fig.savefig(combined_pdf, bbox_inches="tight")
    plt.close(fig)

    for name, col, title, fmt in [
        ("a0", "a0", "a0", "%.2f"),
        ("Ef", "Ef", "Ef", "%.1f"),
        ("KEDF", "dKEDF", "KEDF", "%.1f"),
    ]:
        fig, ax = plt.subplots(figsize=(3.25, 3.05), dpi=260)
        levels = None
        if name == "a0":
            levels = np.array([3.25, 3.50, 3.75, 4.00])
        elif name == "Ef":
            levels = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        elif name == "KEDF":
            levels = np.array([-5.0, -4.0, -3.0, -2.0])
        panel(
            ax,
            df,
            col,
            title,
            fmt,
            blue_levels=levels,
            red_level=4.04 if name == "a0" else None,
            red_point=red_point if name == "Ef" else None,
        )
        fig.tight_layout()
        fig.savefig(outdir / f"fig_{name}_minimal_{tag}.png", bbox_inches="tight")
        fig.savefig(outdir / f"fig_{name}_minimal_{tag}.pdf", bbox_inches="tight")
        plt.close(fig)

    (outdir / f"closest_to_qe_{tag}.txt").write_text(
        f"QE reference: {args.qe_ref} eV\n"
        f"Closest plotted point: {closest['setting']}\n"
        f"lambda={closest['lambda']}, mu={closest['mu']}, Ef={closest['Ef']:.6f} eV\n",
        encoding="utf-8",
    )
    print(combined_png)
    print(combined_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
