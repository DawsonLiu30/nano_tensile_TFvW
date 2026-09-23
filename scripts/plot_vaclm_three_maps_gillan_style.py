#!/usr/bin/env python3
"""Plot Gillan-style lambda/mu maps for the Al single-vacancy scan.

This script is deliberately separate from the raw professor table:

  - the raw table reports direct pristine output values;
  - these plots compute vacancy-formation quantities from pristine + vacancy
    supercells following the scaled supercell subtraction.

Formulae for Al108 -> Al107:

  Ef_vac   = E_vac(Al107) - (107/108) * E_pristine(Al108)
  dKEDF_vac = K_vac(Al107) - (107/108) * K_pristine(Al108)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LAMBDA_VALUES = np.array([round(0.1 * i, 1) for i in range(1, 11)])
MU_VALUES = np.array([round(0.1 * i, 1) for i in range(1, 11)])


def find_default_root() -> Path | None:
    candidates = [
        Path("iservice_packages/VACLM_PROFESSOR_RAW_IO_PACKAGE_20260624_FIXED/results/Al_defects/01_calibration/single_vacancy/dftpy_tfvw_lambda_mu/coarse_10x10_vacancy_formation"),
        Path("iservice_packages/results/Al_defects/01_calibration/single_vacancy/dftpy_tfvw_lambda_mu/coarse_10x10_vacancy_formation"),
    ]
    for candidate in candidates:
        candidate = safe_path(candidate)
        if candidate.exists():
            return candidate
    return None


def safe_path(path: str | Path) -> Path:
    """Return a Windows long-path-safe Path where needed."""
    p = Path(path)
    if os.name == "nt":
        s = str(p.resolve(strict=False))
        if not s.startswith("\\\\?\\"):
            return Path("\\\\?\\" + s)
    return p


def first_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        path = safe_path(path)
        if path.exists():
            return path
    raise FileNotFoundError("None of the candidate input files exists.")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_merged(flat_csv: Path, audit_csv: Path | None, n_pristine: int, n_vacancy: int) -> pd.DataFrame:
    flat = pd.read_csv(flat_csv)

    required = {
        "setting",
        "lambda",
        "mu",
        "role",
        "raw_total_energy",
        "raw_kinetic_energy",
        "lattice_constant_a0_A",
    }
    missing = sorted(required - set(flat.columns))
    if missing:
        raise KeyError(f"Flat CSV is missing required columns: {missing}")

    for col in ["lambda", "mu", "raw_total_energy", "raw_kinetic_energy", "lattice_constant_a0_A"]:
        flat[col] = numeric(flat[col])

    pristine = flat[flat["role"].str.lower() == "pristine"].copy()
    vacancy = flat[flat["role"].str.lower() == "vacancy"].copy()

    pristine = pristine[
        [
            "setting",
            "lambda",
            "mu",
            "status",
            "raw_total_energy",
            "raw_kinetic_energy",
            "lattice_constant_a0_A",
            "source_dir",
        ]
    ].rename(
        columns={
            "status": "pristine_status",
            "raw_total_energy": "E_pristine_eV",
            "raw_kinetic_energy": "K_pristine_eV",
            "lattice_constant_a0_A": "a0_pristine_A",
            "source_dir": "source_dir_pristine",
        }
    )

    vacancy = vacancy[
        [
            "setting",
            "status",
            "raw_total_energy",
            "raw_kinetic_energy",
            "lattice_constant_a0_A",
            "source_dir",
        ]
    ].rename(
        columns={
            "status": "vacancy_status",
            "raw_total_energy": "E_vacancy_eV",
            "raw_kinetic_energy": "K_vacancy_eV",
            "lattice_constant_a0_A": "a0_vacancy_A",
            "source_dir": "source_dir_vacancy",
        }
    )

    merged = pristine.merge(vacancy, on="setting", how="outer")
    factor = n_vacancy / n_pristine
    merged["Ef_vac_eV"] = merged["E_vacancy_eV"] - factor * merged["E_pristine_eV"]
    merged["dKEDF_vac_eV"] = merged["K_vacancy_eV"] - factor * merged["K_pristine_eV"]
    merged["lattice_constant_A"] = merged["a0_pristine_A"]

    if audit_csv and audit_csv.exists():
        audit = pd.read_csv(audit_csv)
        keep = [
            col
            for col in [
                "setting",
                "qualified",
                "latest_status",
                "pristine_fmax",
                "vacancy_fmax",
                "has_result_json",
            ]
            if col in audit.columns
        ]
        merged = merged.merge(audit[keep], on="setting", how="left")
    else:
        merged["qualified"] = merged[["Ef_vac_eV", "dKEDF_vac_eV", "lattice_constant_A"]].notna().all(axis=1)
        merged["latest_status"] = ""

    if "qualified" in merged.columns:
        # CSV may read booleans as strings depending on source.
        merged["qualified_bool"] = merged["qualified"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        merged["qualified_bool"] = False

    merged["has_plot_values"] = merged[["Ef_vac_eV", "dKEDF_vac_eV", "lattice_constant_A"]].notna().all(axis=1)
    return merged


def to_grid(df: pd.DataFrame, value_col: str) -> np.ndarray:
    grid = np.full((len(LAMBDA_VALUES), len(MU_VALUES)), np.nan, dtype=float)
    lam_idx = {round(v, 1): i for i, v in enumerate(LAMBDA_VALUES)}
    mu_idx = {round(v, 1): i for i, v in enumerate(MU_VALUES)}
    for _, row in df.iterrows():
        lam = round(float(row["lambda"]), 1) if pd.notna(row["lambda"]) else None
        mu = round(float(row["mu"]), 1) if pd.notna(row["mu"]) else None
        if lam in lam_idx and mu in mu_idx and pd.notna(row[value_col]):
            grid[lam_idx[lam], mu_idx[mu]] = float(row[value_col])
    return grid


def matrix_csv(path: Path, grid: np.ndarray) -> None:
    rows = [["lambda/mu", *[f"{x:.1f}" for x in MU_VALUES]]]
    for i, lam in enumerate(LAMBDA_VALUES):
        rows.append([f"{lam:.1f}", *["" if np.isnan(x) else f"{x:.10g}" for x in grid[i]]])
    pd.DataFrame(rows).to_csv(path, index=False, header=False)


def plot_cell_map(
    ax: plt.Axes,
    grid: np.ndarray,
    title: str,
    cbar_label: str,
    cmap: str,
    *,
    marker: tuple[float, float] | None = None,
) -> None:
    masked = np.ma.masked_invalid(grid)
    x_edges = np.arange(len(MU_VALUES) + 1) - 0.5
    y_edges = np.arange(len(LAMBDA_VALUES) + 1) - 0.5
    mesh = ax.pcolormesh(x_edges, y_edges, masked, cmap=cmap, shading="flat", edgecolors="#d8d8d8", linewidth=0.45)
    cbar = ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    missing_i, missing_j = np.where(~np.isfinite(grid))
    if len(missing_i):
        ax.scatter(missing_j, missing_i, marker="x", s=16, c="#9a9a9a", linewidths=0.8)

    if marker is not None:
        mu, lam = marker
        j = int(np.where(np.isclose(MU_VALUES, mu))[0][0])
        i = int(np.where(np.isclose(LAMBDA_VALUES, lam))[0][0])
        ax.scatter(j, i, marker="o", s=80, facecolors="none", edgecolors="#d62728", linewidths=1.5)

    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("mu", fontsize=10)
    ax.set_ylabel("lambda", fontsize=10)
    ax.set_xticks(range(len(MU_VALUES)))
    ax.set_yticks(range(len(LAMBDA_VALUES)))
    ax.set_xticklabels([f"{x:.1f}" for x in MU_VALUES], fontsize=8)
    ax.set_yticklabels([f"{x:.1f}" for x in LAMBDA_VALUES], fontsize=8)
    ax.set_xlim(-0.5, len(MU_VALUES) - 0.5)
    ax.set_ylim(-0.5, len(LAMBDA_VALUES) - 0.5)
    ax.set_facecolor("#f7f7f7")


def save_figures(df: pd.DataFrame, outdir: Path, tag: str, qe_ref: float) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)

    grid_a0 = to_grid(df, "lattice_constant_A")
    grid_ef = to_grid(df, "Ef_vac_eV")
    grid_kedf = to_grid(df, "dKEDF_vac_eV")

    matrix_csv(outdir / f"matrix_a0_pristine_{tag}.csv", grid_a0)
    matrix_csv(outdir / f"matrix_Ef_vac_{tag}.csv", grid_ef)
    matrix_csv(outdir / f"matrix_dKEDF_vac_{tag}.csv", grid_kedf)

    marker = None
    valid = df.dropna(subset=["Ef_vac_eV"])
    if not valid.empty:
        best = valid.assign(diff=(valid["Ef_vac_eV"] - qe_ref).abs()).sort_values("diff").iloc[0]
        marker = (float(best["mu"]), float(best["lambda"]))
        (outdir / f"closest_to_qe_reference_{tag}.json").write_text(
            json.dumps(
                {
                    "qe_reference_eV": qe_ref,
                    "setting": best["setting"],
                    "lambda": float(best["lambda"]),
                    "mu": float(best["mu"]),
                    "Ef_vac_eV": float(best["Ef_vac_eV"]),
                    "difference_eV": float(best["diff"]),
                    "note": "Marked by red circle in the Ef_vac map.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    figs: dict[str, str] = {}
    specs = [
        ("a0", grid_a0, "Relaxed pristine lattice constant", "a0 (A)", "viridis", None),
        ("Ef_vac", grid_ef, f"Vacancy formation energy (QE ref ~{qe_ref:.2f} eV)", "Ef_vac (eV)", "YlOrRd", marker),
        ("dKEDF_vac", grid_kedf, "KEDF contribution to vacancy formation", "Delta KEDF_vac (eV)", "cividis", None),
    ]

    for name, grid, title, label, cmap, point in specs:
        fig, ax = plt.subplots(figsize=(4.6, 4.1), dpi=220)
        plot_cell_map(ax, grid, title, label, cmap, marker=point)
        fig.tight_layout()
        png = outdir / f"fig_{name}_{tag}.png"
        pdf = outdir / f"fig_{name}_{tag}.pdf"
        fig.savefig(png, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        figs[f"{name}_png"] = str(png)
        figs[f"{name}_pdf"] = str(pdf)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), dpi=220)
    for ax, (_name, grid, title, label, cmap, point) in zip(axes, specs):
        plot_cell_map(ax, grid, title, label, cmap, marker=point)
    fig.suptitle(f"DFTpy TFvW lambda-mu scan ({tag.replace('_', ' ')})", fontsize=12, y=1.02)
    fig.tight_layout()
    combined_png = outdir / f"professor_three_maps_{tag}.png"
    combined_pdf = outdir / f"professor_three_maps_{tag}.pdf"
    fig.savefig(combined_png, bbox_inches="tight")
    fig.savefig(combined_pdf, bbox_inches="tight")
    plt.close(fig)
    figs["combined_png"] = str(combined_png)
    figs["combined_pdf"] = str(combined_pdf)
    return figs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", default=None, help="coarse_10x10_vacancy_formation root")
    parser.add_argument("--flat-csv", default=None)
    parser.add_argument("--audit-csv", default=None)
    parser.add_argument("--outdir", default="outputs/vaclm_three_maps_20260624")
    parser.add_argument("--qe-ref", type=float, default=0.60, help="QE vacancy formation-energy reference for annotation")
    parser.add_argument("--n-pristine", type=int, default=108)
    parser.add_argument("--n-vacancy", type=int, default=107)
    args = parser.parse_args()

    root = safe_path(args.rootdir) if args.rootdir else find_default_root()
    if root is None:
        raise FileNotFoundError("Cannot find default VACLM package root. Pass --rootdir.")

    flat_csv = safe_path(args.flat_csv) if args.flat_csv else first_existing(
        [
            root / "10_professor_raw_log_table" / "professor_raw_output_flat_pristine_and_vacancy.csv",
            root / "10_professor_raw_log_table" / "professor_raw_log_flat_results.csv",
        ]
    )
    audit_csv = safe_path(args.audit_csv) if args.audit_csv else root / "07_audit" / "vaclm_matrix_status_live.csv"
    outdir = safe_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    merged = load_merged(flat_csv, audit_csv if audit_csv.exists() else None, args.n_pristine, args.n_vacancy)
    merged.to_csv(outdir / "merged_gillan_formation_from_raw_outputs.csv", index=False)

    qualified = merged[merged["has_plot_values"] & merged["qualified_bool"]].copy()
    all_numeric = merged[merged["has_plot_values"]].copy()

    q_figs = save_figures(qualified, outdir / "qualified_only", "qualified_only", args.qe_ref)
    all_figs = save_figures(all_numeric, outdir / "all_numeric_diagnostic", "all_numeric_diagnostic", args.qe_ref)

    summary = {
        "rootdir": str(root),
        "flat_csv": str(flat_csv),
        "audit_csv": str(audit_csv) if audit_csv.exists() else None,
        "formula_total": f"Ef_vac = E_vac(Al{args.n_vacancy}) - ({args.n_vacancy}/{args.n_pristine}) * E_pristine(Al{args.n_pristine})",
        "formula_kedf": f"dKEDF_vac = K_vac(Al{args.n_vacancy}) - ({args.n_vacancy}/{args.n_pristine}) * K_pristine(Al{args.n_pristine})",
        "lattice_constant_definition": "relaxed pristine 3x3x3 cell length divided by 3",
        "qe_reference_eV": args.qe_ref,
        "points_total": int(len(merged)),
        "points_all_numeric": int(len(all_numeric)),
        "points_qualified": int(len(qualified)),
        "qualified_figures": q_figs,
        "all_numeric_diagnostic_figures": all_figs,
    }
    (outdir / "THREE_MAPS_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
