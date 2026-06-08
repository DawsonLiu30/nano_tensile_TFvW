#!/usr/bin/env python3
"""Organize and summarize the DFTpy TFvW fine y-weight scan."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_TARGET_EF_EV = 0.601167
DEFAULT_FORCE_TARGET_EV_A = 0.002


def last_fmax(log_path: Path) -> float:
    if not log_path.exists():
        return math.nan
    vals: list[float] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("step"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                vals.append(float(parts[-1]))
            except ValueError:
                pass
    return vals[-1] if vals else math.nan


def build_summary_from_results(root: Path) -> pd.DataFrame:
    rows = []
    for result_path in sorted((root / "weight_scan").glob("*/result.json")):
        data = json.loads(result_path.read_text(encoding="utf-8"))
        case_dir = result_path.parent
        rows.append(
            {
                "kedf_y": float(data["kedf_y"]),
                "label": case_dir.name,
                "Ef_vac_eV": float(data["vacancy_formation_energy_eV"]),
                "pristine_final_fmax_eV_A": last_fmax(case_dir / "pristine_relax.log"),
                "vacancy_final_fmax_eV_A": last_fmax(case_dir / "vacancy_relax.log"),
                "pristine_energy_eV": float(data["pristine_energy_eV"]),
                "vacancy_energy_eV": float(data["vacancy_energy_eV"]),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No result.json files found under {root / 'weight_scan'}")
    return pd.DataFrame(rows).sort_values("kedf_y").reset_index(drop=True)


def load_or_build_summary(root: Path) -> pd.DataFrame:
    preferred = root / "fine_weight_scan_summary_with_actual_fmax.csv"
    if preferred.exists():
        df = pd.read_csv(preferred)
    else:
        df = build_summary_from_results(root)
        df.to_csv(preferred, index=False)
    return df.sort_values("kedf_y").reset_index(drop=True)


def interpolate_target(df: pd.DataFrame, target: float) -> dict[str, float | str]:
    rows = df.sort_values("kedf_y").to_dict("records")
    for left, right in zip(rows, rows[1:]):
        e1 = float(left["Ef_vac_eV"])
        e2 = float(right["Ef_vac_eV"])
        if (e1 <= target <= e2) or (e2 <= target <= e1):
            y1 = float(left["kedf_y"])
            y2 = float(right["kedf_y"])
            if abs(e2 - e1) < 1e-15:
                y_interp = math.nan
            else:
                y_interp = y1 + (target - e1) * (y2 - y1) / (e2 - e1)
            return {
                "y_interp": y_interp,
                "left_y": y1,
                "right_y": y2,
                "left_Ef_eV": e1,
                "right_Ef_eV": e2,
                "status": "bracketed",
            }
    return {
        "y_interp": math.nan,
        "left_y": math.nan,
        "right_y": math.nan,
        "left_Ef_eV": math.nan,
        "right_Ef_eV": math.nan,
        "status": "not_bracketed",
    }


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_selected_points(root: Path, outdir: Path, df: pd.DataFrame, selected_y: list[float]) -> None:
    selected_root = outdir / "selected_points"
    for y in selected_y:
        sub = df.iloc[(df["kedf_y"] - y).abs().argsort()[:1]]
        if sub.empty:
            continue
        row = sub.iloc[0]
        if abs(float(row["kedf_y"]) - y) > 1e-9:
            continue
        label = str(row["label"])
        case_dir = root / "weight_scan" / label
        point_dir = selected_root / f"y_{str(y).replace('.', 'p')}_{label}"
        for name in [
            "point_manifest.json",
            "result.json",
            "dftpy_pristine_input.ini",
            "dftpy_vacancy_input.ini",
            "pristine_raw.vasp",
            "vacancy_start.vasp",
            "pristine_relaxed.vasp",
            "vacancy_relaxed.vasp",
            "pristine_vc_relaxed.vasp",
            "vacancy_vc_relaxed.vasp",
            "pristine_relax.log",
            "vacancy_relax.log",
        ]:
            copy_if_exists(case_dir / name, point_dir / name)


def write_markdown(
    path: Path,
    *,
    root: Path,
    df: pd.DataFrame,
    target: float,
    force_target: float,
    best_all: pd.Series,
    best_pass: pd.Series,
    interp: dict[str, float | str],
) -> None:
    pass_count = int(df["force_pass"].sum())
    total_count = int(len(df))
    monotonic = bool((df["Ef_vac_eV"].diff().dropna() > 0).all())
    caution = df[~df["force_pass"]]

    lines = [
        "# DFTpy TFvW Fine Weight Scan Summary",
        "",
        f"Source raw folder: `{root}`",
        "",
        "## Setup",
        "",
        "- System: conventional fcc Al 3x3x3 vacancy cell",
        "- Pristine/vacancy atoms: 108 -> 107",
        "- XC: LDA",
        "- Pseudopotential: al.lda.recpot",
        "- KEDF: TFVW",
        "- Fixed TF weight: x = 1.0",
        "- Spacing: 0.20 A",
        f"- Reference target Ef: {target:.6f} eV",
        f"- Force target: fmax < {force_target:.6f} eV/A",
        "",
        "## Main Findings",
        "",
        f"- Completed points: {total_count}",
        f"- Force-passing points: {pass_count}/{total_count}",
        f"- Ef increases monotonically with y: {'yes' if monotonic else 'no'}",
        (
            f"- Best sampled point overall: y = {float(best_all['kedf_y']):.6f}, "
            f"Ef = {float(best_all['Ef_vac_eV']):.6f} eV, "
            f"diff = {float(best_all['diff_to_target_eV']):+.6f} eV"
        ),
        (
            f"- Best force-passing point: y = {float(best_pass['kedf_y']):.6f}, "
            f"Ef = {float(best_pass['Ef_vac_eV']):.6f} eV, "
            f"diff = {float(best_pass['diff_to_target_eV']):+.6f} eV"
        ),
        (
            f"- Linear interpolation: y = {float(interp['y_interp']):.6f} "
            f"between y = {float(interp['left_y']):.6f} and {float(interp['right_y']):.6f}"
            if interp["status"] == "bracketed"
            else "- Linear interpolation: target not bracketed"
        ),
        "",
        "## Recommended Production Values",
        "",
        "- Primary calibrated value: `x = 1.0, y = 0.130`",
        "- Robust production candidates near QE/literature: `y = 0.140` and `y = 0.145`",
        "- Avoid using force-failed diagnostic points as final calibration values without rerun.",
        "",
    ]
    if not caution.empty:
        lines.extend(["## Force Caution Points", ""])
        for _, row in caution.iterrows():
            lines.append(
                f"- y = {float(row['kedf_y']):.3f}: "
                f"pristine fmax = {float(row['pristine_final_fmax_eV_A']):.6f}, "
                f"vacancy fmax = {float(row['vacancy_final_fmax_eV_A']):.6f} eV/A"
            )
        lines.append("")

    lines.extend(["## Clean Table", ""])
    lines.append("| y | Ef_vac (eV) | diff to target (eV) | pristine fmax | vacancy fmax | force pass |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    for _, row in df.iterrows():
        lines.append(
            f"| {float(row['kedf_y']):.3f} | {float(row['Ef_vac_eV']):.6f} | "
            f"{float(row['diff_to_target_eV']):+.6f} | "
            f"{float(row['pristine_final_fmax_eV_A']):.6f} | "
            f"{float(row['vacancy_final_fmax_eV_A']):.6f} | "
            f"{'yes' if bool(row['force_pass']) else 'no'} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_scan(df: pd.DataFrame, target: float, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    passed = df[df["force_pass"]]
    failed = df[~df["force_pass"]]
    ax.plot(df["kedf_y"], df["Ef_vac_eV"], color="#23395d", linewidth=1.8, alpha=0.75)
    ax.scatter(passed["kedf_y"], passed["Ef_vac_eV"], color="#218c5a", label="force pass", zorder=3)
    if not failed.empty:
        ax.scatter(failed["kedf_y"], failed["Ef_vac_eV"], color="#c23b22", label="force caution", zorder=4)
    ax.axhline(target, color="#444444", linestyle="--", linewidth=1.2, label=f"QE target {target:.3f} eV")
    ax.set_xlabel("TFvW von Weizsaecker weight y")
    ax.set_ylabel("Vacancy formation energy (eV)")
    ax.set_title("DFTpy/LDA TFvW fine y-scan for Al 3x3x3 vacancy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=250)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rootdir", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--target-ef", type=float, default=DEFAULT_TARGET_EF_EV)
    ap.add_argument("--force-target", type=float, default=DEFAULT_FORCE_TARGET_EV_A)
    args = ap.parse_args()

    root = args.rootdir.resolve()
    outdir = args.outdir.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")
    outdir.mkdir(parents=True, exist_ok=True)
    for sub in ["tables", "figures", "notes", "selected_points"]:
        (outdir / sub).mkdir(exist_ok=True)

    df = load_or_build_summary(root)
    df["pristine_force_pass"] = df["pristine_final_fmax_eV_A"] <= args.force_target
    df["vacancy_force_pass"] = df["vacancy_final_fmax_eV_A"] <= args.force_target
    df["force_pass"] = df["pristine_force_pass"] & df["vacancy_force_pass"]
    df["diff_to_target_eV"] = df["Ef_vac_eV"] - args.target_ef
    df["abs_diff_to_target_eV"] = df["diff_to_target_eV"].abs()
    df = df.sort_values("kedf_y").reset_index(drop=True)

    best_all = df.sort_values("abs_diff_to_target_eV").iloc[0]
    pass_df = df[df["force_pass"]].copy()
    best_pass = pass_df.sort_values("abs_diff_to_target_eV").iloc[0] if not pass_df.empty else best_all
    interp = interpolate_target(df, args.target_ef)

    clean_cols = [
        "kedf_y",
        "label",
        "Ef_vac_eV",
        "diff_to_target_eV",
        "abs_diff_to_target_eV",
        "pristine_final_fmax_eV_A",
        "vacancy_final_fmax_eV_A",
        "force_pass",
        "pristine_energy_eV",
        "vacancy_energy_eV",
    ]
    df[clean_cols].to_csv(outdir / "tables" / "tfvw_fine_scan_clean_summary.csv", index=False)
    pass_df[clean_cols].to_csv(outdir / "tables" / "tfvw_fine_scan_force_pass_points.csv", index=False)

    recommendation_rows = [
        {
            "category": "best_sampled_overall",
            "kedf_y": float(best_all["kedf_y"]),
            "Ef_vac_eV": float(best_all["Ef_vac_eV"]),
            "diff_to_target_eV": float(best_all["diff_to_target_eV"]),
            "force_pass": bool(best_all["force_pass"]),
        },
        {
            "category": "best_force_passing",
            "kedf_y": float(best_pass["kedf_y"]),
            "Ef_vac_eV": float(best_pass["Ef_vac_eV"]),
            "diff_to_target_eV": float(best_pass["diff_to_target_eV"]),
            "force_pass": bool(best_pass["force_pass"]),
        },
        {
            "category": "linear_interpolated_target",
            "kedf_y": float(interp["y_interp"]),
            "Ef_vac_eV": args.target_ef,
            "diff_to_target_eV": 0.0,
            "force_pass": "",
        },
    ]
    with (outdir / "tables" / "tfvw_fine_scan_recommendations.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(recommendation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(recommendation_rows)

    write_markdown(
        outdir / "notes" / "TFVW_FINE_SCAN_ANALYSIS.md",
        root=root,
        df=df,
        target=args.target_ef,
        force_target=args.force_target,
        best_all=best_all,
        best_pass=best_pass,
        interp=interp,
    )
    plot_scan(df, args.target_ef, outdir / "figures" / "tfvw_fine_scan_Ef_vs_y.png")

    for name in [
        "fine_weight_scan_summary_with_actual_fmax.csv",
        "final_fine_weight_scan_summary_with_actual_fmax_20260606.csv",
        "FINAL_NOTE_TFVW_WEIGHT_FINE_SCAN_20260606.txt",
        "dftpy_conventional_weight_summary.csv",
        "dftpy_weight_actual_fmax_summary.csv",
        "dftpy_conventional_weight_Ef.png",
    ]:
        copy_if_exists(root / name, outdir / "source_summaries" / name)

    copy_selected_points(root, outdir, df, selected_y=[0.13, 0.14, 0.145])

    readme = outdir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Organized DFTpy TFvW Fine Scan Package",
                "",
                f"Raw data source: `{root}`",
                "",
                "Main files:",
                "",
                "- `tables/tfvw_fine_scan_clean_summary.csv`",
                "- `tables/tfvw_fine_scan_recommendations.csv`",
                "- `figures/tfvw_fine_scan_Ef_vs_y.png`",
                "- `notes/TFVW_FINE_SCAN_ANALYSIS.md`",
                "- `selected_points/` contains key y=0.130, 0.140, 0.145 structures/logs.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("============================================================")
    print("TFvW fine scan organization completed")
    print("============================================================")
    print(f"Raw root : {root}")
    print(f"Package  : {outdir}")
    print(f"Best y   : {float(best_pass['kedf_y']):.6f}")
    print(f"Best Ef  : {float(best_pass['Ef_vac_eV']):.6f} eV")
    if interp["status"] == "bracketed":
        print(f"Interp y : {float(interp['y_interp']):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

