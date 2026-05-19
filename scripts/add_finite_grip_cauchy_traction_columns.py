from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EV_A3_TO_GPA = 160.21766208


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _pick_column(columns: list[str], candidates: list[str], required: bool = True) -> str | None:
    lower_map = {col.lower(): col for col in columns}
    for key in candidates:
        if key in columns:
            return key
        if key.lower() in lower_map:
            return lower_map[key.lower()]
    if required:
        raise KeyError(
            "Could not find any of these columns: "
            + ", ".join(candidates)
        )
    return None


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _finite_first(series: pd.Series) -> float:
    values = _safe_numeric(series).to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan
    return float(values[0])


def _area_from_spans(df: pd.DataFrame, span_x_key: str, span_y_key: str, model: str) -> pd.Series:
    dx = _safe_numeric(df[span_x_key])
    dy = _safe_numeric(df[span_y_key])
    if model == "bbox":
        area = dx * dy
    else:
        area = math.pi * dx * dy * 0.25
    return area.where(area > 0.0)


def _detect_area_series(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.Series, pd.Series, str, str]:
    columns = list(df.columns)

    current_key = _pick_column(
        columns,
        [
            args.area_current_column,
            "A_current_A2",
            "area_current_A2",
            "wire_area_current_model_A2",
            "wire_area_current_ellipse_A2",
            "wire_area_current_A2",
            "wire_area_current_hull_A2",
            "wire_area_current_bbox_A2",
        ],
        required=False,
    )
    if current_key:
        area_current = _safe_numeric(df[current_key])
        current_source = current_key
    else:
        span_x_key = _pick_column(
            columns,
            ["wire_span_x_A", "span_x_A", "dx_current_A", "delta_x_current_A"],
            required=True,
        )
        span_y_key = _pick_column(
            columns,
            ["wire_span_y_A", "span_y_A", "dy_current_A", "delta_y_current_A"],
            required=True,
        )
        area_current = _area_from_spans(df, span_x_key, span_y_key, args.area_from_spans)
        current_source = f"{args.area_from_spans}({span_x_key},{span_y_key})"

    ref_key = _pick_column(
        columns,
        [
            args.area_ref_column,
            "A_ref_A2",
            "area_ref_A2",
            "wire_area_ref_model_A2",
            "wire_area_ref_ellipse_A2",
            "wire_area_ref_A2",
        ],
        required=False,
    )
    if ref_key:
        area_ref = _safe_numeric(df[ref_key])
        ref_source = ref_key
    else:
        span_x_ref_key = _pick_column(
            columns,
            ["wire_span_x_ref_A", "span_x_ref_A", "dx_ref_A", "delta_x_ref_A"],
            required=False,
        )
        span_y_ref_key = _pick_column(
            columns,
            ["wire_span_y_ref_A", "span_y_ref_A", "dy_ref_A", "delta_y_ref_A"],
            required=False,
        )
        if span_x_ref_key and span_y_ref_key:
            area_ref = _area_from_spans(df, span_x_ref_key, span_y_ref_key, args.area_from_spans)
            ref_source = f"{args.area_from_spans}({span_x_ref_key},{span_y_ref_key})"
        else:
            # A finite-grip nominal stress needs a fixed reference area. If the
            # CSV does not store it explicitly, use the first valid current area.
            area0 = _finite_first(area_current)
            area_ref = pd.Series(area0, index=df.index, dtype=float)
            ref_source = f"first finite {current_source}"

    return area_current, area_ref, current_source, ref_source


def _baseline_index(df: pd.DataFrame, baseline_cycle: float | None) -> int:
    if baseline_cycle is not None:
        cycle_key = _pick_column(list(df.columns), ["cycle", "step", "cycle_index"], required=False)
        if cycle_key:
            cycle = _safe_numeric(df[cycle_key])
            match = df.index[np.isclose(cycle.to_numpy(dtype=float), float(baseline_cycle), equal_nan=False)]
            if len(match) > 0:
                return int(match[0])
    return int(df.index[0])


def _choose_x(df: pd.DataFrame) -> tuple[pd.Series, str, float]:
    key = _pick_column(
        list(df.columns),
        ["strain", "engineering_strain", "eps", "epsilon", "cycle", "step"],
        required=False,
    )
    if key is None:
        return pd.Series(np.arange(len(df)), index=df.index, dtype=float), "row index", 1.0
    x = _safe_numeric(df[key])
    if key.lower() in {"strain", "engineering_strain", "eps", "epsilon"}:
        return x * 100.0, "Engineering strain (%)", 100.0
    return x, key, 1.0


def add_finite_grip_columns(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    columns = list(df.columns)
    top_force_key = _pick_column(
        columns,
        [
            args.top_force_column,
            "grip_force_top_z_eVA",
            "top_grip_force_z_eVA",
            "force_top_z_eVA",
            "F_top_z_eVA",
            "F_top_z_eV_A",
            "top_force_z_eVA",
        ],
    )
    bottom_force_key = _pick_column(
        columns,
        [
            args.bottom_force_column,
            "grip_force_bottom_z_eVA",
            "bottom_grip_force_z_eVA",
            "force_bottom_z_eVA",
            "F_bottom_z_eVA",
            "F_bottom_z_eV_A",
            "bottom_force_z_eVA",
        ],
    )

    area_current, area_ref, current_source, ref_source = _detect_area_series(df, args)
    top_force = _safe_numeric(df[top_force_key])
    bottom_force = _safe_numeric(df[bottom_force_key])

    out = df.copy()
    out["area_current_used_A2"] = area_current
    out["area_ref_used_A2"] = area_ref
    out["area_ratio_Aref_over_Acurrent"] = area_ref / area_current
    out["area_current_source"] = current_source
    out["area_ref_source"] = ref_source

    out["grip_nominal_top_GPa"] = args.top_sign * top_force / area_ref * EV_A3_TO_GPA
    out["grip_nominal_bottom_GPa"] = args.bottom_sign * bottom_force / area_ref * EV_A3_TO_GPA
    out["grip_nominal_raw_GPa"] = 0.5 * (
        out["grip_nominal_top_GPa"] + out["grip_nominal_bottom_GPa"]
    )

    out["grip_apparent_cauchy_top_GPa"] = args.top_sign * top_force / area_current * EV_A3_TO_GPA
    out["grip_apparent_cauchy_bottom_GPa"] = (
        args.bottom_sign * bottom_force / area_current * EV_A3_TO_GPA
    )
    out["grip_apparent_cauchy_raw_GPa"] = 0.5 * (
        out["grip_apparent_cauchy_top_GPa"] + out["grip_apparent_cauchy_bottom_GPa"]
    )

    baseline_idx = _baseline_index(out, args.baseline_cycle)
    out["grip_nominal_primary_GPa"] = (
        out["grip_nominal_raw_GPa"] - out.loc[baseline_idx, "grip_nominal_raw_GPa"]
    )
    out["grip_apparent_cauchy_primary_GPa"] = (
        out["grip_apparent_cauchy_raw_GPa"]
        - out.loc[baseline_idx, "grip_apparent_cauchy_raw_GPa"]
    )

    avg_mag = 0.5 * (
        out["grip_nominal_top_GPa"].abs() + out["grip_nominal_bottom_GPa"].abs()
    )
    out["grip_top_bottom_balance_rel"] = (
        (out["grip_nominal_top_GPa"] - out["grip_nominal_bottom_GPa"]).abs()
        / avg_mag.replace(0.0, np.nan)
    )
    out["grip_top_bottom_balance_signed_GPa"] = (
        out["grip_nominal_top_GPa"] - out["grip_nominal_bottom_GPa"]
    )

    sigma_cell_key = _pick_column(columns, ["sigma_cell_zz_GPa", "cauchy_cell_zz_GPa"], required=False)
    cell_area_key = _pick_column(columns, ["cell_area_xy_A2", "A_cell_A2", "area_cell_A2"], required=False)
    if sigma_cell_key and cell_area_key:
        sigma_cell = _safe_numeric(df[sigma_cell_key])
        area_cell = _safe_numeric(df[cell_area_key])
        out["diagnostic_cell_cauchy_wire_current_GPa"] = sigma_cell * area_cell / area_current
        out["diagnostic_cell_nominal_wire_ref_GPa"] = sigma_cell * area_cell / area_ref

    out["stress_definition_note"] = (
        "primary=preload-corrected nominal grip-reaction stress; "
        "apparent_cauchy=current-area correction used only as a sensitivity check; "
        "cell stress is diagnostic because vacuum dilutes the simulation-cell tensor"
    )
    return out


def _plot(out_png: Path, df: pd.DataFrame, title: str) -> None:
    x, xlabel, _ = _choose_x(df)
    fig, (ax_stress, ax_balance) = plt.subplots(
        2,
        1,
        figsize=(7.8, 6.6),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.35]},
    )

    ax_stress.plot(
        x,
        df["grip_nominal_primary_GPa"],
        "-o",
        lw=2.0,
        ms=4.0,
        label="Nominal grip-reaction stress (primary)",
    )
    ax_stress.plot(
        x,
        df["grip_apparent_cauchy_primary_GPa"],
        "--s",
        lw=1.6,
        ms=3.5,
        label="Current-area apparent Cauchy stress",
    )
    ax_stress.axhline(0.0, color="0.35", lw=0.8)
    ax_stress.set_ylabel("Stress (GPa)")
    ax_stress.set_title(title)
    ax_stress.grid(True, alpha=0.25)
    ax_stress.legend(frameon=True)

    ax_balance.plot(
        x,
        df["grip_top_bottom_balance_rel"],
        "-o",
        lw=1.4,
        ms=3.2,
        color="0.25",
    )
    ax_balance.axhline(0.1, color="0.65", lw=0.8, ls=":")
    ax_balance.set_xlabel(xlabel)
    ax_balance.set_ylabel("Top-bottom\nimbalance")
    ax_balance.grid(True, alpha=0.25)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add finite-grip Cauchy-traction-compatible stress columns to a tensile summary. "
            "The primary curve remains preload-corrected nominal grip-reaction stress "
            "Fz/A_ref. The current-area apparent Cauchy stress Fz/A_current is added as "
            "a sensitivity check, not as a replacement for the primary curve."
        )
    )
    parser.add_argument("--summary", required=True, help="Input finite-grip tensile CSV/TSV.")
    parser.add_argument("--out-csv", default="", help="Output CSV path.")
    parser.add_argument("--plot", default="", help="Optional output PNG path.")
    parser.add_argument("--title", default="Finite-grip tensile stress definition check")
    parser.add_argument("--top-force-column", default="", help="Override top z-force column.")
    parser.add_argument("--bottom-force-column", default="", help="Override bottom z-force column.")
    parser.add_argument("--area-current-column", default="", help="Override current area column.")
    parser.add_argument("--area-ref-column", default="", help="Override fixed reference area column.")
    parser.add_argument(
        "--area-from-spans",
        choices=["ellipse", "bbox"],
        default="ellipse",
        help="Area model if only x/y spans are available.",
    )
    parser.add_argument(
        "--baseline-cycle",
        type=float,
        default=0.0,
        help="Cycle/step used for preload correction when a cycle column exists.",
    )
    parser.add_argument("--top-sign", type=float, default=-1.0, help="Sign multiplier for top z-force.")
    parser.add_argument(
        "--bottom-sign",
        type=float,
        default=1.0,
        help="Sign multiplier for bottom z-force.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = _resolve(args.summary)
    if not summary.exists():
        raise FileNotFoundError(summary)

    df = pd.read_csv(summary, sep=None, engine="python")
    if df.empty:
        raise ValueError(f"No rows found in {summary}")

    out = add_finite_grip_columns(df, args)
    out_csv = _resolve(args.out_csv) if args.out_csv else summary.with_name("summary_with_grip_cauchy_traction.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    print(f"[finite-grip] input : {summary}")
    print(f"[finite-grip] output: {out_csv}")
    print("[finite-grip] primary stress: grip_nominal_primary_GPa")
    print("[finite-grip] sensitivity : grip_apparent_cauchy_primary_GPa")
    print("[finite-grip] balance     : grip_top_bottom_balance_rel")

    if args.plot:
        plot_path = _resolve(args.plot)
        _plot(plot_path, out, title=args.title)
        print(f"[finite-grip] plot  : {plot_path}")


if __name__ == "__main__":
    main()
