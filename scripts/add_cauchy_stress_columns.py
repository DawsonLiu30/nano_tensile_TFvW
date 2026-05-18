from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _to_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return math.nan
    return float(value)


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.12f}"


def _detect_area_columns(row: dict[str, str]) -> tuple[str, str]:
    current_candidates = [
        "wire_area_current_model_A2",
        "wire_area_current_ellipse_A2",
        "wire_area_current_A2",
    ]
    ref_candidates = [
        "wire_area_ref_model_A2",
        "wire_area_ref_ellipse_A2",
        "wire_area_ref_A2",
    ]
    current = next((key for key in current_candidates if key in row), "")
    ref = next((key for key in ref_candidates if key in row), "")
    if not current or not ref:
        raise KeyError(
            "Missing wire area columns. Expected wire_area_current_ellipse_A2 "
            "and wire_area_ref_ellipse_A2, or equivalent *_A2 columns."
        )
    return current, ref


def add_cauchy_columns(rows: list[dict[str, str]], stress_sign: float) -> list[dict[str, str]]:
    area_current_key, area_ref_key = _detect_area_columns(rows[0])

    output: list[dict[str, str]] = []
    for row in rows:
        out = dict(row)
        strain = _to_float(row, "strain")
        sigma_cell = stress_sign * _to_float(row, "sigma_cell_zz_GPa")
        area_cell = _to_float(row, "cell_area_xy_A2")
        area_current = _to_float(row, area_current_key)
        area_ref = _to_float(row, area_ref_key)

        true_strain = math.log1p(strain) if strain > -1.0 else math.nan
        cauchy_wire = sigma_cell * area_cell / area_current
        nominal_wire = sigma_cell * area_cell / area_ref

        out["true_strain"] = _fmt(true_strain)
        out["cauchy_cell_zz_GPa"] = _fmt(sigma_cell)
        out["cauchy_wire_zz_GPa"] = _fmt(cauchy_wire)
        out["nominal_wire_zz_GPa"] = _fmt(nominal_wire)

        output.append(out)
    return output


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, rows: list[dict[str, str]], title: str) -> None:
    x = [100.0 * float(row["strain"]) for row in rows]
    cauchy = [float(row["cauchy_wire_zz_GPa"]) for row in rows]
    nominal = [float(row["nominal_wire_zz_GPa"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(x, cauchy, "-o", lw=2.1, ms=4.0, label="Cauchy wire stress, current area")
    ax.plot(x, nominal, "--s", lw=1.4, ms=3.2, color="0.45", label="Nominal wire stress, reference area")

    ax.axhline(0.0, color="0.35", lw=0.8)
    ax.set_xlabel("Engineering strain (%)")
    ax.set_ylabel("Stress (GPa)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append Cauchy/true-stress columns to a tensile summary.csv. "
            "For vacuum-padded nanowires, cauchy_wire_zz_GPa = "
            "sigma_cell_zz_GPa * A_cell / A_wire,current."
        )
    )
    parser.add_argument("--summary", required=True, help="Input tensile summary.csv.")
    parser.add_argument("--out-csv", default="", help="Output CSV path.")
    parser.add_argument("--plot", default="", help="Optional output PNG path.")
    parser.add_argument("--title", default="", help="Optional plot title.")
    parser.add_argument(
        "--stress-sign",
        type=float,
        default=1.0,
        help="Use -1 if the stress convention is opposite to positive tension.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = _resolve(args.summary)
    rows = _read_rows(summary)
    out_rows = add_cauchy_columns(rows, stress_sign=float(args.stress_sign))

    out_csv = _resolve(args.out_csv) if args.out_csv else summary.with_name("summary_with_cauchy_stress.csv")
    _write_csv(out_csv, out_rows)

    print(f"[cauchy] input : {summary}")
    print(f"[cauchy] output: {out_csv}")

    if args.plot:
        title = args.title.strip() or "Tensile Cauchy stress"
        plot_path = _resolve(args.plot)
        _plot(plot_path, out_rows, title=title)
        print(f"[cauchy] plot  : {plot_path}")


if __name__ == "__main__":
    main()
