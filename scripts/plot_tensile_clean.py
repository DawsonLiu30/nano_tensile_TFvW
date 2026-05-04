from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _label_from_result_dir(result_dir: Path) -> str:
    case_dir = result_dir.parent.parent if result_dir.parent.name == "results" else result_dir
    match = re.search(r"_(\d+(?:\.\d+)?)nm", case_dir.name)
    if not match:
        return case_dir.name
    diameter = match.group(1)
    return f"d = {diameter} nm"


def _load_result(result_dir: Path, label: str | None) -> dict:
    events_path = result_dir / "tensile_events.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing tensile_events.csv; run analyze_tensile_events.py first: {result_dir}")
    events = _read_csv(events_path)
    if not events:
        raise ValueError(f"No rows in {events_path}")
    summary = _read_json(result_dir / "tensile_event_summary.json")
    return {
        "result_dir": result_dir,
        "label": label or _label_from_result_dir(result_dir),
        "events": events,
        "summary": summary,
        "cycle": np.asarray([int(row["cycle"]) for row in events], dtype=int),
        "strain": np.asarray([float(row["strain_pct"]) for row in events], dtype=float),
        "stress": np.asarray([float(row["sigma_primary_grip_offset_GPa"]) for row in events], dtype=float),
    }


def _idx_for_cycle(data: dict, cycle: int | None) -> int | None:
    if cycle is None:
        return None
    matches = np.where(data["cycle"] == int(cycle))[0]
    return int(matches[0]) if matches.size else None


def _event_indices(data: dict) -> dict[str, int | None]:
    summary = data["summary"]
    return {
        "peak": _idx_for_cycle(data, summary.get("yield_strength_candidate_cycle")),
        "plastic": _idx_for_cycle(data, summary.get("plastic_onset_cycle")),
        "fracture": _idx_for_cycle(data, summary.get("complete_fracture_cycle")),
    }


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13.5,
            "axes.labelsize": 11.5,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _comparison_colors(n: int) -> list[str]:
    base = [
        "#1f77b4",  # r1 blue
        "#d95f02",  # r2 orange
        "#2ca02c",  # r3 green
        "#9467bd",  # r4 purple
        "#8c564b",  # r5 brown
        "#e377c2",  # r6 pink
        "#17becf",  # spare cyan
        "#bcbd22",  # spare olive
    ]
    if n <= len(base):
        return base[:n]
    return [base[i % len(base)] for i in range(n)]


def _plot_events(ax, data: dict, *, color: str, include_event_labels: bool, shade: bool) -> None:
    idx = _event_indices(data)
    strain = data["strain"]
    stress = data["stress"]

    if shade and idx["plastic"] is not None:
        ax.axvspan(strain[idx["plastic"]], strain[-1], color="#7a5195", alpha=0.055, lw=0, zorder=0)
        ax.axvline(strain[idx["plastic"]], color="#7a5195", lw=1.05, ls="--", alpha=0.78, zorder=1)

    if idx["peak"] is not None:
        ax.scatter(
            [strain[idx["peak"]]],
            [stress[idx["peak"]]],
            marker="D",
            s=78,
            facecolor="#b23a1b",
            edgecolor="black",
            linewidth=0.65,
            zorder=5,
            label="Peak load / elastic limit" if include_event_labels else None,
        )

    if idx["plastic"] is not None:
        ax.scatter(
            [strain[idx["plastic"]]],
            [stress[idx["plastic"]]],
            marker="P",
            s=84,
            facecolor="#7a5195",
            edgecolor="black",
            linewidth=0.55,
            zorder=5,
            label="Plastic onset" if include_event_labels else None,
        )

    if idx["fracture"] is not None:
        ax.scatter(
            [strain[idx["fracture"]]],
            [stress[idx["fracture"]]],
            marker="X",
            s=88,
            facecolor="#111111",
            edgecolor="#111111",
            linewidth=0.55,
            zorder=6,
            label="Complete fracture" if include_event_labels else None,
        )

    ax.scatter(
        [strain[-1]],
        [stress[-1]],
        s=80,
        facecolor="white",
        edgecolor=color,
        linewidth=2.1,
        zorder=5,
        label="Latest completed" if include_event_labels else None,
    )


def plot_single(data: dict, out: Path, title: str, *, color: str, shade: bool, pdf: bool, hide_events: bool) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.7), constrained_layout=True)
    ax.plot(data["strain"], data["stress"], "-o", color=color, lw=2.2, ms=4.0, label="Primary stress")
    if not hide_events:
        _plot_events(ax, data, color=color, include_event_labels=True, shade=shade)
    ax.axhline(0.0, color="0.45", lw=0.8)
    ax.set_xlabel("Engineering strain from grip separation (%)")
    ax.set_ylabel("Offset grip-reaction stress (GPa)")
    ax.set_title(title)
    ax.grid(True, alpha=0.23)
    ax.legend(loc="best", frameon=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    if pdf:
        fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def plot_comparison(data_list: list[dict], out: Path, title: str, *, shade: bool, pdf: bool, hide_events: bool) -> None:
    n_series = len(data_list)
    fig_width = 8.6 if n_series >= 5 else 7.4
    fig, ax = plt.subplots(figsize=(fig_width, 5.1), constrained_layout=True)
    colors = _comparison_colors(n_series)
    for i, data in enumerate(data_list):
        color = colors[i]
        ax.plot(data["strain"], data["stress"], "-o", color=color, lw=2.15, ms=3.9, label=data["label"])
        if not hide_events:
            _plot_events(ax, data, color=color, include_event_labels=False, shade=shade)
    ax.axhline(0.0, color="0.45", lw=0.8)
    ax.set_xlabel("Engineering strain from grip separation (%)")
    ax.set_ylabel("Offset grip-reaction stress (GPa)")
    ax.set_title(title)
    ax.grid(True, alpha=0.23)
    ax.legend(loc="best", frameon=True, ncol=2 if n_series >= 5 else 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    if pdf:
        fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean proposal-ready finite-grip stress-strain plots.")
    parser.add_argument("--results-dir", action="append", required=True)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--color", default="#1f77b4", help="Primary curve color for a single-result plot.")
    parser.add_argument("--no-shade", action="store_true")
    parser.add_argument("--hide-events", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_style()
    labels = list(args.label)
    data_list = []
    for idx, result_text in enumerate(args.results_dir):
        data_list.append(_load_result(_resolve(result_text), labels[idx] if idx < len(labels) else None))
    out = _resolve(args.out)
    title = args.title.strip()
    if len(data_list) == 1:
        if not title:
            title = f"Finite-grip vacancy tensile, {data_list[0]['label']}"
        plot_single(
            data_list[0],
            out,
            title,
            color=str(args.color),
            shade=not args.no_shade,
            pdf=bool(args.pdf),
            hide_events=bool(args.hide_events),
        )
    else:
        if not title:
            title = "Finite-grip vacancy tensile response"
        plot_comparison(
            data_list,
            out,
            title,
            shade=not args.no_shade,
            pdf=bool(args.pdf),
            hide_events=bool(args.hide_events),
        )
    print(f"[clean-plot] output: {out}")
    if args.pdf:
        print(f"[clean-plot] output: {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
