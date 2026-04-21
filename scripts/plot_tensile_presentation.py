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
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _label_from_result_dir(result_dir: Path) -> str:
    case_dir = result_dir.parent.parent if result_dir.parent.name == "results" else result_dir
    match = re.search(r"_(\d+(?:\.\d+)?)nm", case_dir.name)
    if match:
        d = match.group(1)
        r = d.rstrip("0").rstrip(".")
        return f"r={r}, d={d} nm"
    return case_dir.name


def _load_result(result_dir: Path, label: str | None = None) -> dict:
    events = _read_csv(result_dir / "tensile_events.csv")
    if not events:
        raise FileNotFoundError(f"tensile_events.csv not found or empty: {result_dir}")
    summary = _read_json(result_dir / "tensile_event_summary.json")
    cycles = np.asarray([int(row["cycle"]) for row in events], dtype=int)
    tags = [{part for part in row.get("event_tags", "").split(";") if part} for row in events]
    return {
        "result_dir": result_dir,
        "label": label or _label_from_result_dir(result_dir),
        "events": events,
        "summary": summary,
        "cycles": cycles,
        "strain": np.asarray([float(row["strain_pct"]) for row in events], dtype=float),
        "primary": np.asarray([float(row["sigma_primary_grip_offset_GPa"]) for row in events], dtype=float),
        "diagnostic": np.asarray([float(row["sigma_cell_wire_GPa"]) for row in events], dtype=float),
        "diagnostic_delta": np.asarray([float(row["diagnostic_delta_abs_GPa"]) for row in events], dtype=float),
        "tags": tags,
    }


def _idx_for_tag(data: dict, tag: str) -> list[int]:
    return [i for i, tags in enumerate(data["tags"]) if tag in tags]


def _idx_for_cycle(data: dict, cycle: int | None) -> int | None:
    if cycle is None:
        return None
    matches = np.where(data["cycles"] == int(cycle))[0]
    return int(matches[0]) if matches.size else None


def _key_anomaly_indices(data: dict, limit: int = 2) -> list[int]:
    anomaly_idx = _idx_for_tag(data, "diagnostic_anomaly")
    if not anomaly_idx:
        return []
    groups: list[list[int]] = []
    current: list[int] = [anomaly_idx[0]]
    for idx in anomaly_idx[1:]:
        if int(data["cycles"][idx]) == int(data["cycles"][current[-1]]) + 1:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)
    chosen: list[int] = []
    for group in groups:
        best = max(group, key=lambda i: float(data["diagnostic_delta"][i]))
        chosen.append(int(best))
    if len(chosen) > limit:
        chosen = sorted(chosen, key=lambda i: float(data["diagnostic_delta"][i]), reverse=True)[:limit]
        chosen = sorted(chosen)
    return chosen


def _fmt_optional(value, fmt: str) -> str:
    if value is None:
        return "not detected"
    return format(value, fmt)


def _summary_lines(data_list: list[dict], *, comparison: bool) -> list[str]:
    lines = [
        "Event definitions",
        "Primary stress = grip reaction - cycle-0 preload",
        "Peak load / elastic limit = last robust load peak",
        "before structural plasticity begins.",
        "Plastic onset = first structural rearrangement",
        "from bond-change / non-affine proxy.",
        "Final fracture = gap > 3*d111 or major cluster.",
        "",
        "Event summary",
    ]
    for data in data_list:
        s = data["summary"]
        label = data["label"]
        if comparison:
            lines.append(f"{label}:")
        strength_cycle = s.get("yield_strength_candidate_cycle")
        strength_strain = s.get("yield_strength_candidate_strain_pct")
        strength_stress = s.get("yield_strength_candidate_primary_stress_GPa")
        plastic_cycle = s.get("plastic_onset_cycle")
        plastic_strain = s.get("plastic_onset_strain_pct")
        latest_cycle = s.get("latest_cycle")
        latest_strain = s.get("latest_strain_pct")
        latest_stress = s.get("latest_primary_stress_GPa")
        fracture_cycle = s.get("complete_fracture_cycle")
        lines.extend(
            [
                (
                    f"Peak load / elastic limit: C{strength_cycle}, "
                    f"{strength_strain:.2f}%, {strength_stress:.2f} GPa"
                    if strength_cycle is not None
                    else "Peak load / elastic limit: not detected"
                ),
                (
                    f"Plastic onset: C{plastic_cycle}, {plastic_strain:.2f}%"
                    if plastic_cycle is not None
                    else "Plastic onset: not detected"
                ),
                (
                    f"Latest completed: C{latest_cycle}, {latest_strain:.2f}%, {latest_stress:.2f} GPa"
                    if latest_cycle is not None
                    else "Latest completed: unavailable"
                ),
                f"Complete fracture: {fracture_cycle if fracture_cycle is not None else 'not reached'}",
            ]
        )
        if not comparison and fracture_cycle is None:
            lines.extend(["", "Current completed range only;", "final fracture has not been reached."])
        lines.append("")
    return lines[:-1] if lines and lines[-1] == "" else lines


def _plot_common_events(
    ax,
    data: dict,
    color: str,
    *,
    show_anomalies: bool,
    annotate: bool,
    include_labels: bool = False,
) -> None:
    strain = data["strain"]
    primary = data["primary"]
    summary = data["summary"]
    strength_i = _idx_for_cycle(data, summary.get("yield_strength_candidate_cycle"))
    plastic_i = _idx_for_cycle(data, summary.get("plastic_onset_cycle"))
    fracture_i = _idx_for_cycle(data, summary.get("complete_fracture_cycle"))

    if plastic_i is not None:
        ax.axvline(strain[plastic_i], color="#7a5195", lw=1.25, ls="--", alpha=0.92)
        ax.axvspan(strain[plastic_i], strain[-1], color="#7a5195", alpha=0.075, lw=0)
        ax.scatter(
            [strain[plastic_i]],
            [primary[plastic_i]],
            s=116,
            marker="P",
            color="#7a5195",
            edgecolor="black",
            lw=0.6,
            zorder=7,
            label="Plastic onset" if include_labels else None,
        )
        if annotate:
            ax.annotate(
                f"Plastic onset\nC{int(data['cycles'][plastic_i])}, {strain[plastic_i]:.2f}%",
                xy=(strain[plastic_i], primary[plastic_i]),
                xytext=(18, -48),
                textcoords="offset points",
                fontsize=9.2,
                arrowprops=dict(arrowstyle="->", color="#7a5195", lw=0.9),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#7a5195", alpha=0.92),
            )
        y_top = float(np.nanmax(primary))
        x_mid = 0.5 * (strain[plastic_i] + strain[-1])
        ax.text(
            x_mid,
            y_top * 0.93 if y_top > 0 else y_top,
            "Post-yield regime",
            color="#7a5195",
            fontsize=9.4,
            ha="center",
            va="top",
            alpha=0.72,
        )

    if strength_i is not None:
        ax.scatter(
            [strain[strength_i]],
            [primary[strength_i]],
            marker="D",
            s=118,
            color="#b23a1b",
            edgecolor="black",
            lw=0.7,
            zorder=8,
            label="Peak load / elastic limit" if include_labels else None,
        )
        if annotate:
            y_offset = -64 if primary[strength_i] > 0.72 * np.nanmax(primary) else 18
            ax.annotate(
                f"Peak load /\nelastic limit\n{primary[strength_i]:.2f} GPa",
                xy=(strain[strength_i], primary[strength_i]),
                xytext=(14, y_offset),
                textcoords="offset points",
                fontsize=9.2,
                arrowprops=dict(arrowstyle="->", color="#b23a1b", lw=0.9),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#b23a1b", alpha=0.92),
            )

    if show_anomalies:
        anomaly_idx = _key_anomaly_indices(data, limit=2)
        if anomaly_idx:
            ax.scatter(
                strain[anomaly_idx],
                primary[anomaly_idx],
                marker="^",
                s=92,
                color="#d8a31a",
                edgecolor="black",
                lw=0.55,
                zorder=7,
                label="Key diagnostic anomaly",
            )

    if fracture_i is not None:
        ax.axvline(strain[fracture_i], color="#111111", lw=1.1, ls="--", alpha=0.95)
        ax.scatter([strain[fracture_i]], [primary[fracture_i]], marker="X", s=118, color="#111111", zorder=9)

    ax.scatter(
        [strain[-1]],
        [primary[-1]],
        s=102,
        facecolor="white",
        edgecolor=color,
        linewidth=2.4,
        zorder=8,
        label="Latest completed" if include_labels else None,
    )


def plot_single(data: dict, out: Path, *, show_diagnostic: bool, show_anomalies: bool, title: str | None) -> None:
    fig = plt.figure(figsize=(12.2, 6.7), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.25, 1.25])
    ax = fig.add_subplot(gs[0, 0])
    info_ax = fig.add_subplot(gs[0, 1])
    info_ax.axis("off")

    color = "#1f77b4"
    ax.plot(data["strain"], data["primary"], "-o", lw=2.4, ms=4.5, color=color, label="Primary stress")
    if show_diagnostic:
        ax.plot(data["strain"], data["diagnostic"], ":", lw=1.2, color="0.45", alpha=0.42, label="Diagnostic cell-wire stress")

    _plot_common_events(ax, data, color, show_anomalies=show_anomalies, annotate=True, include_labels=True)
    ax.axhline(0.0, color="0.35", lw=0.8)
    ax.set_xlabel("Engineering strain from grip separation (%)")
    ax.set_ylabel("Stress (GPa)")
    ax.set_title(title or f"Finite-grip vacancy tensile, {data['label']}", fontsize=16, pad=12)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=True)

    info_ax.text(
        0.0,
        1.0,
        "\n".join(_summary_lines([data], comparison=False)),
        ha="left",
        va="top",
        fontsize=8.45,
        linespacing=1.18,
        bbox=dict(boxstyle="round,pad=0.55", fc="white", ec="0.65", alpha=0.97),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_comparison(
    data_list: list[dict],
    out: Path,
    *,
    show_diagnostic: bool,
    show_anomalies: bool,
    title: str | None,
) -> None:
    fig = plt.figure(figsize=(12.8, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.35, 1.35])
    ax = fig.add_subplot(gs[0, 0])
    info_ax = fig.add_subplot(gs[0, 1])
    info_ax.axis("off")
    colors = ["#1f77b4", "#d95f02", "#2ca02c", "#9467bd"]

    for data, color in zip(data_list, colors):
        ax.plot(data["strain"], data["primary"], "-o", lw=2.4, ms=4.3, color=color, label=f"{data['label']} primary")
        if show_diagnostic:
            ax.plot(data["strain"], data["diagnostic"], ":", lw=1.1, color=color, alpha=0.28, label=f"{data['label']} diagnostic")
        _plot_common_events(ax, data, color, show_anomalies=show_anomalies, annotate=False)

    ax.scatter([], [], marker="D", s=92, color="#b23a1b", edgecolor="black", label="Peak load / elastic limit")
    ax.scatter([], [], marker="P", s=92, color="#7a5195", edgecolor="black", label="Plastic onset")
    ax.scatter([], [], s=90, facecolor="white", edgecolor="0.2", linewidth=2.0, label="Latest completed")
    if show_anomalies:
        ax.scatter([], [], marker="^", s=88, color="#d8a31a", edgecolor="black", label="Diagnostic anomaly")
    ax.axhline(0.0, color="0.35", lw=0.8)
    ax.set_xlabel("Engineering strain from grip separation (%)")
    ax.set_ylabel("Primary stress (GPa)")
    ax.set_title(title or "Finite-grip Al [111] vacancy tensile: r=1/r=2 event comparison", fontsize=14, pad=12)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=True, fontsize=9.0)
    info_ax.text(
        0.0,
        1.0,
        "\n".join(_summary_lines(data_list, comparison=True)),
        ha="left",
        va="top",
        fontsize=8.8,
        linespacing=1.18,
        bbox=dict(boxstyle="round,pad=0.55", fc="white", ec="0.65", alpha=0.97),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Presentation-grade finite-grip tensile event plots.")
    ap.add_argument("--results-dir", action="append", required=True, help="Result directory with tensile_events.csv. Repeat for comparison.")
    ap.add_argument("--label", action="append", default=[], help="Optional label. Repeat to match --results-dir.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--show-diagnostic", action="store_true", help="Show faded diagnostic cell-wire stress.")
    ap.add_argument("--show-anomalies", action="store_true", help="Show compressed key diagnostic anomalies.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    labels = list(args.label)
    data_list = []
    for i, result_text in enumerate(args.results_dir):
        result_dir = _resolve(result_text)
        label = labels[i] if i < len(labels) else None
        data_list.append(_load_result(result_dir, label=label))
    out = _resolve(args.out)
    title = args.title.strip() or None
    if len(data_list) == 1:
        plot_single(data_list[0], out, show_diagnostic=args.show_diagnostic, show_anomalies=args.show_anomalies, title=title)
    else:
        plot_comparison(data_list, out, show_diagnostic=args.show_diagnostic, show_anomalies=args.show_anomalies, title=title)
    print(f"[presentation] output: {out}")


if __name__ == "__main__":
    main()
