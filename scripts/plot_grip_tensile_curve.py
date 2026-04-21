from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _latest_result_dir(case_dir: Path) -> Path:
    results_dir = case_dir / "results"
    candidates = sorted(
        [p for p in results_dir.glob("*") if (p / "summary.csv").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No result folder with summary.csv found in {results_dir}")
    return candidates[0]


def _read_rows(summary_csv: Path) -> list[dict[str, str]]:
    with open(summary_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No data rows found in {summary_csv}")
    return rows


def _read_optional_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_event_rows(result_dir: Path) -> list[dict[str, str]]:
    return _read_optional_rows(result_dir / "tensile_events.csv")


def _case_dir_from_summary(summary_csv: Path) -> Path | None:
    # Expected layout: cases/<case>/results/<run>/summary.csv
    try:
        if summary_csv.parent.parent.name == "results":
            return summary_csv.parent.parent.parent
    except IndexError:
        return None
    return None


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_float(value, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _has_columns(rows: list[dict[str, str]], columns: list[str]) -> bool:
    return bool(rows) and all(column in rows[0] for column in columns)


def _event_indices(event_rows: list[dict[str, str]], tag: str, cycles: np.ndarray) -> list[int]:
    indices: list[int] = []
    cycle_to_index = {int(cycle): idx for idx, cycle in enumerate(cycles)}
    for row in event_rows:
        tags = {part.strip() for part in str(row.get("event_tags", "")).split(";") if part.strip()}
        if tag in tags:
            cycle = int(row["cycle"])
            if cycle in cycle_to_index:
                indices.append(cycle_to_index[cycle])
    return indices


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot finite-grip vacancy tensile stress-strain curves.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--summary", help="Path to a summary.csv file.")
    src.add_argument("--case-dir", help="Case directory; latest results/*/summary.csv is used.")
    src.add_argument("--results-dir", help="Specific results directory containing summary.csv.")
    ap.add_argument("--out", default="", help="Output PNG path. Defaults beside summary.csv.")
    ap.add_argument("--title", default="")
    ap.add_argument("--cycles-target", type=int, default=20)
    ap.add_argument("--copy-csv", action="store_true", help="Copy summary.csv beside the output PNG.")
    ap.add_argument("--dpi", type=int, default=180)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.summary:
        summary = _resolve(args.summary)
        result_dir = summary.parent
        case_dir = _case_dir_from_summary(summary)
    elif args.results_dir:
        result_dir = _resolve(args.results_dir)
        summary = result_dir / "summary.csv"
        case_dir = _case_dir_from_summary(summary)
    else:
        case_dir = _resolve(args.case_dir)
        result_dir = _latest_result_dir(case_dir)
        summary = result_dir / "summary.csv"

    if not summary.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary}")

    rows = _read_rows(summary)
    fracture_rows = _read_optional_rows(result_dir / "fracture_status.csv")
    event_rows = _read_event_rows(result_dir)
    cycle = np.asarray([int(row["cycle"]) for row in rows], dtype=int)
    strain_pct = np.asarray([float(row["strain"]) for row in rows], dtype=float) * 100.0
    raw_grip_stress = np.asarray([float(row["grip_stress_avg_GPa"]) for row in rows], dtype=float)
    raw_grip_stress0 = float(raw_grip_stress[0])
    grip_stress_offset = raw_grip_stress - raw_grip_stress0

    paper_stress_columns = ["sigma_cell_zz_GPa", "cell_area_xy_A2", "wire_area_ref_ellipse_A2"]
    if _has_columns(rows, paper_stress_columns):
        sigma_cell_zz = np.asarray([float(row["sigma_cell_zz_GPa"]) for row in rows], dtype=float)
        cell_area = np.asarray([float(row["cell_area_xy_A2"]) for row in rows], dtype=float)
        wire_area = np.asarray([float(row["wire_area_ref_ellipse_A2"]) for row in rows], dtype=float)
        diagnostic_stress = sigma_cell_zz * cell_area / wire_area
        diagnostic_label = "Diagnostic cell-wire stress"
    else:
        diagnostic_stress = np.full_like(grip_stress_offset, np.nan)
        diagnostic_label = "Diagnostic cell-wire stress unavailable"

    # For finite-grip tensile tests the machine-like primary signal is the
    # grip reaction after removing the initial preload. The cell-derived
    # wire stress is kept as a diagnostic channel because vacuum and surface
    # residual stress can dominate small-radius wires.
    primary_stress = grip_stress_offset
    primary_label = "Primary: offset grip reaction"
    primary_note_lines = [
        "Primary stress:",
        "sigma_primary = grip reaction",
        "minus cycle-0 preload",
        "",
        "Diagnostic stress:",
        "sigma_cell-wire = sigma_cell,zz",
        "x A_cell/A_wire",
    ]

    if event_rows and len(event_rows) == len(rows):
        primary_stress = np.asarray([float(row["sigma_primary_grip_offset_GPa"]) for row in event_rows], dtype=float)
        diagnostic_stress = np.asarray([float(row["sigma_cell_wire_GPa"]) for row in event_rows], dtype=float)

    out = _resolve(args.out) if args.out else result_dir / "stress_strain_publication.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    metadata = _load_json(case_dir / "inputs" / "grip_metadata.json") if case_dir else {}
    manifest = _load_json(case_dir / "inputs" / "grip_vacancy_manifest.json") if case_dir else {}
    vacancy_site = manifest.get("vacancy", {}).get("selected_site", {})

    completed = int(cycle[-1])
    in_progress = completed < int(args.cycles_target)
    title = args.title.strip()
    if not title:
        case_label = case_dir.name if case_dir else result_dir.name
        diameter_match = re.search(r"_(\d+(?:\.\d+)?)nm", case_label)
        diameter_text = diameter_match.group(1) if diameter_match else "?"
        suffix = " (in progress)" if in_progress else ""
        title = f"Finite-grip vacancy tensile, d = {diameter_text} nm{suffix}"

    yield_indices = _event_indices(event_rows, "yield_strength_candidate", cycle)
    anomaly_indices = _event_indices(event_rows, "diagnostic_anomaly", cycle)
    plastic_indices = _event_indices(event_rows, "plastic_onset", cycle)
    complete_fracture_indices = _event_indices(event_rows, "complete_fracture", cycle)
    peak_idx = int(yield_indices[0]) if yield_indices else int(np.nanargmax(primary_stress))
    last_idx = len(rows) - 1

    fig = plt.figure(figsize=(13.8, 7.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.8])
    ax = fig.add_subplot(gs[0, 0])
    info_ax = fig.add_subplot(gs[0, 1])
    info_ax.axis("off")

    ax.plot(strain_pct, primary_stress, "-o", lw=2.15, ms=4.2, label=primary_label)
    if np.any(np.isfinite(diagnostic_stress)):
        ax.plot(
            strain_pct,
            diagnostic_stress,
            ":",
            lw=1.45,
            color="0.35",
            alpha=0.85,
            label=diagnostic_label,
        )
    ax.plot(
        strain_pct,
        raw_grip_stress,
        "--s",
        lw=1.35,
        ms=3.2,
        color="0.62",
        alpha=0.62,
        label="Raw grip reaction",
    )
    if anomaly_indices:
        ax.scatter(
            strain_pct[anomaly_indices],
            primary_stress[anomaly_indices],
            s=82,
            marker="^",
            color="#d8a31a",
            edgecolor="black",
            linewidth=0.5,
            zorder=6,
            label="Diagnostic anomaly",
        )
    if plastic_indices:
        ax.scatter(
            strain_pct[plastic_indices],
            primary_stress[plastic_indices],
            s=72,
            marker="P",
            color="#7a5195",
            edgecolor="black",
            linewidth=0.5,
            zorder=6,
            label="Plastic onset",
        )
    ax.scatter(
        [strain_pct[peak_idx]],
        [primary_stress[peak_idx]],
        s=92,
        marker="D",
        color="#b23a1b",
        edgecolor="black",
        linewidth=0.6,
        zorder=7,
        label="Yield/strength candidate",
    )
    ax.scatter(
        [strain_pct[last_idx]],
        [primary_stress[last_idx]],
        s=76,
        facecolor="white",
        edgecolor="#1f6f8b",
        linewidth=1.8,
        zorder=7,
        label="Latest completed point",
    )
    ax.axhline(0.0, color="0.35", lw=0.8)

    fracture_row = None
    for row in fracture_rows:
        if str(row.get("fractured", "")).lower() == "true":
            fracture_row = row
            break
    if complete_fracture_indices:
        fracture_idx = int(complete_fracture_indices[0])
        ax.axvline(strain_pct[fracture_idx], color="#111111", lw=1.0, ls="--", alpha=0.85)
        ax.scatter(
            [strain_pct[fracture_idx]],
            [primary_stress[fracture_idx]],
            s=94,
            marker="X",
            color="#111111",
            zorder=8,
            label="Complete fracture",
        )
    elif fracture_row is not None:
        fracture_cycle = int(fracture_row["cycle"])
        fracture_matches = np.where(cycle == fracture_cycle)[0]
        if fracture_matches.size:
            fracture_idx = int(fracture_matches[0])
            ax.axvline(strain_pct[fracture_idx], color="#111111", lw=1.0, ls="--", alpha=0.85)
            ax.scatter(
                [strain_pct[fracture_idx]],
                [primary_stress[fracture_idx]],
                s=88,
                marker="X",
                color="#111111",
                zorder=7,
                label="Complete fracture",
            )

    info_lines = [
        "Definitions",
        "",
        "Yield/strength candidate:",
        "first robust primary-stress peak",
        "followed by sustained stress drop",
        "and structural rearrangement.",
        "",
        "Plastic onset:",
        "first structural rearrangement",
        "event from bond-change or",
        "non-affine displacement proxy.",
        "",
        "Diagnostic anomaly:",
        "large disagreement between",
        "primary and cell-wire stress.",
        "",
        "Complete fracture marker:",
        "max atomic z-gap > 3*d111",
        "or a major disconnected cluster.",
        "",
        "Run summary",
        *primary_note_lines,
        f"Completed cycles: 0-{completed} / {int(args.cycles_target)}",
        f"Latest primary: {primary_stress[last_idx]:.2f} GPa",
        f"Latest strain: {strain_pct[last_idx]:.2f}%",
        f"Yield candidate: {primary_stress[peak_idx]:.2f} GPa",
        f"Yield strain: {strain_pct[peak_idx]:.2f}%",
        f"Cycle-0 raw grip stress: {raw_grip_stress0:.2f} GPa",
    ]
    if event_rows:
        summary_json = result_dir / "tensile_event_summary.json"
        event_summary = _load_json(summary_json)
        if event_summary:
            info_lines.extend(
                [
                    "",
                    "Three-layer event analysis",
                    f"Elastic limit cycle: {event_summary.get('elastic_limit_cycle')}",
                    f"Plastic onset cycle: {event_summary.get('plastic_onset_cycle')}",
                    f"Strength cycle: {event_summary.get('yield_strength_candidate_cycle')}",
                    f"Complete fracture: {event_summary.get('complete_fracture_cycle')}",
                ]
            )
    if fracture_rows:
        first_fracture = fracture_row
        ref_row = first_fracture if first_fracture is not None else fracture_rows[-1]
        d0z = _maybe_float(ref_row.get("d0z_A"))
        threshold = _maybe_float(ref_row.get("gap_threshold_A"))
        max_gap = _maybe_float(ref_row.get("max_gap_all_A"))
        max_gap_free = _maybe_float(ref_row.get("max_gap_free_A"))
        if d0z is not None and threshold is not None:
            info_lines.extend(
                [
                    "",
                    "Gap threshold",
                    f"max atomic z-gap > 3*d111",
                    f"d111 = {d0z:.3f} A, threshold = {threshold:.3f} A",
                ]
            )
        if first_fracture is not None:
            info_lines.append(f"Fracture cycle: {int(first_fracture['cycle'])}")
            if "component_fractured" in first_fracture:
                info_lines.append(f"Cluster fracture: {first_fracture.get('component_fractured')}")
        elif max_gap is not None:
            info_lines.append(f"No fracture yet; latest max gap = {max_gap:.3f} A")
            if max_gap_free is not None:
                info_lines.append(f"Free-region max gap = {max_gap_free:.3f} A")
    l0 = _maybe_float(metadata.get("grip_distance_ref_A"))
    span = _maybe_float(metadata.get("wire_span_ref_A"))
    if l0 is not None and span is not None:
        info_lines.append(f"L0 = {l0:.3f} A, wire span = {span:.3f} A")
    if metadata.get("bottom_grip_indices") and metadata.get("top_grip_indices"):
        info_lines.append(
            f"Fixed grips: {len(metadata['bottom_grip_indices'])}/{len(metadata['top_grip_indices'])} atoms"
        )
    vacancy_radial = _maybe_float(vacancy_site.get("radial_distance_A"))
    if vacancy_radial is not None:
        info_lines.append(f"Vacancy radial position = {vacancy_radial:.3f} A")

    if len(primary_stress) > 2:
        drops = np.diff(primary_stress)
        if float(np.nanmin(drops)) < -1.0:
            drop_i = int(np.nanargmin(drops) + 1)
            ax.axvline(strain_pct[drop_i], color="0.35", lw=0.9, ls=":", alpha=0.75)
            ax.scatter(
                [strain_pct[drop_i]],
                [primary_stress[drop_i]],
                s=48,
                facecolor="white",
                edgecolor="0.25",
                zorder=6,
            )
            info_lines.extend(
                [
                    "",
                    "Largest primary-stress drop",
                    f"Cycle: {cycle[drop_i]}",
                    f"Delta stress: {drops[drop_i - 1]:.2f} GPa",
                    f"Strain: {strain_pct[drop_i]:.2f}%",
                ]
            )

    info_ax.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        transform=info_ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.05,
        linespacing=1.26,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "white", "edgecolor": "0.65", "alpha": 0.96},
    )

    ax.set_xlabel("Engineering strain from grip separation (%)")
    ax.set_ylabel("Stress (GPa)")
    ax.set_title(title)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=True)
    fig.savefig(out, dpi=int(args.dpi))
    plt.close(fig)

    if args.copy_csv:
        shutil.copy2(summary, out.with_suffix(".csv"))
        fracture_csv = result_dir / "fracture_status.csv"
        if fracture_csv.exists():
            shutil.copy2(fracture_csv, out.with_suffix(".fracture_status.csv"))
        events_csv = result_dir / "tensile_events.csv"
        if events_csv.exists():
            shutil.copy2(events_csv, out.with_suffix(".tensile_events.csv"))
        events_json = result_dir / "tensile_event_summary.json"
        if events_json.exists():
            shutil.copy2(events_json, out.with_suffix(".tensile_event_summary.json"))

    print(f"[plot] summary: {summary}")
    print(f"[plot] output : {out}")
    if args.copy_csv:
        print(f"[plot] copied : {out.with_suffix('.csv')}")
        fracture_copy = out.with_suffix(".fracture_status.csv")
        if fracture_copy.exists():
            print(f"[plot] copied : {fracture_copy}")
        events_copy = out.with_suffix(".tensile_events.csv")
        if events_copy.exists():
            print(f"[plot] copied : {events_copy}")
        events_summary_copy = out.with_suffix(".tensile_event_summary.json")
        if events_summary_copy.exists():
            print(f"[plot] copied : {events_summary_copy}")
    print(
        f"[plot] latest cycle={completed} strain={strain_pct[last_idx]:.6f}% "
        f"primary_stress={primary_stress[last_idx]:.6f} GPa"
    )


if __name__ == "__main__":
    main()
