from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from ase.io import read
from ase.visualize.plot import plot_atoms


ROOT = Path(__file__).resolve().parents[1]

COLOR_BY_RADIUS = {
    1: "#1f77b4",
    2: "#d95f02",
    3: "#2ca02c",
    4: "#9467bd",
    5: "#8c564b",
    6: "#e377c2",
    7: "#17becf",
    8: "#bcbd22",
}


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _case_name(radius: int) -> str:
    return f"finite_grip_111_{radius}.0nm_vacancy_tfvw"


def _latest_result_dir(case_dir: Path) -> Path:
    candidates = sorted(
        [p for p in (case_dir / "results").glob("*") if (p / "tensile_event_summary.json").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No analyzed result folder found under {case_dir / 'results'}")
    return candidates[0]


def _metadata_for_case(case_dir: Path) -> dict:
    metadata_path = _first_existing(
        [
            case_dir / "inputs" / "grip_metadata.json",
            case_dir / "inputs_preview" / "grip_metadata_preview.json",
        ]
    )
    return _read_json(metadata_path) if metadata_path else {}


def _has_verified_grip_metadata(case_dir: Path) -> bool:
    return (case_dir / "inputs" / "grip_metadata.json").exists()


def _manifest_for_case(case_dir: Path) -> dict:
    manifest_path = _first_existing(
        [
            case_dir / "inputs" / "grip_vacancy_manifest.json",
            case_dir / "inputs_preview" / "grip_vacancy_preview_manifest.json",
        ]
    )
    return _read_json(manifest_path) if manifest_path else {}


def _load_case(radius: int) -> dict:
    case_dir = ROOT / "cases" / _case_name(radius)
    result_dir = _latest_result_dir(case_dir)
    summary = _read_json(result_dir / "tensile_event_summary.json")
    events = _read_csv(result_dir / "tensile_events.csv")
    if not events:
        raise ValueError(f"No events in {result_dir / 'tensile_events.csv'}")
    metadata = _metadata_for_case(case_dir)
    manifest = _manifest_for_case(case_dir)
    return {
        "radius": radius,
        "diameter_nm": float(radius),
        "case_dir": case_dir,
        "result_dir": result_dir,
        "summary": summary,
        "events": events,
        "metadata": metadata,
        "manifest": manifest,
        "color": COLOR_BY_RADIUS.get(radius, "#444444"),
        "has_verified_grip_metadata": _has_verified_grip_metadata(case_dir),
    }


def _try_load_case(radius: int) -> dict | None:
    try:
        return _load_case(radius)
    except (FileNotFoundError, ValueError):
        return None


def _event_cycle(case: dict) -> tuple[int, str]:
    summary = case["summary"]
    plastic = summary.get("plastic_onset_cycle")
    if plastic is not None:
        return int(plastic), "plastic onset"
    strength = summary.get("yield_strength_candidate_cycle")
    if strength is not None:
        return int(strength), "peak load"
    return int(summary["latest_cycle"]), "latest available"


def _event_row(case: dict, cycle: int) -> dict[str, str]:
    for row in case["events"]:
        if int(row["cycle"]) == int(cycle):
            return row
    return case["events"][-1]


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _render_curve_figures(cases: list[dict], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_script = ROOT / "scripts" / "plot_tensile_clean.py"

    overview_cmd = [
        sys.executable,
        str(plot_script),
    ]
    for case in cases:
        overview_cmd += ["--results-dir", str(case["result_dir"]), "--label", f"d = {case['diameter_nm']:.1f} nm"]
    overview_cmd += [
        "--out",
        str(figures_dir / "00_tensile_r1_r6_overview_clean.png"),
        "--title",
        "Finite-grip vacancy tensile response, d = 1.0-6.0 nm",
        "--no-shade",
        "--hide-events",
        "--pdf",
    ]
    subprocess.run(overview_cmd, check=True, cwd=ROOT)

    detail_cmd = [
        sys.executable,
        str(plot_script),
    ]
    for case in cases:
        detail_cmd += ["--results-dir", str(case["result_dir"]), "--label", f"d = {case['diameter_nm']:.1f} nm"]
    detail_cmd += [
        "--out",
        str(figures_dir / "01_tensile_r1_r6_overview_events.png"),
        "--title",
        "Finite-grip vacancy tensile response, d = 1.0-6.0 nm (event markers)",
        "--no-shade",
        "--pdf",
    ]
    subprocess.run(detail_cmd, check=True, cwd=ROOT)

    per_radius_dir = figures_dir / "per_radius"
    per_radius_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        cmd = [
            sys.executable,
            str(plot_script),
            "--results-dir",
            str(case["result_dir"]),
            "--out",
            str(per_radius_dir / f"r{case['radius']}_tensile_current.png"),
            "--title",
            f"Finite-grip vacancy tensile response, d = {case['diameter_nm']:.1f} nm (current range)",
            "--color",
            case["color"],
            "--pdf",
        ]
        subprocess.run(cmd, check=True, cwd=ROOT)


def _write_key_metrics(cases: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "radius",
        "diameter_nm",
        "result_dir",
        "elastic_limit_cycle",
        "elastic_limit_strain_pct",
        "elastic_limit_primary_stress_GPa",
        "plastic_onset_cycle",
        "plastic_onset_strain_pct",
        "plastic_onset_primary_stress_GPa",
        "yield_strength_candidate_cycle",
        "yield_strength_candidate_basis",
        "yield_strength_candidate_strain_pct",
        "yield_strength_candidate_primary_stress_GPa",
        "latest_cycle",
        "latest_strain_pct",
        "latest_primary_stress_GPa",
        "latest_complete_fracture",
        "complete_fracture_cycle",
        "complete_fracture_strain_pct",
        "complete_fracture_primary_stress_GPa",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            summary = case["summary"]
            row = {key: summary.get(key, "") for key in fieldnames if key not in {"radius", "diameter_nm", "result_dir"}}
            row["radius"] = case["radius"]
            row["diameter_nm"] = f"{case['diameter_nm']:.1f}"
            row["result_dir"] = str(case["result_dir"])
            writer.writerow(row)


def _write_curve_data(cases: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "radius",
        "diameter_nm",
        "cycle",
        "strain_pct",
        "sigma_primary_grip_offset_GPa",
        "sigma_grip_raw_GPa",
        "sigma_cell_wire_GPa",
        "diagnostic_anomaly",
        "structural_event",
        "complete_fracture",
        "regime",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            for row in case["events"]:
                writer.writerow(
                    {
                        "radius": case["radius"],
                        "diameter_nm": f"{case['diameter_nm']:.1f}",
                        "cycle": row["cycle"],
                        "strain_pct": row["strain_pct"],
                        "sigma_primary_grip_offset_GPa": row["sigma_primary_grip_offset_GPa"],
                        "sigma_grip_raw_GPa": row["sigma_grip_raw_GPa"],
                        "sigma_cell_wire_GPa": row["sigma_cell_wire_GPa"],
                        "diagnostic_anomaly": row["diagnostic_anomaly"],
                        "structural_event": row["structural_event"],
                        "complete_fracture": row["complete_fracture"],
                        "regime": row["regime"],
                    }
                )


def _write_available_current_status(cases: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "radius",
        "diameter_nm",
        "result_dir",
        "yield_strength_candidate_cycle",
        "yield_strength_candidate_strain_pct",
        "yield_strength_candidate_primary_stress_GPa",
        "plastic_onset_cycle",
        "plastic_onset_strain_pct",
        "plastic_onset_primary_stress_GPa",
        "latest_cycle",
        "latest_strain_pct",
        "latest_primary_stress_GPa",
        "latest_complete_fracture",
        "complete_fracture_cycle",
        "complete_fracture_strain_pct",
        "complete_fracture_primary_stress_GPa",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            summary = case["summary"]
            row = {key: summary.get(key, "") for key in fieldnames if key not in {"radius", "diameter_nm", "result_dir"}}
            row["radius"] = case["radius"]
            row["diameter_nm"] = f"{case['diameter_nm']:.1f}"
            row["result_dir"] = str(case["result_dir"])
            writer.writerow(row)


def _render_available_current_figures(cases: list[dict], figures_dir: Path) -> None:
    if not cases:
        return
    plot_script = ROOT / "scripts" / "plot_tensile_clean.py"
    available_cmd = [
        sys.executable,
        str(plot_script),
    ]
    for case in cases:
        available_cmd += ["--results-dir", str(case["result_dir"]), "--label", f"d = {case['diameter_nm']:.1f} nm"]
    available_cmd += [
        "--out",
        str(figures_dir / "02_tensile_available_r1_r6_r8_current.png"),
        "--title",
        "Finite-grip vacancy tensile response, currently available cases",
        "--no-shade",
        "--pdf",
    ]
    subprocess.run(available_cmd, check=True, cwd=ROOT)

    per_radius_dir = figures_dir / "per_radius"
    per_radius_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        cmd = [
            sys.executable,
            str(plot_script),
            "--results-dir",
            str(case["result_dir"]),
            "--out",
            str(per_radius_dir / f"r{case['radius']}_tensile_current.png"),
            "--title",
            f"Finite-grip vacancy tensile response, d = {case['diameter_nm']:.1f} nm (current range)",
            "--color",
            case["color"],
            "--pdf",
        ]
        subprocess.run(cmd, check=True, cwd=ROOT)


def _copy_case_data(cases: list[dict], data_dir: Path) -> list[dict]:
    copied: list[dict] = []
    for case in cases:
        radius = case["radius"]
        summary = case["summary"]
        event_cycle, event_label = _event_cycle(case)
        case_out = data_dir / f"r{radius}"
        case_out.mkdir(parents=True, exist_ok=True)
        result_dir = case["result_dir"]
        case_dir = case["case_dir"]

        files = {
            "summary_csv": result_dir / "summary.csv",
            "fracture_status_csv": result_dir / "fracture_status.csv",
            "tensile_events_csv": result_dir / "tensile_events.csv",
            "tensile_event_summary_json": result_dir / "tensile_event_summary.json",
            "initial_xyz": result_dir / "cycle_000_relaxed.xyz",
            "event_xyz": result_dir / f"cycle_{event_cycle:03d}_relaxed.xyz",
            "grip_metadata_json": _first_existing(
                [
                    case_dir / "inputs" / "grip_metadata.json",
                    case_dir / "inputs_preview" / "grip_metadata_preview.json",
                ]
            ),
            "grip_manifest_json": _first_existing(
                [
                    case_dir / "inputs" / "grip_vacancy_manifest.json",
                    case_dir / "inputs_preview" / "grip_vacancy_preview_manifest.json",
                ]
            ),
        }

        copied_row = {
            "radius": radius,
            "diameter_nm": case["diameter_nm"],
            "event_cycle": event_cycle,
            "event_label": event_label,
        }
        for key, src in files.items():
            if src is None:
                continue
            dst = case_out / src.name
            _copy_if_exists(src, dst)
            copied_row[key] = dst
        copied_row["result_dir"] = result_dir
        copied.append(copied_row)
    return copied


def _write_index_markdown(cases: list[dict], copied: list[dict], out_md: Path) -> None:
    lines = [
        "# PPT Data Index",
        "",
        "| Radius | Key status | summary.csv | tensile_events.csv | tensile_event_summary.json | fracture_status.csv | initial xyz | event xyz | grip metadata | manifest |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    copied_by_radius = {int(row["radius"]): row for row in copied}
    for case in cases:
        radius = case["radius"]
        row = copied_by_radius[radius]
        summary = case["summary"]
        status = f"latest C{summary['latest_cycle']}, {float(summary['latest_strain_pct']):.2f}%, {float(summary['latest_primary_stress_GPa']):.2f} GPa"
        def _link(path_key: str, label: str) -> str:
            path = row.get(path_key)
            if not path:
                return ""
            path = Path(path)
            return f"[{label}]({path.as_posix()})"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"r={radius}",
                    status,
                    _link("summary_csv", "summary"),
                    _link("tensile_events_csv", "events"),
                    _link("tensile_event_summary_json", "event summary"),
                    _link("fracture_status_csv", "fracture"),
                    _link("initial_xyz", "cycle 0"),
                    _link("event_xyz", f"C{row['event_cycle']}"),
                    _link("grip_metadata_json", "metadata"),
                    _link("grip_manifest_json", "manifest"),
                ]
            )
            + " |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _atom_colors(n_atoms: int, fixed_idx: np.ndarray, free_color: str, *, distinguish_fixed: bool) -> list[str]:
    if not distinguish_fixed:
        return [free_color] * int(n_atoms)
    fixed_mask = np.zeros(int(n_atoms), dtype=bool)
    valid = fixed_idx[(fixed_idx >= 0) & (fixed_idx < int(n_atoms))]
    fixed_mask[valid] = True
    return ["#c7c7c7" if fixed_mask[i] else free_color for i in range(int(n_atoms))]


def _sanity_check_fixed_indices(atoms, fixed_idx: np.ndarray) -> bool:
    pos = np.asarray(atoms.get_positions(), dtype=float)
    if pos.size == 0:
        return False
    z = pos[:, 2]
    valid = fixed_idx[(fixed_idx >= 0) & (fixed_idx < len(z))]
    free = np.setdiff1d(np.arange(len(z), dtype=int), valid)
    if valid.size == 0 or free.size == 0:
        return False
    z_fixed = z[valid]
    z_free = z[free]
    fixed_inside_free = np.any((z_fixed > float(np.min(z_free))) & (z_fixed < float(np.max(z_free))))
    free_at_extremes = np.any((z_free <= float(np.min(z_fixed)) + 1.0e-8) | (z_free >= float(np.max(z_fixed)) - 1.0e-8))
    return not bool(fixed_inside_free or free_at_extremes)


def _fixed_indices_from_structure(case: dict, atoms) -> np.ndarray:
    pos = np.asarray(atoms.get_positions(), dtype=float)
    z = pos[:, 2]
    zmin = float(np.min(z))
    zmax = float(np.max(z))
    thickness = float(case["metadata"].get("grip_thickness_A") or case["manifest"].get("grips", {}).get("grip_thickness_A") or 3.0)
    by_geometry = np.where((z <= zmin + thickness + 1.0e-8) | (z >= zmax - thickness - 1.0e-8))[0].astype(int)
    fixed_idx = np.asarray(case["metadata"].get("fixed_indices", []), dtype=int)
    if fixed_idx.size and _sanity_check_fixed_indices(atoms, fixed_idx):
        return fixed_idx
    return by_geometry


def _plot_atoms_panel(ax, atoms, *, rotation: str, colors: list[str], radii: float = 0.68) -> None:
    plot_atoms(
        atoms,
        ax=ax,
        rotation=rotation,
        colors=colors,
        radii=radii,
        show_unit_cell=0,
    )
    ax.set_axis_off()
    ax.set_facecolor("white")
    ax.set_aspect("equal")


def _structure_legend(fig) -> None:
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#c7c7c7", markeredgecolor="black", markersize=9, label="Fixed grips"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d95f02", markeredgecolor="black", markersize=9, label="Free deformable region"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.985), frameon=False, ncol=2, fontsize=9.5)


def _radii_tag(cases: list[dict]) -> str:
    return "_".join(f"r{case['radius']}" for case in cases)


def _render_initial_structure_gallery(cases: list[dict], out_png: Path, out_pdf: Path) -> None:
    fig, axes = plt.subplots(2, len(cases), figsize=(2.45 * len(cases), 5.7), constrained_layout=True)
    if len(cases) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, case in enumerate(cases):
        atoms = read(str(case["result_dir"] / "cycle_000_relaxed.xyz"))
        fixed_idx = _fixed_indices_from_structure(case, atoms)
        top_colors = _atom_colors(len(atoms), fixed_idx, case["color"], distinguish_fixed=False)
        side_colors = _atom_colors(len(atoms), fixed_idx, case["color"], distinguish_fixed=True)

        _plot_atoms_panel(axes[0, col], atoms, rotation="0x,0y,0z", colors=top_colors)
        _plot_atoms_panel(axes[1, col], atoms, rotation="90x,0y,0z", colors=side_colors)

        axes[0, col].set_title(f"d = {case['diameter_nm']:.1f} nm", fontsize=11.5, color=case["color"], pad=8, weight="bold")
        axes[0, col].text(0.5, -0.08, "Top view", transform=axes[0, col].transAxes, ha="center", va="top", fontsize=9.4)
        axes[1, col].text(0.5, -0.08, "Side view", transform=axes[1, col].transAxes, ha="center", va="top", fontsize=9.4)

    fig.text(0.018, 0.74, "Top", rotation=90, va="center", ha="center", fontsize=12, weight="bold")
    fig.text(0.018, 0.27, "Side", rotation=90, va="center", ha="center", fontsize=12, weight="bold")
    fig.suptitle("Finite-grip Al [111] vacancy wires: initial structures (cycle 0)", fontsize=14, weight="bold", y=1.03)
    _structure_legend(fig)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _render_event_side_gallery(cases: list[dict], out_png: Path, out_pdf: Path) -> None:
    fig, axes = plt.subplots(2, len(cases), figsize=(2.45 * len(cases), 5.9), constrained_layout=True)
    if len(cases) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, case in enumerate(cases):
        initial_atoms = read(str(case["result_dir"] / "cycle_000_relaxed.xyz"))
        fixed_idx = _fixed_indices_from_structure(case, initial_atoms)
        event_cycle, event_label = _event_cycle(case)
        event_atoms = read(str(case["result_dir"] / f"cycle_{event_cycle:03d}_relaxed.xyz"))
        colors_initial = _atom_colors(len(initial_atoms), fixed_idx, case["color"], distinguish_fixed=True)
        colors_event = _atom_colors(len(event_atoms), fixed_idx, case["color"], distinguish_fixed=True)

        _plot_atoms_panel(axes[0, col], initial_atoms, rotation="90x,0y,0z", colors=colors_initial)
        _plot_atoms_panel(axes[1, col], event_atoms, rotation="90x,0y,0z", colors=colors_event)

        axes[0, col].set_title(f"d = {case['diameter_nm']:.1f} nm", fontsize=11.5, color=case["color"], pad=8, weight="bold")
        axes[0, col].text(0.5, -0.08, "Cycle 0", transform=axes[0, col].transAxes, ha="center", va="top", fontsize=9.2)

        event_row = _event_row(case, event_cycle)
        strain = float(event_row["strain_pct"])
        stress = float(event_row["sigma_primary_grip_offset_GPa"])
        label_text = f"C{event_cycle} {event_label}\n{strain:.2f}%, {stress:.2f} GPa"
        axes[1, col].text(0.5, -0.11, label_text, transform=axes[1, col].transAxes, ha="center", va="top", fontsize=8.3)

    fig.text(0.018, 0.74, "Initial", rotation=90, va="center", ha="center", fontsize=12, weight="bold")
    fig.text(0.018, 0.27, "Event state", rotation=90, va="center", ha="center", fontsize=12, weight="bold")
    fig.suptitle(
        "Finite-grip Al [111] vacancy wires: side views at the first post-elastic event",
        fontsize=14,
        weight="bold",
        y=1.03,
    )
    _structure_legend(fig)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _write_structure_provenance(cases: list[dict], out_md: Path) -> None:
    lines = [
        "# Structure Provenance",
        "",
        "This note records whether the fixed-grip / free-region coloring can be traced directly to saved production grip metadata.",
        "",
        "## Defense-safe rule",
        "",
        "- Stress-strain plots in `ppt_ready/figures/` are defense-safe because they come directly from saved result files.",
        "- Structure panels are defense-safe only when the fixed/free coloring comes directly from saved production grip metadata (`inputs/grip_metadata.json`).",
        "- If only preview metadata exists, full-series fixed/free coloring should not be used in the defense.",
        "",
        "| Radius | Metadata status | Defense status | Basis |",
        "| --- | --- | --- | --- |",
    ]
    for case in cases:
        radius = case["radius"]
        if case["has_verified_grip_metadata"]:
            status = "Verified production metadata"
            defense = "Defense-safe"
            basis = f"`cases/{_case_name(radius)}/inputs/grip_metadata.json`"
        elif (case["case_dir"] / "inputs_preview" / "grip_metadata_preview.json").exists():
            status = "Preview metadata only"
            defense = "Do not use full-structure grip coloring"
            basis = f"`cases/{_case_name(radius)}/inputs_preview/grip_metadata_preview.json`"
        else:
            status = "No grip metadata found"
            defense = "Do not use structure coloring"
            basis = "Metadata missing"
        lines.append(f"| r={radius} | {status} | {defense} | {basis} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_structures_readme(out_md: Path, verified_cases: list[dict], provenance_md: Path) -> None:
    lines = [
        "# Structure Figures Status",
        "",
        "Only verified structure figures should live in this folder.",
        "",
        "## Current rule",
        "",
        "- Keep only structure figures whose fixed/free coloring comes directly from saved production grip metadata.",
        "- Do not place reconstructed or geometry-inferred fixed/free coloring here for the defense package.",
        "",
        f"- [Structure provenance note]({provenance_md.as_posix()})",
        "",
    ]
    if verified_cases:
        tag = _radii_tag(verified_cases)
        lines += [
            "## Available verified figures",
            "",
            f"- [Initial structure gallery ({tag}, top + side)]({(out_md.parent / f'{tag}_initial_structures_top_side_verified.png').as_posix()})",
            f"- [Initial vs event side-view comparison ({tag})]({(out_md.parent / f'{tag}_initial_vs_event_structures_side_verified.png').as_posix()})",
            "",
        ]
    else:
        lines += [
            "## Available verified figures",
            "",
            "- None yet. Wait until production grip metadata is present for the target radii.",
            "",
        ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_package_readme(
    out_md: Path,
    figures_dir: Path,
    data_dir: Path,
    structures_dir: Path,
    verified_cases: list[dict],
    provenance_md: Path,
) -> None:
    lines = [
        "# PPT-Ready Professor Review Package",
        "",
        "## Figures",
        "",
        f"- [Main overview curve]({(figures_dir / '00_tensile_r1_r6_overview_clean.png').as_posix()})",
        f"- [Event-marked comparison]({(figures_dir / '01_tensile_r1_r6_overview_events.png').as_posix()})",
        f"- [Per-radius curve folder]({(figures_dir / 'per_radius').as_posix()})",
        "",
        "## Structures",
        "",
        "Only verified structure figures are listed here.",
        "",
    ]
    if verified_cases:
        tag = _radii_tag(verified_cases)
        lines += [
            f"- [Initial structure gallery ({tag}, top + side)]({(structures_dir / f'{tag}_initial_structures_top_side_verified.png').as_posix()})",
            f"- [Initial vs event side-view comparison ({tag})]({(structures_dir / f'{tag}_initial_vs_event_structures_side_verified.png').as_posix()})",
        ]
    else:
        lines += [
            "- No defense-safe structure gallery is currently available.",
        ]
    lines += [
        f"- [Structure provenance note]({provenance_md.as_posix()})",
        "",
        "## Data",
        "",
        f"- [Key metrics CSV]({(data_dir / 'r1_r6_key_metrics.csv').as_posix()})",
        f"- [Curve data CSV]({(data_dir / 'r1_r6_curve_data.csv').as_posix()})",
        f"- [Clickable data index]({(data_dir / 'r1_r6_data_index.md').as_posix()})",
        "",
        "Each `data/r*/` folder contains copied source files from the corresponding case result directory for quick verification.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean PPT-ready professor review package.")
    parser.add_argument("--radii", default="1,2,3,4,5,6")
    parser.add_argument("--available-radii", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--outdir", default="results/professor_review/ppt_ready")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    radii = [int(text.strip()) for text in args.radii.split(",") if text.strip()]
    available_radii = [int(text.strip()) for text in args.available_radii.split(",") if text.strip()]
    outdir = _resolve(args.outdir)
    figures_dir = outdir / "figures"
    data_dir = outdir / "data"
    structures_dir = outdir / "structures"
    for path in [figures_dir, data_dir, structures_dir]:
        path.mkdir(parents=True, exist_ok=True)
    for stale in list(structures_dir.glob("*.png")) + list(structures_dir.glob("*.pdf")):
        stale.unlink()

    cases = [_load_case(radius) for radius in radii]

    _render_curve_figures(cases, figures_dir)
    copied = _copy_case_data(cases, data_dir)
    _write_key_metrics(cases, data_dir / "r1_r6_key_metrics.csv")
    _write_curve_data(cases, data_dir / "r1_r6_curve_data.csv")
    _write_index_markdown(cases, copied, data_dir / "r1_r6_data_index.md")
    verified_cases = [case for case in cases if case["has_verified_grip_metadata"]]
    if verified_cases:
        tag = _radii_tag(verified_cases)
        _render_initial_structure_gallery(
            verified_cases,
            structures_dir / f"{tag}_initial_structures_top_side_verified.png",
            structures_dir / f"{tag}_initial_structures_top_side_verified.pdf",
        )
        _render_event_side_gallery(
            verified_cases,
            structures_dir / f"{tag}_initial_vs_event_structures_side_verified.png",
            structures_dir / f"{tag}_initial_vs_event_structures_side_verified.pdf",
        )
    provenance_md = data_dir / "structure_provenance.md"
    _write_structure_provenance(cases, provenance_md)
    _write_structures_readme(structures_dir / "README.md", verified_cases, provenance_md)
    _write_package_readme(outdir / "README.md", figures_dir, data_dir, structures_dir, verified_cases, provenance_md)

    available_cases = []
    seen = set()
    for radius in available_radii:
        case = _try_load_case(radius)
        if case is None or case["radius"] in seen:
            continue
        available_cases.append(case)
        seen.add(case["radius"])
    available_cases.sort(key=lambda case: case["radius"])
    _write_available_current_status(available_cases, data_dir / "available_current_status.csv")
    _render_available_current_figures(available_cases, figures_dir)

    print(f"[ppt-package] output: {outdir}")
    print(f"[ppt-package] figures: {figures_dir}")
    print(f"[ppt-package] data   : {data_dir}")
    print(f"[ppt-package] structures: {structures_dir}")


if __name__ == "__main__":
    main()
