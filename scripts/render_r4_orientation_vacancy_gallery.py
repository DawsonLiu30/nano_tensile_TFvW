from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from ase.io import write

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ase_nanocrystal import build_circular_nanowire
from scripts.prepare_vacancy_periodic_wire import _choose_surface_vacancy_index
from scripts.prepare_grip_vacancy_wire import (
    build_finite_grip_wire,
    choose_free_surface_vacancy,
    select_grip_indices,
)


ORIENTATIONS = ["100", "110", "111"]
ATOM_FILL = "#c7adad"
ATOM_EDGE = "#151515"
VACANCY_EDGE = "#ba1b1d"
TITLE_COLOR = {
    "100": "#2f80c1",
    "110": "#cc6b1d",
    "111": "#3aa05d",
}


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _repeat_z(atoms, repeat_z: int):
    return atoms.repeat((1, 1, int(repeat_z)))


def _remove_atom(atoms, atom_index: int):
    defect = atoms.copy()
    del defect[int(atom_index)]
    return defect


def _estimate_atom_radius(atoms) -> float:
    pos = np.asarray(atoms.get_positions(), dtype=float)
    if len(pos) < 2:
        return 1.2
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    dist[dist < 1.0e-8] = np.inf
    nn = float(np.min(dist))
    return max(0.8, 0.42 * nn)


def _project_positions(atoms, view: str) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(atoms.get_positions(), dtype=float)
    if view == "top":
        xy = pos[:, [0, 1]]
        depth = pos[:, 2]
    elif view == "side":
        xy = pos[:, [0, 2]]
        depth = pos[:, 1]
    else:
        raise ValueError(f"Unsupported view: {view}")
    order = np.argsort(depth)
    return xy[order], depth[order]


def _project_single(point_xyz: np.ndarray, *, view: str) -> np.ndarray:
    point = np.asarray(point_xyz, dtype=float)
    if view == "top":
        return point[[0, 1]]
    if view == "side":
        return point[[0, 2]]
    raise ValueError(f"Unsupported view: {view}")


def _draw_atoms(
    ax,
    atoms,
    *,
    view: str,
    atom_radius: float,
    vacancy_xyz: np.ndarray | None = None,
    pad_scale_x: float = 2.6,
    pad_scale_y: float = 2.6,
) -> None:
    coords, _depth = _project_positions(atoms, view)
    for x, y in coords:
        ax.add_patch(
            Circle(
                (float(x), float(y)),
                radius=float(atom_radius),
                facecolor=ATOM_FILL,
                edgecolor=ATOM_EDGE,
                linewidth=0.7,
            )
        )

    if vacancy_xyz is not None:
        vx, vy = _project_single(vacancy_xyz, view=view)
        ax.add_patch(
            Circle(
                (float(vx), float(vy)),
                radius=1.25 * float(atom_radius),
                facecolor="none",
                edgecolor=VACANCY_EDGE,
                linewidth=1.8,
            )
        )
        ax.add_patch(
            Circle(
                (float(vx), float(vy)),
                radius=0.32 * float(atom_radius),
                facecolor=VACANCY_EDGE,
                edgecolor="none",
            )
        )

    xmin = float(np.min(coords[:, 0]))
    xmax = float(np.max(coords[:, 0]))
    ymin = float(np.min(coords[:, 1]))
    ymax = float(np.max(coords[:, 1]))
    pad_x = float(pad_scale_x) * float(atom_radius)
    pad_y = float(pad_scale_y) * float(atom_radius)
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect("equal")
    ax.set_axis_off()


def _annotate_periodic_side(ax, *, color: str) -> None:
    ax.text(0.50, 1.03, "⋮", transform=ax.transAxes, ha="center", va="bottom", fontsize=18, color=color)
    ax.text(0.50, -0.03, "⋮", transform=ax.transAxes, ha="center", va="top", fontsize=18, color=color)
    ax.text(
        1.03,
        0.50,
        "PBC in z",
        transform=ax.transAxes,
        ha="left",
        va="center",
        rotation=90,
        fontsize=10.5,
        color=color,
        weight="bold",
    )


def _build_models(
    *,
    a0: float,
    diameter_nm: float,
    vacuum: float,
    z_repeat: int,
    z_window_fraction: float,
    finite_length_A: float,
    grip_thickness_A: float,
    z_vacuum_A: float,
) -> list[dict]:
    rows: list[dict] = []
    for orientation in ORIENTATIONS:
        nanocolumn = build_circular_nanowire(
            a0=float(a0),
            diameter_nm=float(diameter_nm),
            length_z=1.0,
            vacuum=float(vacuum),
            orientation=orientation,
        )
        nanocolumn = _repeat_z(nanocolumn, int(z_repeat))

        finite_pristine = build_finite_grip_wire(
            a0=float(a0),
            diameter_nm=float(diameter_nm),
            wire_length=float(finite_length_A),
            xy_vacuum=float(vacuum),
            z_vacuum=float(z_vacuum_A),
            orientation=orientation,
        )
        bottom_idx, top_idx = select_grip_indices(finite_pristine, grip_thickness=float(grip_thickness_A))
        vacancy_index, site = choose_free_surface_vacancy(
            finite_pristine,
            bottom_idx=bottom_idx,
            top_idx=top_idx,
            z_window_fraction=float(z_window_fraction),
        )
        vacancy_xyz = np.asarray(
            [
                float(site["x_A"]),
                float(site["y_A"]),
                float(site["z_A"]),
            ],
            dtype=float,
        )
        vacancy_nanocrystal = _remove_atom(finite_pristine, int(vacancy_index))
        rows.append(
            {
                "orientation": orientation,
                "nanocolumn": nanocolumn,
                "nanocrystal_pristine": finite_pristine,
                "vacancy_nanocrystal": vacancy_nanocrystal,
                "vacancy_index": int(vacancy_index),
                "vacancy_site": site,
                "vacancy_xyz": vacancy_xyz,
                "column_atom_radius": _estimate_atom_radius(nanocolumn),
                "nanocrystal_atom_radius": _estimate_atom_radius(finite_pristine),
                "finite_length_A": float(finite_length_A),
                "grip_thickness_A": float(grip_thickness_A),
            }
        )
    return rows


def _write_structures(
    rows: list[dict],
    outdir: Path,
    *,
    diameter_nm: float,
    z_repeat: int,
    finite_length_A: float,
) -> None:
    structures_dir = outdir / "generated_r4_orientation_models"
    structures_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []
    for row in rows:
        orientation = row["orientation"]
        nanocolumn = row["nanocolumn"]
        nanocrystal_pristine = row["nanocrystal_pristine"]
        vacancy_nanocrystal = row["vacancy_nanocrystal"]

        nanocolumn_base = structures_dir / f"nanocolumn_{orientation}_r{float(diameter_nm):.1f}nm"
        nanocrystal_pristine_base = structures_dir / f"nanocrystal_pristine_{orientation}_r{float(diameter_nm):.1f}nm"
        vacancy_base = structures_dir / f"vacancy_nanocrystal_{orientation}_r{float(diameter_nm):.1f}nm"
        write(str(nanocolumn_base.with_suffix(".xyz")), nanocolumn)
        write(str(nanocolumn_base.with_suffix(".vasp")), nanocolumn, vasp5=True, direct=True)
        write(str(nanocrystal_pristine_base.with_suffix(".xyz")), nanocrystal_pristine)
        write(str(nanocrystal_pristine_base.with_suffix(".vasp")), nanocrystal_pristine, vasp5=True, direct=True)
        write(str(vacancy_base.with_suffix(".xyz")), vacancy_nanocrystal)
        write(str(vacancy_base.with_suffix(".vasp")), vacancy_nanocrystal, vasp5=True, direct=True)

        manifest_rows.append(
            {
                "orientation": orientation,
                "diameter_nm": float(diameter_nm),
                "nanocolumn_z_repeat": int(z_repeat),
                "nanocrystal_target_length_A": float(finite_length_A),
                "nanocolumn_atoms": int(len(nanocolumn)),
                "nanocrystal_pristine_atoms": int(len(nanocrystal_pristine)),
                "vacancy_nanocrystal_atoms": int(len(vacancy_nanocrystal)),
                "vacancy_index_removed_from_nanocrystal_pristine": int(row["vacancy_index"]),
                "vacancy_site": row["vacancy_site"],
                "nanocolumn_xyz": str(nanocolumn_base.with_suffix(".xyz")),
                "nanocrystal_pristine_xyz": str(nanocrystal_pristine_base.with_suffix(".xyz")),
                "vacancy_xyz": str(vacancy_base.with_suffix(".xyz")),
            }
        )

    manifest = {
        "created_at": _ts(),
        "description": "Representative r=4 orientation comparison figure inputs for Chapter 4.2",
        "builder": "Periodic nanocolumn: build_circular_nanowire; finite vacancy nanocrystal: build_finite_grip_wire + near-surface vacancy removal",
        "rows": manifest_rows,
    }
    (structures_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _render_gallery(rows: list[dict], out_png: Path, out_pdf: Path, *, diameter_nm: float) -> None:
    fig = plt.figure(figsize=(14.6, 10.2), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.875, bottom=0.075)
    outer = fig.add_gridspec(2, 3, wspace=0.08, hspace=0.12)

    for i, row in enumerate(rows):
        orientation = row["orientation"]
        title_color = TITLE_COLOR[orientation]
        nanocolumn = row["nanocolumn"]
        vacancy_nanocrystal = row["vacancy_nanocrystal"]
        column_atom_radius = row["column_atom_radius"]
        nanocrystal_atom_radius = row["nanocrystal_atom_radius"]
        vacancy_xyz = row["vacancy_xyz"]

        inner = outer[0, i].subgridspec(2, 1, hspace=0.02)
        ax_top = fig.add_subplot(inner[0, 0])
        ax_side = fig.add_subplot(inner[1, 0])
        _draw_atoms(ax_top, nanocolumn, view="top", atom_radius=column_atom_radius)
        _draw_atoms(ax_side, nanocolumn, view="side", atom_radius=column_atom_radius, pad_scale_y=1.8)
        _annotate_periodic_side(ax_side, color=title_color)
        ax_top.set_title(
            f"<{orientation}> nanocolumn (PBC in z)",
            fontsize=15.5,
            color=title_color,
            weight="bold",
            pad=10,
        )
        ax_top.text(0.5, -0.05, "Top view", transform=ax_top.transAxes, ha="center", va="top", fontsize=10.5)
        ax_side.text(0.5, -0.05, "Side view", transform=ax_side.transAxes, ha="center", va="top", fontsize=10.5)

        inner2 = outer[1, i].subgridspec(2, 1, hspace=0.02)
        ax_top_v = fig.add_subplot(inner2[0, 0])
        ax_side_v = fig.add_subplot(inner2[1, 0])
        _draw_atoms(
            ax_top_v,
            vacancy_nanocrystal,
            view="top",
            atom_radius=nanocrystal_atom_radius,
            vacancy_xyz=vacancy_xyz,
        )
        _draw_atoms(
            ax_side_v,
            vacancy_nanocrystal,
            view="side",
            atom_radius=nanocrystal_atom_radius,
            vacancy_xyz=vacancy_xyz,
            pad_scale_y=4.2,
        )
        ax_top_v.set_title(
            f"<{orientation}> finite vacancy nanocrystal",
            fontsize=15.5,
            color=title_color,
            weight="bold",
            pad=10,
        )
        ax_top_v.text(0.5, -0.05, "Top view", transform=ax_top_v.transAxes, ha="center", va="top", fontsize=10.5)
        ax_side_v.text(0.5, -0.05, "Side view", transform=ax_side_v.transAxes, ha="center", va="top", fontsize=10.5)

    fig.suptitle(
        f"Representative r = {float(diameter_nm):.1f} nm Al models: nanocolumns and vacancy nanocrystals",
        fontsize=19,
        weight="bold",
        y=0.945,
    )
    fig.text(0.02, 0.74, "Periodic\nnanocolumn", rotation=90, va="center", ha="center", fontsize=16, weight="bold")
    fig.text(0.02, 0.25, "Finite\nvacancy nanocrystal", rotation=90, va="center", ha="center", fontsize=16, weight="bold")
    fig.text(
        0.5,
        0.01,
        "Top row: periodic nanocolumns with z-direction PBC continuation. Bottom row: finite-length vacancy nanocrystals with exposed end facets. Red marker denotes the removed near-surface Al site selected from the central z-window.",
        ha="center",
        va="bottom",
        fontsize=11,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Render a 6-panel comparison of r=4 nanocolumn and vacancy nanocrystal models for <100>/<110>/<111>."
    )
    ap.add_argument("--a0", type=float, default=4.118877004246)
    ap.add_argument("--diameter-nm", type=float, default=4.0)
    ap.add_argument("--vacuum", type=float, default=10.0)
    ap.add_argument("--z-repeat", type=int, default=2)
    ap.add_argument("--finite-length-A", type=float, default=21.0)
    ap.add_argument("--selection-grip-thickness-A", type=float, default=3.0)
    ap.add_argument("--z-vacuum-A", type=float, default=10.0)
    ap.add_argument("--vacancy-z-window-fraction", type=float, default=0.25)
    ap.add_argument("--outdir", default="results/professor_review/chapter_4_models")
    ap.add_argument("--basename", default="04_2_r4_nanocolumn_vacancy_orientations")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir = _resolve(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = _build_models(
        a0=float(args.a0),
        diameter_nm=float(args.diameter_nm),
        vacuum=float(args.vacuum),
        z_repeat=int(args.z_repeat),
        z_window_fraction=float(args.vacancy_z_window_fraction),
        finite_length_A=float(args.finite_length_A),
        grip_thickness_A=float(args.selection_grip_thickness_A),
        z_vacuum_A=float(args.z_vacuum_A),
    )
    _write_structures(
        rows,
        outdir,
        diameter_nm=float(args.diameter_nm),
        z_repeat=int(args.z_repeat),
        finite_length_A=float(args.finite_length_A),
    )

    out_png = outdir / f"{args.basename}.png"
    out_pdf = out_png.with_suffix(".pdf")
    _render_gallery(rows, out_png, out_pdf, diameter_nm=float(args.diameter_nm))

    print(f"[r4-orientation-gallery] output: {out_png}")
    print(f"[r4-orientation-gallery] output: {out_pdf}")


if __name__ == "__main__":
    main()
