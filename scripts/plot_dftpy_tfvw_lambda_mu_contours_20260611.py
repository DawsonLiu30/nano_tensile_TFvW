#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D


QE_EF_EV = 0.601167
QE_A0_A = 4.039865


def read_matrix(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    mu = np.array([float(value) for value in rows[0][1:]], dtype=float)
    lam = np.array([float(row[0]) for row in rows[1:]], dtype=float)
    values = np.array(
        [
            [
                np.nan if value.strip().lower() in {"", "nan"} else float(value)
                for value in row[1:]
            ]
            for row in rows[1:]
        ],
        dtype=float,
    )
    return lam, mu, values


def read_quality(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    mu = np.array([float(value) for value in rows[0][1:]], dtype=float)
    lam = np.array([float(row[0]) for row in rows[1:]], dtype=float)
    values = np.array([row[1:] for row in rows[1:]], dtype=object)
    return lam, mu, values


def style_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(r"$\lambda$ (TF weight)", fontsize=11)
    ax.set_ylabel(r"$\mu$ (vW weight)", fontsize=11)
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(0.1, 1.0)
    ax.set_xticks(np.arange(0.1, 1.01, 0.1))
    ax.set_yticks(np.arange(0.1, 1.01, 0.1))
    ax.grid(color="white", linewidth=0.45, alpha=0.28)
    ax.set_aspect("equal", adjustable="box")


def add_point_markers(
    ax: plt.Axes,
    lam: np.ndarray,
    mu: np.ndarray,
    quality: np.ndarray,
) -> None:
    xx, yy = np.meshgrid(lam, mu)
    quality_t = quality.T
    complete = quality_t != "TIMEOUT"
    flagged = complete & (quality_t != "PASS")
    timeout = quality_t == "TIMEOUT"

    ax.scatter(
        xx[complete],
        yy[complete],
        s=8,
        c="black",
        alpha=0.42,
        linewidths=0,
        zorder=5,
    )
    ax.scatter(
        xx[flagged],
        yy[flagged],
        marker="x",
        s=32,
        c="#202020",
        linewidths=1.1,
        zorder=7,
    )
    ax.scatter(
        xx[timeout],
        yy[timeout],
        marker="s",
        s=42,
        facecolors="#d9d9d9",
        edgecolors="#555555",
        linewidths=0.8,
        zorder=8,
    )


def contour_panel(
    ax: plt.Axes,
    lam: np.ndarray,
    mu: np.ndarray,
    values: np.ndarray,
    quality: np.ndarray,
    *,
    title: str,
    levels: list[float] | np.ndarray,
    cmap: str,
    colorbar_label: str,
    highlight_value: float | None = None,
    highlight_label: str | None = None,
) -> None:
    xx, yy = np.meshgrid(lam, mu)
    z = np.ma.masked_invalid(values.T)
    norm = BoundaryNorm(levels, ncolors=256, clip=False)
    filled = ax.contourf(
        xx,
        yy,
        z,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
        corner_mask=False,
    )
    line_levels = np.asarray(levels[1:-1])
    lines = ax.contour(
        xx,
        yy,
        z,
        levels=line_levels,
        colors="#2d2d2d",
        linewidths=0.55,
        alpha=0.68,
        corner_mask=False,
    )
    ax.clabel(lines, inline=True, fontsize=7, fmt="%g")

    if highlight_value is not None:
        highlight = ax.contour(
            xx,
            yy,
            z,
            levels=[highlight_value],
            colors="#d62728",
            linewidths=2.8,
            corner_mask=False,
            zorder=9,
        )
        if highlight.allsegs and any(len(segment) for segment in highlight.allsegs[0]):
            ax.clabel(
                highlight,
                inline=True,
                fontsize=9,
                fmt={highlight_value: highlight_label or f"{highlight_value:g}"},
                colors="#a51414",
            )
        ax.text(
            0.98,
            0.965,
            highlight_label or f"{highlight_value:g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="#a51414",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#d62728",
                "alpha": 0.9,
            },
            zorder=10,
        )

    add_point_markers(ax, lam, mu, quality)
    style_axis(ax, title)
    colorbar = ax.figure.colorbar(filled, ax=ax, pad=0.025, fraction=0.05)
    colorbar.set_label(colorbar_label, fontsize=10)
    colorbar.ax.tick_params(labelsize=8)


def build_figures(tables: Path, output: Path) -> None:
    lam_a0, mu_a0, lattice = read_matrix(
        tables / "matrix_pristine_lattice_constant_A.csv"
    )
    lam_ef, mu_ef, formation = read_matrix(
        tables / "matrix_vacancy_formation_energy_eV.csv"
    )
    lam_ke, mu_ke, kinetic = read_matrix(
        tables / "matrix_pristine_kedf_energy_eV_per_atom.csv"
    )
    lam_qc, mu_qc, quality = read_quality(tables / "matrix_quality_status.csv")

    if not (
        np.array_equal(lam_a0, lam_ef)
        and np.array_equal(lam_a0, lam_ke)
        and np.array_equal(lam_a0, lam_qc)
        and np.array_equal(mu_a0, mu_ef)
        and np.array_equal(mu_a0, mu_ke)
        and np.array_equal(mu_a0, mu_qc)
    ):
        raise ValueError("Lambda/mu grids differ between source matrices.")

    output.mkdir(parents=True, exist_ok=True)

    panel_specs = [
        dict(
            values=lattice,
            title="Relaxed lattice constant",
            levels=np.arange(2.8, 4.41, 0.1),
            cmap="YlGnBu",
            colorbar_label=r"$a_0$ (Å)",
            highlight_value=QE_A0_A,
            highlight_label=f"QE {QE_A0_A:.4f} Å",
            filename="01_lattice_constant_contour.png",
        ),
        dict(
            values=formation,
            title="Vacancy formation energy",
            levels=[0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 20.0],
            cmap="YlOrRd",
            colorbar_label=r"$E_f^{vac}$ (eV)",
            highlight_value=QE_EF_EV,
            highlight_label=f"QE {QE_EF_EV:.3f} eV",
            filename="02_vacancy_formation_energy_contour.png",
        ),
        dict(
            values=kinetic,
            title="Kinetic-energy functional energy",
            levels=np.arange(7.0, 24.51, 1.0),
            cmap="viridis",
            colorbar_label="KEDF energy (eV/atom)",
            highlight_value=None,
            highlight_label=None,
            filename="03_kedf_energy_contour.png",
        ),
    ]

    legend_items = [
        Line2D(
            [0],
            [0],
            marker="x",
            color="#202020",
            linestyle="None",
            markersize=7,
            label="QC flagged / collapsed",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            markerfacecolor="#d9d9d9",
            markeredgecolor="#555555",
            linestyle="None",
            markersize=7,
            label="Timeout / unavailable",
        ),
        Line2D(
            [0],
            [0],
            color="#d62728",
            linewidth=2.8,
            label="QE reference contour",
        ),
    ]

    for spec in panel_specs:
        figure, axis = plt.subplots(figsize=(7.3, 6.2), constrained_layout=True)
        contour_panel(
            axis,
            lam_a0,
            mu_a0,
            spec["values"],
            quality,
            title=spec["title"],
            levels=spec["levels"],
            cmap=spec["cmap"],
            colorbar_label=spec["colorbar_label"],
            highlight_value=spec["highlight_value"],
            highlight_label=spec["highlight_label"],
        )
        axis.legend(
            handles=legend_items if spec["highlight_value"] is not None else legend_items[:2],
            loc="upper left",
            fontsize=8,
            framealpha=0.92,
        )
        figure.savefig(output / spec["filename"], dpi=320, bbox_inches="tight")
        plt.close(figure)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18.2, 6.25),
        constrained_layout=True,
    )
    for axis, spec in zip(axes, panel_specs):
        contour_panel(
            axis,
            lam_a0,
            mu_a0,
            spec["values"],
            quality,
            title=spec["title"],
            levels=spec["levels"],
            cmap=spec["cmap"],
            colorbar_label=spec["colorbar_label"],
            highlight_value=spec["highlight_value"],
            highlight_label=spec["highlight_label"],
        )
    figure.suptitle(
        "DFTpy LDA–TFvW λ–μ response maps after full atom-and-cell relaxation",
        fontsize=16,
        fontweight="bold",
    )
    figure.legend(
        handles=legend_items,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.035),
        fontsize=9,
        frameon=False,
    )
    combined_png = output / "00_three_contour_maps_combined.png"
    figure.savefig(combined_png, dpi=320, bbox_inches="tight")
    figure.savefig(output / "00_three_contour_maps_combined.pdf", bbox_inches="tight")
    plt.close(figure)

    completed = int(np.isfinite(formation).sum())
    timeout = int(formation.size - completed)
    with (output / "PLOT_DATA_NOTE.txt").open("w", encoding="utf-8") as handle:
        handle.write(
            "DFTpy LDA-TFvW lambda-mu contour maps\n\n"
            f"Completed data points used: {completed}/100\n"
            f"Unavailable points left blank: {timeout}\n"
            f"QE vacancy reference: {QE_EF_EV:.6f} eV\n"
            f"QE lattice reference: {QE_A0_A:.6f} Angstrom\n"
            "Axes: horizontal lambda (TF weight), vertical mu (vW weight).\n"
            "Black crosses indicate QC-flagged or collapsed calculations.\n"
        )

    print(f"Output: {output}")
    print(f"Completed points: {completed}/100")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_figures(args.tables.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
