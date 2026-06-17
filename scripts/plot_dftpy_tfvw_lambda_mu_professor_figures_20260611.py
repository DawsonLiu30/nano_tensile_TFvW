#!/usr/bin/env python3
"""Plot the three lambda-mu response maps requested by the advisor."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm


QE_VACANCY_FORMATION_ENERGY_EV = 0.601167


def read_matrix(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    mu = np.asarray([float(value) for value in rows[0][1:]], dtype=float)
    lam = np.asarray([float(row[0]) for row in rows[1:]], dtype=float)
    values = np.asarray(
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
    mu = np.asarray([float(value) for value in rows[0][1:]], dtype=float)
    lam = np.asarray([float(row[0]) for row in rows[1:]], dtype=float)
    values = np.asarray([row[1:] for row in rows[1:]], dtype=object)
    return lam, mu, values


def validate_grid(
    reference: tuple[np.ndarray, np.ndarray],
    candidate: tuple[np.ndarray, np.ndarray],
    label: str,
) -> None:
    if not (
        np.array_equal(reference[0], candidate[0])
        and np.array_equal(reference[1], candidate[1])
    ):
        raise ValueError(f"Lambda-mu grid mismatch for {label}")


def style_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, fontsize=14, fontweight="bold", pad=12)
    axis.set_xlabel(r"$\lambda$ (TF weight)", fontsize=12)
    axis.set_ylabel(r"$\mu$ (vW weight)", fontsize=12)
    axis.set_xlim(0.1, 1.0)
    axis.set_ylim(0.1, 1.0)
    axis.set_xticks(np.arange(0.1, 1.01, 0.1))
    axis.set_yticks(np.arange(0.1, 1.01, 0.1))
    axis.tick_params(labelsize=9)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="white", linewidth=0.45, alpha=0.24)


def plot_panel(
    axis: plt.Axes,
    lam: np.ndarray,
    mu: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    levels: np.ndarray | list[float],
    cmap: str,
    colorbar_label: str,
    contour_format: str,
    qe_reference: float | None = None,
    valid_mask: np.ndarray | None = None,
) -> None:
    xx, yy = np.meshgrid(lam, mu)
    finite = np.isfinite(values.T)
    if valid_mask is None:
        contour_valid = finite
    else:
        contour_valid = finite & valid_mask.T
    z = np.ma.masked_where(~contour_valid, values.T)
    norm = BoundaryNorm(levels, ncolors=256, clip=False)

    filled = axis.contourf(
        xx,
        yy,
        z,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
        corner_mask=False,
    )
    lines = axis.contour(
        xx,
        yy,
        z,
        levels=np.asarray(levels[1:-1]),
        colors="#252525",
        linewidths=0.55,
        alpha=0.72,
        corner_mask=False,
    )
    axis.clabel(lines, inline=True, fontsize=7, fmt=contour_format)

    axis.scatter(
        xx[contour_valid],
        yy[contour_valid],
        s=9,
        c="#1f1f1f",
        alpha=0.42,
        linewidths=0,
        zorder=5,
    )
    invalid = finite & ~contour_valid
    if invalid.any():
        axis.scatter(
            xx[invalid],
            yy[invalid],
            marker="x",
            s=46,
            c="#343434",
            linewidths=1.2,
            zorder=8,
            label="QC excluded",
        )
    missing = ~finite
    if missing.any():
        axis.scatter(
            xx[missing],
            yy[missing],
            marker="s",
            s=50,
            facecolors="#d9d9d9",
            edgecolors="#555555",
            linewidths=0.8,
            zorder=8,
            label="Unavailable",
        )

    if qe_reference is not None:
        reference = axis.contour(
            xx,
            yy,
            z,
            levels=[qe_reference],
            colors="#d62728",
            linewidths=3.0,
            corner_mask=False,
            zorder=9,
        )
        if reference.allsegs and any(len(segment) for segment in reference.allsegs[0]):
            axis.clabel(
                reference,
                inline=True,
                fontsize=9,
                fmt={qe_reference: f"QE {qe_reference:.3f} eV"},
                colors="#a51414",
            )
        axis.text(
            0.98,
            0.965,
            f"QE reference = {qe_reference:.6f} eV",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="#a51414",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#d62728",
                "alpha": 0.92,
            },
            zorder=10,
        )

    style_axis(axis, title)
    colorbar = axis.figure.colorbar(filled, ax=axis, pad=0.025, fraction=0.05)
    colorbar.set_label(colorbar_label, fontsize=10)
    colorbar.ax.tick_params(labelsize=8)


def save_single(
    output: Path,
    filename: str,
    lam: np.ndarray,
    mu: np.ndarray,
    values: np.ndarray,
    **kwargs,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    plot_panel(axis, lam, mu, values, **kwargs)
    figure.savefig(output / f"{filename}.png", dpi=400, bbox_inches="tight")
    figure.savefig(output / f"{filename}.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lattice", required=True, type=Path)
    parser.add_argument("--formation", required=True, type=Path)
    parser.add_argument("--kinetic", required=True, type=Path)
    parser.add_argument("--quality", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    lam, mu, lattice = read_matrix(args.lattice.resolve())
    lam_f, mu_f, formation = read_matrix(args.formation.resolve())
    lam_k, mu_k, kinetic = read_matrix(args.kinetic.resolve())
    lam_q, mu_q, quality = read_quality(args.quality.resolve())
    validate_grid((lam, mu), (lam_f, mu_f), "formation energy")
    validate_grid((lam, mu), (lam_k, mu_k), "kinetic energy")
    validate_grid((lam, mu), (lam_q, mu_q), "quality status")
    formation_valid = quality == "PASS"

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    specifications = [
        {
            "values": lattice,
            "title": "Relaxed lattice constant",
            "levels": np.arange(2.8, 4.41, 0.1),
            "cmap": "YlGnBu",
            "colorbar_label": r"$a_0$ ($\AA$)",
            "contour_format": "%.1f",
            "qe_reference": None,
            "valid_mask": None,
            "filename": "01_lattice_constant_lambda_mu",
        },
        {
            "values": formation,
            "title": "Vacancy formation energy",
            "levels": [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 20.0],
            "cmap": "YlOrRd",
            "colorbar_label": r"$E_f^{vac}$ (eV)",
            "contour_format": "%g",
            "qe_reference": QE_VACANCY_FORMATION_ENERGY_EV,
            "valid_mask": formation_valid,
            "filename": "02_vacancy_formation_energy_lambda_mu",
        },
        {
            "values": kinetic,
            "title": "Kinetic energy",
            "levels": np.arange(8.0, 24.51, 1.0),
            "cmap": "viridis",
            "colorbar_label": "KEDF energy (eV/atom)",
            "contour_format": "%.0f",
            "qe_reference": None,
            "valid_mask": None,
            "filename": "03_kedf_kinetic_energy_lambda_mu",
        },
    ]

    for spec in specifications:
        save_single(
            output,
            spec["filename"],
            lam,
            mu,
            spec["values"],
            title=spec["title"],
            levels=spec["levels"],
            cmap=spec["cmap"],
            colorbar_label=spec["colorbar_label"],
            contour_format=spec["contour_format"],
            qe_reference=spec["qe_reference"],
            valid_mask=spec["valid_mask"],
        )

    figure, axes = plt.subplots(1, 3, figsize=(18.5, 6.2), constrained_layout=True)
    for axis, spec in zip(axes, specifications):
        plot_panel(
            axis,
            lam,
            mu,
            spec["values"],
            title=spec["title"],
            levels=spec["levels"],
            cmap=spec["cmap"],
            colorbar_label=spec["colorbar_label"],
            contour_format=spec["contour_format"],
            qe_reference=spec["qe_reference"],
            valid_mask=spec["valid_mask"],
        )
    figure.suptitle(
        "DFTpy LDA-TFvW response maps after full atom-and-cell relaxation",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(output / "00_professor_three_maps_combined.png", dpi=400, bbox_inches="tight")
    figure.savefig(output / "00_professor_three_maps_combined.pdf", bbox_inches="tight")
    plt.close(figure)

    with (output / "DATA_STATUS.txt").open("w", encoding="utf-8") as handle:
        handle.write(
            "Professor-requested lambda-mu response maps\n"
            f"Lattice constant: {np.isfinite(lattice).sum()}/100 points\n"
            f"Vacancy formation energy: {np.isfinite(formation).sum()}/100 points\n"
            f"KEDF kinetic energy: {np.isfinite(kinetic).sum()}/100 points\n"
            f"Formation-energy PASS points used for contours: {formation_valid.sum()}/100\n"
            f"QE vacancy reference: {QE_VACANCY_FORMATION_ENERGY_EV:.6f} eV\n"
            "Horizontal axis: lambda (TF weight)\n"
            "Vertical axis: mu (vW weight)\n"
        )

    print(f"Output: {output}")
    print(f"Lattice points: {np.isfinite(lattice).sum()}/100")
    print(f"Formation-energy points: {np.isfinite(formation).sum()}/100")
    print(f"KEDF points: {np.isfinite(kinetic).sum()}/100")


if __name__ == "__main__":
    main()
