#!/usr/bin/env python3
"""Create minimal advisor-style contour maps using only QC-PASS points."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


QE_EF_EV = 0.601167


def read_numeric_matrix(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def read_quality_matrix(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    mu = np.asarray([float(value) for value in rows[0][1:]], dtype=float)
    lam = np.asarray([float(row[0]) for row in rows[1:]], dtype=float)
    values = np.asarray([row[1:] for row in rows[1:]], dtype=object)
    return lam, mu, values


def check_grid(
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
    axis.set_title(title, fontsize=18, pad=10)
    axis.set_xlabel(r"$\mu$", fontsize=17)
    axis.set_ylabel(r"$\lambda$", fontsize=17)
    axis.set_xlim(0.1, 1.0)
    axis.set_ylim(0.1, 1.0)
    axis.set_xticks(np.arange(0.1, 1.01, 0.1))
    axis.set_yticks(np.arange(0.1, 1.01, 0.1))
    axis.tick_params(labelsize=10, width=1.0, length=4)
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)


def contour_only(
    axis: plt.Axes,
    mu: np.ndarray,
    lam: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    *,
    title: str,
    levels: list[float],
    label_format: str,
    qe_reference: float | None = None,
) -> None:
    xx, yy = np.meshgrid(mu, lam)
    masked = np.ma.masked_where(~valid, values)
    contours = axis.contour(
        xx,
        yy,
        masked,
        levels=levels,
        colors="black",
        linewidths=1.5,
        corner_mask=True,
    )
    axis.clabel(contours, inline=True, fontsize=10, fmt=label_format)

    if qe_reference is not None:
        qe_contour = axis.contour(
            xx,
            yy,
            masked,
            levels=[qe_reference],
            colors="red",
            linewidths=3.2,
            corner_mask=True,
        )
        if qe_contour.allsegs and any(len(segment) for segment in qe_contour.allsegs[0]):
            axis.clabel(
                qe_contour,
                inline=True,
                fontsize=11,
                fmt={qe_reference: f"QE {qe_reference:.3f} eV"},
                colors="red",
            )

    style_axis(axis, title)


def save_panel(
    output: Path,
    name: str,
    mu: np.ndarray,
    lam: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    **kwargs,
) -> None:
    figure, axis = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)
    contour_only(axis, mu, lam, values, valid, **kwargs)
    figure.savefig(output / f"{name}.png", dpi=350, bbox_inches="tight")
    figure.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lattice", required=True, type=Path)
    parser.add_argument("--formation", required=True, type=Path)
    parser.add_argument("--kinetic", required=True, type=Path)
    parser.add_argument("--quality", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    lam, mu, lattice = read_numeric_matrix(args.lattice.resolve())
    lam_f, mu_f, formation = read_numeric_matrix(args.formation.resolve())
    lam_k, mu_k, kinetic = read_numeric_matrix(args.kinetic.resolve())
    lam_q, mu_q, quality = read_quality_matrix(args.quality.resolve())
    check_grid((lam, mu), (lam_f, mu_f), "formation energy")
    check_grid((lam, mu), (lam_k, mu_k), "kinetic energy")
    check_grid((lam, mu), (lam_q, mu_q), "quality status")

    pass_mask = quality == "PASS"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    panels = [
        {
            "name": "panel_1_lattice_constant_valid",
            "values": lattice,
            "title": r"Pristine lattice constant $a_0$ ($\AA$)",
            "levels": [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2],
            "label_format": "%.2f",
            "qe_reference": None,
        },
        {
            "name": "panel_2_formation_energy_valid",
            "values": formation,
            "title": r"Vacancy formation energy $E_f^{vac}$ (eV)",
            "levels": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
            "label_format": "%.1f",
            "qe_reference": QE_EF_EV,
        },
        {
            "name": "panel_3_kinetic_energy_valid",
            "values": kinetic,
            "title": "Pristine KEDF energy (eV/atom)",
            "levels": [12.0, 14.0, 16.0, 18.0, 20.0, 22.0],
            "label_format": "%.0f",
            "qe_reference": None,
        },
    ]

    for panel in panels:
        valid = pass_mask & np.isfinite(panel["values"])
        save_panel(
            output,
            panel["name"],
            mu,
            lam,
            panel["values"],
            valid,
            title=panel["title"],
            levels=panel["levels"],
            label_format=panel["label_format"],
            qe_reference=panel["qe_reference"],
        )

    figure, axes = plt.subplots(1, 3, figsize=(17.2, 5.2), constrained_layout=True)
    for axis, panel in zip(axes, panels):
        valid = pass_mask & np.isfinite(panel["values"])
        contour_only(
            axis,
            mu,
            lam,
            panel["values"],
            valid,
            title=panel["title"],
            levels=panel["levels"],
            label_format=panel["label_format"],
            qe_reference=panel["qe_reference"],
        )
    figure.suptitle(r"DFTpy TFvW parameter maps in $(\lambda,\mu)$ space", fontsize=21)
    figure.savefig(
        output / "professor_three_maps_extreme_simple_valid.png",
        dpi=350,
        bbox_inches="tight",
    )
    figure.savefig(
        output / "professor_three_maps_extreme_simple_valid.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)

    with (output / "DATA_STATUS.txt").open("w", encoding="utf-8") as handle:
        handle.write(
            "Extreme-simple professor plots\n"
            f"QC-PASS grid points used: {pass_mask.sum()}/100\n"
            f"QE vacancy formation-energy contour: {QE_EF_EV:.6f} eV\n"
            "Horizontal axis: mu (vW weight)\n"
            "Vertical axis: lambda (TF weight)\n"
            "Left: final relaxed pristine lattice constant a0 in A.\n"
            "Middle: vacancy formation energy E_vac - (107/108) E_pristine in eV.\n"
            "Right: final relaxed pristine KEDF energy in eV/atom.\n"
            "No failed, collapsed, stress-failed, force-failed, or timeout points are plotted.\n"
        )

    print(f"Output: {output}")
    print(f"QC-PASS points: {pass_mask.sum()}/100")


if __name__ == "__main__":
    main()
