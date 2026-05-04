from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.aluminum_defaults import (
    AL_FCC_A0_BENCHMARK_ANG,
    AL_FCC_BULK_MODULUS_BENCHMARK_GPA,
)


EV_PER_A3_TO_GPA = 160.21766208
GPA_TO_EV_PER_A3 = 1.0 / EV_PER_A3_TO_GPA


def _read_scan(path: Path) -> tuple[np.ndarray, np.ndarray]:
    a_vals: list[float] = []
    e_vals: list[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a_vals.append(float(row["a0_ang"]))
            e_vals.append(float(row["energy_ev_per_atom"]))
    if len(a_vals) < 4:
        raise ValueError(f"Need at least four scan points in {path}")
    return np.asarray(a_vals, dtype=float), np.asarray(e_vals, dtype=float)


def _fcc_volume_per_atom(a0: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(a0, dtype=float)
    out = (arr**3) / 4.0
    if np.isscalar(a0):
        return float(out)
    return out


def _birch_murnaghan_energy(v_atom: np.ndarray, e0: float, v0: float, b0_eva3: float, b0_prime: float) -> np.ndarray:
    eta = (v0 / np.asarray(v_atom, dtype=float)) ** (2.0 / 3.0)
    return e0 + 9.0 * v0 * b0_eva3 / 16.0 * (
        (eta - 1.0) ** 3 * b0_prime + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
    )


def _fit_metrics(label: str, a0_scan: np.ndarray, e_scan: np.ndarray) -> dict[str, float | str]:
    coeffs = np.polyfit(a0_scan, e_scan, deg=2)
    c2, c1 = float(coeffs[0]), float(coeffs[1])
    if c2 <= 0.0:
        raise ValueError(f"Quadratic curvature must be positive for {label}")

    idx_min = int(np.argmin(e_scan))
    e0_guess = float(e_scan[idx_min])
    v_atom = np.asarray(_fcc_volume_per_atom(a0_scan), dtype=float)
    v0_guess = float(v_atom[idx_min])
    a0_quad = float(-c1 / (2.0 * c2))
    d2e_da2 = float(2.0 * c2)
    b0_guess_eva3 = max((4.0 / (9.0 * a0_quad)) * d2e_da2, 1.0e-3)

    params, _ = curve_fit(
        _birch_murnaghan_energy,
        v_atom,
        e_scan,
        p0=(e0_guess, v0_guess, b0_guess_eva3, 4.0),
        bounds=(
            (-np.inf, float(v_atom.min()) * 0.95, 1.0e-4, 1.0),
            (np.inf, float(v_atom.max()) * 1.05, 5.0, 12.0),
        ),
        maxfev=20000,
    )

    e0, v0_atom, b0_eva3, b0_prime = [float(x) for x in params]
    a0_eos = float((4.0 * v0_atom) ** (1.0 / 3.0))
    return {
        "method": label,
        "e0_eV_per_atom": e0,
        "v0_atom_A3": v0_atom,
        "a0_A": a0_eos,
        "a0_quadratic_A": a0_quad,
        "bulk_modulus_eV_per_A3": b0_eva3,
        "bulk_modulus_GPa": b0_eva3 * EV_PER_A3_TO_GPA,
        "bulk_modulus_prime": b0_prime,
    }


def _relative_energy_vs_a(a_dense: np.ndarray, metrics: dict[str, float | str]) -> np.ndarray:
    v_dense = np.asarray(_fcc_volume_per_atom(a_dense), dtype=float)
    e_dense = _birch_murnaghan_energy(
        v_dense,
        float(metrics["e0_eV_per_atom"]),
        float(metrics["v0_atom_A3"]),
        float(metrics["bulk_modulus_eV_per_A3"]),
        float(metrics["bulk_modulus_prime"]),
    )
    return e_dense - float(metrics["e0_eV_per_atom"])


def _relative_energy_vs_delta_v(delta_v: np.ndarray, metrics: dict[str, float | str]) -> np.ndarray:
    v0 = float(metrics["v0_atom_A3"])
    v_dense = v0 + np.asarray(delta_v, dtype=float)
    e_dense = _birch_murnaghan_energy(
        v_dense,
        float(metrics["e0_eV_per_atom"]),
        v0,
        float(metrics["bulk_modulus_eV_per_A3"]),
        float(metrics["bulk_modulus_prime"]),
    )
    return e_dense - float(metrics["e0_eV_per_atom"])


def _write_comparison_csv(path: Path, benchmark_label: str, benchmark_a0: float, benchmark_b0: float, rows: list[dict[str, float | str]]) -> None:
    fieldnames = [
        "method",
        "a0_A",
        "v0_atom_A3",
        "bulk_modulus_GPa",
        "bulk_modulus_prime",
        "delta_a0_vs_benchmark_A",
        "delta_a0_vs_benchmark_pct",
        "delta_B0_vs_benchmark_GPa",
        "delta_B0_vs_benchmark_pct",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "method": benchmark_label,
                "a0_A": f"{benchmark_a0:.12f}",
                "v0_atom_A3": f"{_fcc_volume_per_atom(benchmark_a0):.12f}",
                "bulk_modulus_GPa": f"{benchmark_b0:.12f}",
                "bulk_modulus_prime": "",
                "delta_a0_vs_benchmark_A": f"{0.0:.12f}",
                "delta_a0_vs_benchmark_pct": f"{0.0:.12f}",
                "delta_B0_vs_benchmark_GPa": f"{0.0:.12f}",
                "delta_B0_vs_benchmark_pct": f"{0.0:.12f}",
            }
        )
        for row in rows:
            a0_val = float(row["a0_A"])
            b0_val = float(row["bulk_modulus_GPa"])
            writer.writerow(
                {
                    "method": row["method"],
                    "a0_A": f"{a0_val:.12f}",
                    "v0_atom_A3": f"{float(row['v0_atom_A3']):.12f}",
                    "bulk_modulus_GPa": f"{b0_val:.12f}",
                    "bulk_modulus_prime": f"{float(row['bulk_modulus_prime']):.12f}",
                    "delta_a0_vs_benchmark_A": f"{(a0_val - benchmark_a0):.12f}",
                    "delta_a0_vs_benchmark_pct": f"{((a0_val - benchmark_a0) / benchmark_a0 * 100.0):.12f}",
                    "delta_B0_vs_benchmark_GPa": f"{(b0_val - benchmark_b0):.12f}",
                    "delta_B0_vs_benchmark_pct": f"{((b0_val - benchmark_b0) / benchmark_b0 * 100.0):.12f}",
                }
            )


def _write_summary_txt(path: Path, benchmark_label: str, benchmark_a0: float, benchmark_b0: float, qe: dict[str, float | str], ofdft: dict[str, float | str]) -> None:
    lines = [
        f"benchmark_label={benchmark_label}",
        f"benchmark_a0_A={benchmark_a0:.12f}",
        f"benchmark_bulk_modulus_GPa={benchmark_b0:.12f}",
        f"qe_a0_A={float(qe['a0_A']):.12f}",
        f"qe_v0_atom_A3={float(qe['v0_atom_A3']):.12f}",
        f"qe_bulk_modulus_GPa={float(qe['bulk_modulus_GPa']):.12f}",
        f"qe_bulk_modulus_prime={float(qe['bulk_modulus_prime']):.12f}",
        f"ofdft_a0_A={float(ofdft['a0_A']):.12f}",
        f"ofdft_v0_atom_A3={float(ofdft['v0_atom_A3']):.12f}",
        f"ofdft_bulk_modulus_GPa={float(ofdft['bulk_modulus_GPa']):.12f}",
        f"ofdft_bulk_modulus_prime={float(ofdft['bulk_modulus_prime']):.12f}",
        f"delta_a0_ofdft_minus_qe_A={(float(ofdft['a0_A']) - float(qe['a0_A'])):.12f}",
        f"delta_a0_ofdft_minus_qe_pct={((float(ofdft['a0_A']) - float(qe['a0_A'])) / float(qe['a0_A']) * 100.0):.12f}",
        f"delta_B0_ofdft_minus_qe_GPa={(float(ofdft['bulk_modulus_GPa']) - float(qe['bulk_modulus_GPa'])):.12f}",
        f"delta_B0_ofdft_minus_qe_pct={((float(ofdft['bulk_modulus_GPa']) - float(qe['bulk_modulus_GPa'])) / float(qe['bulk_modulus_GPa']) * 100.0):.12f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_readme(path: Path, benchmark_label: str, benchmark_a0: float, benchmark_b0: float, qe: dict[str, float | str], ofdft: dict[str, float | str]) -> None:
    lines = [
        "# Bulk fcc Al benchmark: EOS comparison of QE KS-DFT and DFTpy OF-DFT (TFvW)",
        "",
        "This folder stores the preferred bulk comparison dataset for thesis / professor-review use.",
        "",
        "## Data source",
        "",
        "- QE data: `qe_bulk_a0_scan.csv`",
        "- DFTpy TFvW data: `dftpy_bulk_a0_scan_dense.csv`",
        "- Combined figure: `bulk_compare_FINAL.png` and `bulk_compare_FINAL.pdf`",
        "- Numeric comparison table: `bulk_benchmark_comparison.csv`",
        "",
        "## EOS quantities extracted from `E(V)` fits",
        "",
        f"- Benchmark used for comparison: `{benchmark_label}`",
        f"- Benchmark lattice constant `a0`: `{benchmark_a0:.4f} A`",
        f"- Benchmark bulk modulus `B0`: `{benchmark_b0:.1f} GPa`",
        f"- QE / KS-DFT fitted `a0`: `{float(qe['a0_A']):.6f} A`",
        f"- QE / KS-DFT fitted `B0`: `{float(qe['bulk_modulus_GPa']):.2f} GPa`",
        f"- DFTpy / OF-DFT (TFvW) fitted `a0`: `{float(ofdft['a0_A']):.6f} A`",
        f"- DFTpy / OF-DFT (TFvW) fitted `B0`: `{float(ofdft['bulk_modulus_GPa']):.2f} GPa`",
        f"- `a0` difference of OF-DFT vs QE: `{(float(ofdft['a0_A']) - float(qe['a0_A'])):.6f} A`",
        f"- `B0` difference of OF-DFT vs QE: `{(float(ofdft['bulk_modulus_GPa']) - float(qe['bulk_modulus_GPa'])):.2f} GPa`",
        "",
        "## Why this is the preferred bulk figure",
        "",
        "The earlier comparison emphasized only the equilibrium lattice constant. The updated EOS workflow keeps the same scan data,",
        "but now extracts both the equilibrium lattice constant and the bulk modulus through an `E(V)` fit so that the professor's",
        "requested benchmark-style comparison is visible in a single place.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(out_png: Path, out_pdf: Path, benchmark_label: str, benchmark_a0: float, benchmark_b0: float, qe_scan: tuple[np.ndarray, np.ndarray], dftpy_scan: tuple[np.ndarray, np.ndarray], qe: dict[str, float | str], ofdft: dict[str, float | str]) -> None:
    qe_a, qe_e = qe_scan
    ofdft_a, ofdft_e = dftpy_scan

    qe_rel = qe_e - float(qe["e0_eV_per_atom"])
    ofdft_rel = ofdft_e - float(ofdft["e0_eV_per_atom"])
    qe_dense_a = np.linspace(float(qe_a.min()), float(qe_a.max()), 400)
    ofdft_dense_a = np.linspace(float(ofdft_a.min()), float(ofdft_a.max()), 400)

    qe_dense_rel = _relative_energy_vs_a(qe_dense_a, qe)
    ofdft_dense_rel = _relative_energy_vs_a(ofdft_dense_a, ofdft)

    max_delta_v = max(
        abs(float(_fcc_volume_per_atom(qe_a.max())) - float(qe["v0_atom_A3"])),
        abs(float(_fcc_volume_per_atom(qe_a.min())) - float(qe["v0_atom_A3"])),
        abs(float(_fcc_volume_per_atom(ofdft_a.max())) - float(ofdft["v0_atom_A3"])),
        abs(float(_fcc_volume_per_atom(ofdft_a.min())) - float(ofdft["v0_atom_A3"])),
    )
    dense_delta_v = np.linspace(-max_delta_v, max_delta_v, 400)
    qe_curv = _relative_energy_vs_delta_v(dense_delta_v, qe)
    ofdft_curv = _relative_energy_vs_delta_v(dense_delta_v, ofdft)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10.2, 10.8))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.08, top=0.94, hspace=0.24)

    ax_top.scatter(qe_a, qe_rel, color="#1f77b4", s=28, label="QE (KS-DFT)")
    ax_top.plot(qe_dense_a, qe_dense_rel, "--", color="#ff7f0e", linewidth=1.6, label="QE Birch-Murnaghan fit")
    ax_top.scatter(ofdft_a, ofdft_rel, color="#2ca02c", s=28, marker="s", label="OF-DFT (TFvW)")
    ax_top.plot(ofdft_dense_a, ofdft_dense_rel, "--", color="#d62728", linewidth=1.6, label="OF-DFT Birch-Murnaghan fit")
    ax_top.axvline(float(qe["a0_A"]), color="#1f77b4", linestyle=":", linewidth=1.0)
    ax_top.axvline(float(ofdft["a0_A"]), color="#2ca02c", linestyle=":", linewidth=1.0)
    ax_top.axvline(benchmark_a0, color="0.4", linestyle="-.", linewidth=1.0)
    ax_top.set_title("Bulk fcc Al: EOS comparison of QE and OF-DFT")
    ax_top.set_xlabel("Lattice constant a0 (A)")
    ax_top.set_ylabel("ΔE (eV/atom)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="upper left")
    ax_top.text(
        0.55,
        0.96,
        "\n".join(
            [
                f"Benchmark a0 = {benchmark_a0:.4f} A",
                f"Benchmark B0 = {benchmark_b0:.1f} GPa",
                f"QE: a0 = {float(qe['a0_A']):.4f} A, B0 = {float(qe['bulk_modulus_GPa']):.1f} GPa",
                f"OF-DFT: a0 = {float(ofdft['a0_A']):.4f} A, B0 = {float(ofdft['bulk_modulus_GPa']):.1f} GPa",
            ]
        ),
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=8.9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.75", alpha=0.96),
    )

    ax_bottom.plot(dense_delta_v, qe_curv, "--", color="#ff7f0e", linewidth=1.7, label=f"QE fit (B0 = {float(qe['bulk_modulus_GPa']):.1f} GPa)")
    ax_bottom.plot(dense_delta_v, ofdft_curv, "--", color="#d62728", linewidth=1.7, label=f"OF-DFT fit (B0 = {float(ofdft['bulk_modulus_GPa']):.1f} GPa)")
    ax_bottom.scatter(
        np.asarray(_fcc_volume_per_atom(qe_a), dtype=float) - float(qe["v0_atom_A3"]),
        qe_rel,
        color="#1f77b4",
        s=28,
        label="QE points",
    )
    ax_bottom.scatter(
        np.asarray(_fcc_volume_per_atom(ofdft_a), dtype=float) - float(ofdft["v0_atom_A3"]),
        ofdft_rel,
        color="#2ca02c",
        s=28,
        marker="s",
        label="OF-DFT points",
    )
    ax_bottom.axvline(0.0, color="0.5", linestyle=":", linewidth=0.9)
    ax_bottom.set_title("Bulk fcc Al: curvature around the equilibrium volume")
    ax_bottom.set_xlabel("ΔVatom = Vatom - V0,atom (A^3/atom)")
    ax_bottom.set_ylabel("ΔE (eV/atom)")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="upper left")
    ax_bottom.text(
        0.58,
        0.95,
        "\n".join(
            [
                "EOS uses E(V), not only E(a).",
                "For fcc Al with energy reported per atom,",
                "the comparison volume is Vatom = a^3 / 4.",
                f"OF-DFT B0 error vs benchmark = {((float(ofdft['bulk_modulus_GPa']) - benchmark_b0) / benchmark_b0 * 100.0):+.1f}%",
                f"QE B0 error vs benchmark = {((float(qe['bulk_modulus_GPa']) - benchmark_b0) / benchmark_b0 * 100.0):+.1f}%",
            ]
        ),
        transform=ax_bottom.transAxes,
        ha="left",
        va="top",
        fontsize=9.3,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.75", alpha=0.96),
    )

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare bulk fcc Al EOS metrics between QE and DFTpy OF-DFT.")
    ap.add_argument(
        "--indir",
        default="results/bulk_DFT_OFDFT_QE_DFTpy_20260416",
        help="Directory containing qe_bulk_a0_scan.csv and dftpy_bulk_a0_scan_dense.csv",
    )
    ap.add_argument("--qe-csv", default="qe_bulk_a0_scan.csv")
    ap.add_argument("--ofdft-csv", default="dftpy_bulk_a0_scan_dense.csv")
    ap.add_argument("--benchmark-label", default="Standard room-temperature reference")
    ap.add_argument("--benchmark-a0", type=float, default=AL_FCC_A0_BENCHMARK_ANG)
    ap.add_argument("--benchmark-b0", type=float, default=AL_FCC_BULK_MODULUS_BENCHMARK_GPA)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    indir = (ROOT / args.indir).resolve()
    indir.mkdir(parents=True, exist_ok=True)

    qe_scan = _read_scan(indir / args.qe_csv)
    ofdft_scan = _read_scan(indir / args.ofdft_csv)
    qe_metrics = _fit_metrics("QE (KS-DFT)", *qe_scan)
    ofdft_metrics = _fit_metrics("OF-DFT (TFvW)", *ofdft_scan)

    out_png = indir / "bulk_compare_FINAL.png"
    out_pdf = indir / "bulk_compare_FINAL.pdf"
    _plot(
        out_png,
        out_pdf,
        args.benchmark_label,
        float(args.benchmark_a0),
        float(args.benchmark_b0),
        qe_scan,
        ofdft_scan,
        qe_metrics,
        ofdft_metrics,
    )
    _write_comparison_csv(
        indir / "bulk_benchmark_comparison.csv",
        args.benchmark_label,
        float(args.benchmark_a0),
        float(args.benchmark_b0),
        [qe_metrics, ofdft_metrics],
    )
    _write_summary_txt(
        indir / "qe_dftpy_bulk_summary.txt",
        args.benchmark_label,
        float(args.benchmark_a0),
        float(args.benchmark_b0),
        qe_metrics,
        ofdft_metrics,
    )
    _write_readme(
        indir / "README.md",
        args.benchmark_label,
        float(args.benchmark_a0),
        float(args.benchmark_b0),
        qe_metrics,
        ofdft_metrics,
    )

    print(f"[bulk-eos] Wrote figure : {out_png}")
    print(f"[bulk-eos] Wrote table  : {indir / 'bulk_benchmark_comparison.csv'}")
    print(f"[bulk-eos] QE a0 = {float(qe_metrics['a0_A']):.6f} A, B0 = {float(qe_metrics['bulk_modulus_GPa']):.3f} GPa")
    print(f"[bulk-eos] OFDFT a0 = {float(ofdft_metrics['a0_A']):.6f} A, B0 = {float(ofdft_metrics['bulk_modulus_GPa']):.3f} GPa")


if __name__ == "__main__":
    main()
