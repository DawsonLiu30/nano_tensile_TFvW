from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED = (
    Path.home()
    / "Desktop"
    / "latest_professor_pull_20260511"
    / "qe_bulk_b_convergence_20260506"
    / "processed_bulk_B_convergence"
)
DEFAULT_OUT = ROOT / "outputs" / "qe_bulk_convergence_replot_20260519"


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _style_axis(ax) -> None:
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_b0_vs_ecut(processed_dir: Path, out_dir: Path) -> Path:
    csv_path = processed_dir / "qe_bulk_ecut_fit_summary.csv"
    df = pd.read_csv(csv_path).sort_values("setting_value")

    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax.plot(df["setting_value"], df["B_GPa"], marker="o", lw=1.8, ms=4.5)
    ax.set_xlabel("Energy cutoff (eV)")
    ax.set_ylabel("Bulk modulus B0 (GPa)")
    ax.set_title("QE B0 convergence vs ecut")
    _style_axis(ax)

    out_path = out_dir / "qe_bulk_B0_vs_ecut_from_processed_csv.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_b0_vs_high_kmesh(processed_dir: Path, out_dir: Path) -> Path:
    csv_path = processed_dir / "qe_bulk_high_kmesh_fit_summary_with_k20.csv"
    df = pd.read_csv(csv_path).sort_values("k_value")

    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax.plot(df["k_value"], df["B_GPa"], marker="o", lw=1.8, ms=4.5, color="#0f6f8f")
    for _, row in df.iterrows():
        ax.annotate(
            f"{row['B_GPa']:.2f}",
            (row["k_value"], row["B_GPa"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("k-point mesh density n x n x n")
    ax.set_ylabel("Bulk modulus B0 (GPa)")
    ax.set_title("QE bulk B0 convergence in the high-k region")
    _style_axis(ax)

    out_path = out_dir / "qe_bulk_B0_vs_high_kmesh_from_processed_csv.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replot the two QE bulk EOS convergence figures used in the report "
            "from processed CSV data using pandas/matplotlib."
        )
    )
    parser.add_argument(
        "--processed-dir",
        default=str(DEFAULT_PROCESSED),
        help="Folder containing qe_bulk_ecut_fit_summary.csv and qe_bulk_high_kmesh_fit_summary_with_k20.csv.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT),
        help="Output folder for regenerated PNG figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = _resolve(args.processed_dir)
    out_dir = _resolve(args.out_dir)

    print(f"[input]  {processed_dir}")
    print(f"[output] {out_dir}")
    paths = [
        plot_b0_vs_ecut(processed_dir, out_dir),
        plot_b0_vs_high_kmesh(processed_dir, out_dir),
    ]
    for path in paths:
        print(f"[figure] {path}")


if __name__ == "__main__":
    main()
