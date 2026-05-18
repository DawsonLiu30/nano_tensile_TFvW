from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "vacancy_nanocrystal_actual_points_20260518"

INPUT_CSV = Path(
    r"C:\Users\dawso\Desktop\vacancy_qe_ofdft_results_2026-04-25"
    r"\vacancy_qe_ofdft_results_2026-04-25\comparison_vacancy_ofdft_vs_qe.csv"
)

DIRECTIONS = ["100", "110", "111"]
COLORS = {
    "100": "#006b8f",
    "110": "#bf6b00",
    "111": "#2d7a3e",
}


def load_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)
    df["Direction"] = df["Direction"].astype(str)
    df["Radius"] = df["Radius"].astype(float)
    df["Atoms"] = df["Atoms"].astype(int)
    df = df.sort_values(["Direction", "Radius"]).reset_index(drop=True)
    return df


def plot_actual_points(df: pd.DataFrame, y_col: str, ylabel: str, title: str, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9), dpi=180)
    for ax, direction in zip(axes, DIRECTIONS):
        sub = df[df["Direction"] == direction].sort_values("Radius")
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.scatter(
            sub["Radius"],
            sub[y_col],
            s=70,
            color=COLORS[direction],
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_title(f"[{direction}]")
        ax.set_xlabel("Radius (A)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.22)
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def write_outputs(df: pd.DataFrame) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    actual_cols = [
        "Case",
        "Direction",
        "Radius",
        "ZR",
        "Atoms",
        "Energy_eV_atom_QE",
        "Energy_eV_atom_OFDFT",
        "Delta_Energy_eV_atom_OFDFT_minus_QE",
        "Final_Max_Force_eVA_QE",
        "Final_Max_Force_eVA_OFDFT",
    ]
    existing_cols = [col for col in actual_cols if col in df.columns]
    df[existing_cols].to_csv(OUTDIR / "vacancy_nanocrystal_actual_points_table.csv", index=False)

    plot_actual_points(
        df,
        "Energy_eV_atom_QE",
        "QE/PBE total energy (eV/atom)",
        "Actual QE/PBE vacancy nanostructure points only",
        OUTDIR / "actual_points_qe_energy_per_atom.png",
    )
    plot_actual_points(
        df,
        "Energy_eV_atom_OFDFT",
        "OFDFT total energy (eV/atom)",
        "Actual OFDFT vacancy nanostructure points only",
        OUTDIR / "actual_points_ofdft_energy_per_atom.png",
    )
    if "Delta_Energy_eV_atom_OFDFT_minus_QE" in df.columns:
        plot_actual_points(
            df,
            "Delta_Energy_eV_atom_OFDFT_minus_QE",
            "Raw OFDFT - QE (eV/atom)",
            "Actual raw OFDFT-QE energy difference points only",
            OUTDIR / "actual_points_raw_ofdft_minus_qe_eV_atom.png",
        )

    readme = f"""# Actual Vacancy Nanocrystal Points Only

Date: 2026-05-18

Input CSV:

```text
{INPUT_CSV}
```

No reference subtraction, no normalization, no endpoint anchoring, no fitted curve, and no predicted points were used.

Generated files:

- `actual_points_qe_energy_per_atom.png`
- `actual_points_ofdft_energy_per_atom.png`
- `actual_points_raw_ofdft_minus_qe_eV_atom.png`
- `vacancy_nanocrystal_actual_points_table.csv`

Actual data table:

```text
{df[existing_cols].to_string(index=False)}
```
"""
    (OUTDIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    df = load_data()
    write_outputs(df)
    print(f"Input: {INPUT_CSV}")
    print(f"Wrote: {OUTDIR}")
    for path in sorted(OUTDIR.glob("*")):
        print(path)


if __name__ == "__main__":
    main()
