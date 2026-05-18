from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PULL_ROOT = Path(r"C:\Users\dawso\Desktop\latest_professor_pull_20260511")
OUTDIR = ROOT / "outputs" / "true_vacancy_formation_20260518"

QE_CSV = (
    PULL_ROOT
    / "qe_vacancy_convergence_20260506"
    / "processed_vacancy_convergence"
    / "qe_vacancy_all_recursive_summary.csv"
)
DFTPY_SPACING_FIXED_QE_A0_CSV = (
    PULL_ROOT
    / "dftpy_vacancy_convergence_primitive4_qe_a0_20260508"
    / "summary.csv"
)
DFTPY_SPACING_OWN_A0_CSV = (
    PULL_ROOT
    / "dftpy_vacancy_convergence_primitive4_20260508"
    / "summary.csv"
)
DFTPY_SIZE_CSV = (
    PULL_ROOT
    / "dftpy_vacancy_size_primitive_qe_a0_spacing020_20260511"
    / "dftpy_primitive_size_summary.csv"
)


LITERATURE_LINES = [
    ("Gillan 1989 calc.", 0.56, "#4d4d4d", "--"),
    ("Experiment ref.", 0.66, "#7f7f7f", ":"),
]


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _kmesh_int(kmesh: str) -> int:
    return int(str(kmesh).split("x")[0])


def _clean_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def load_qe() -> pd.DataFrame:
    df = pd.read_csv(_require(QE_CSV))
    df["done"] = _clean_bool_series(df["pristine_done"]) & _clean_bool_series(df["vacancy_done"])
    df["ecut_eV_round"] = df["ecut_eV"].round().astype(int)
    df["kmesh_n"] = df["kmesh"].map(_kmesh_int)
    return df


def load_dftpy_spacing() -> pd.DataFrame:
    rows = []
    for label, path in [
        ("DFTpy fixed QE-a0", DFTPY_SPACING_FIXED_QE_A0_CSV),
        ("DFTpy own-a0", DFTPY_SPACING_OWN_A0_CSV),
    ]:
        df = pd.read_csv(_require(path))
        df["series"] = label
        df = df.rename(columns={"vacancy_formation_energy_eV": "Ef_vac_eV"})
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def load_dftpy_size() -> pd.DataFrame:
    df = pd.read_csv(_require(DFTPY_SIZE_CSV))
    df["done"] = _clean_bool_series(df["done"])
    return df


def _add_literature(ax, *, label_once: bool = False) -> None:
    for label, y, color, ls in LITERATURE_LINES:
        ax.axhline(y, color=color, linestyle=ls, linewidth=1.2, alpha=0.8, label=label if label_once else None)


def plot_qe_kmesh(qe: pd.DataFrame, out_png: Path) -> None:
    df = qe[(qe["mode"] == "kmesh") & qe["done"]].copy()
    df = df.sort_values("kmesh_n")

    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=180)
    ax.plot(df["kmesh_n"], df["Ef_vac_eV"], marker="o", linewidth=2.0, color="#006b8f")
    ax.scatter(df.loc[df["Ef_vac_eV"] < 0, "kmesh_n"], df.loc[df["Ef_vac_eV"] < 0, "Ef_vac_eV"], color="#b24a2a", zorder=4)
    _add_literature(ax, label_once=True)
    ax.set_title("QE vacancy formation energy vs k-point mesh")
    ax.set_xlabel("k-point mesh n in n x n x n")
    ax.set_ylabel("Vacancy formation energy (eV)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_qe_dense_ecut(qe: pd.DataFrame, out_png: Path) -> pd.DataFrame:
    dense = qe[(qe["mode"] == "dense_or_extra") & qe["done"]].copy()
    k5_600 = qe[(qe["mode"] == "kmesh") & qe["done"] & (qe["kmesh_n"] == 5)].copy()
    if not k5_600.empty:
        dense = pd.concat([dense, k5_600], ignore_index=True)
    dense = dense[dense["kmesh_n"] == 5].copy()
    dense = dense.sort_values("ecut_eV_round")

    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=180)
    ax.plot(dense["ecut_eV_round"], dense["Ef_vac_eV"], marker="o", linewidth=2.0, color="#285a84")
    _add_literature(ax, label_once=True)
    ax.set_title("QE dense-k vacancy cutoff check")
    ax.set_xlabel("ecut (eV), k = 5 x 5 x 5")
    ax.set_ylabel("Vacancy formation energy (eV)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return dense


def plot_dftpy_spacing(spacing: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    colors = {
        "DFTpy fixed QE-a0": "#9a6a00",
        "DFTpy own-a0": "#7a3b8f",
    }
    for label, sub in spacing.sort_values("spacing_A").groupby("series"):
        sub = sub.sort_values("spacing_A", ascending=False)
        ax.plot(sub["spacing_A"], sub["Ef_vac_eV"], marker="o", linewidth=2.0, label=label, color=colors.get(label))
    ax.invert_xaxis()
    ax.set_title("DFTpy vacancy formation energy vs real-space spacing")
    ax.set_xlabel("Grid spacing (A), smaller is denser")
    ax.set_ylabel("Vacancy formation energy (eV)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_dftpy_size(size: pd.DataFrame, out_png: Path) -> None:
    df = size[size["done"]].copy().sort_values("repeat_n")
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    ax.plot(df["repeat_n"], df["Ef_vac_eV"], marker="o", linewidth=2.0, color="#734222")
    ax.set_title("DFTpy primitive supercell-size check")
    ax.set_xlabel("Primitive supercell repeat n in n x n x n")
    ax.set_ylabel("Vacancy formation energy (eV)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_combined(qe: pd.DataFrame, spacing: pd.DataFrame, size: pd.DataFrame, dense: pd.DataFrame, out_png: Path) -> None:
    kmesh = qe[(qe["mode"] == "kmesh") & qe["done"]].copy().sort_values("kmesh_n")
    size_done = size[size["done"]].copy().sort_values("repeat_n")

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), dpi=180)
    ax = axes[0, 0]
    ax.plot(kmesh["kmesh_n"], kmesh["Ef_vac_eV"], marker="o", linewidth=2.0, color="#006b8f")
    _add_literature(ax, label_once=True)
    ax.set_title("QE k-point convergence")
    ax.set_xlabel("kmesh n")
    ax.set_ylabel("Ef^vac (eV)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 1]
    ax.plot(dense["ecut_eV_round"], dense["Ef_vac_eV"], marker="o", linewidth=2.0, color="#285a84")
    _add_literature(ax)
    ax.set_title("QE dense-k cutoff check")
    ax.set_xlabel("ecut (eV), k=5x5x5")
    ax.set_ylabel("Ef^vac (eV)")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    for label, sub in spacing.sort_values("spacing_A").groupby("series"):
        sub = sub.sort_values("spacing_A", ascending=False)
        ax.plot(sub["spacing_A"], sub["Ef_vac_eV"], marker="o", linewidth=2.0, label=label)
    ax.invert_xaxis()
    ax.set_title("DFTpy spacing convergence")
    ax.set_xlabel("spacing (A)")
    ax.set_ylabel("Ef^vac (eV)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    ax.plot(size_done["repeat_n"], size_done["Ef_vac_eV"], marker="o", linewidth=2.0, color="#734222")
    ax.set_title("DFTpy supercell-size check")
    ax.set_xlabel("primitive repeat n")
    ax.set_ylabel("Ef^vac (eV)")
    ax.grid(True, alpha=0.25)

    fig.suptitle("Vacancy formation energy from pulled archive data", fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def write_report(qe: pd.DataFrame, spacing: pd.DataFrame, size: pd.DataFrame, dense: pd.DataFrame) -> None:
    kmesh = qe[(qe["mode"] == "kmesh") & qe["done"]].copy().sort_values("kmesh_n")
    ecut = qe[(qe["mode"] == "ecut") & qe["done"]].copy().sort_values("ecut_eV_round")
    size_done = size[size["done"]].copy().sort_values("repeat_n")

    combined_rows = []
    for _, row in kmesh.iterrows():
        combined_rows.append({"source": "QE kmesh", "setting": row["kmesh"], "x": row["kmesh_n"], "Ef_vac_eV": row["Ef_vac_eV"]})
    for _, row in dense.iterrows():
        combined_rows.append({"source": "QE dense k=5 ecut", "setting": f"{row['ecut_eV_round']} eV", "x": row["ecut_eV_round"], "Ef_vac_eV": row["Ef_vac_eV"]})
    for _, row in spacing.iterrows():
        combined_rows.append({"source": row["series"], "setting": row["setting"], "x": row["spacing_A"], "Ef_vac_eV": row["Ef_vac_eV"]})
    for _, row in size_done.iterrows():
        combined_rows.append({"source": "DFTpy size fixed QE-a0 spacing=0.20 A", "setting": row["setting"], "x": row["repeat_n"], "Ef_vac_eV": row["Ef_vac_eV"]})
    combined = pd.DataFrame(combined_rows)
    combined.to_csv(OUTDIR / "vacancy_formation_combined_summary.csv", index=False)

    report = f"""# True Vacancy Formation Energy Plots

Date: 2026-05-18

All plots in this folder were generated with pandas/matplotlib from pulled local CSV data, not from manually drawn or reference-normalized curves.

## Input data

- QE vacancy summary: `{QE_CSV}`
- DFTpy spacing, fixed QE-a0: `{DFTPY_SPACING_FIXED_QE_A0_CSV}`
- DFTpy spacing, own-a0: `{DFTPY_SPACING_OWN_A0_CSV}`
- DFTpy primitive size: `{DFTPY_SIZE_CSV}`

## Generated figures

- `vacancy_true_formation_summary_2x2.png`
- `01_qe_vacancy_kmesh_convergence.png`
- `02_qe_dense_k5_ecut_convergence.png`
- `03_dftpy_spacing_convergence.png`
- `04_dftpy_supercell_size_convergence.png`

## Key numbers from archive-verified data

QE kmesh convergence at ecut = 600 eV:

```text
{kmesh[['kmesh', 'Ef_vac_eV', 'pristine_done', 'vacancy_done']].to_string(index=False)}
```

QE ecut convergence at k = 2x2x2:

```text
{ecut[['ecut_eV_round', 'kmesh', 'Ef_vac_eV']].to_string(index=False)}
```

QE dense k=5 cutoff check:

```text
{dense[['ecut_eV_round', 'kmesh', 'Ef_vac_eV']].to_string(index=False)}
```

DFTpy spacing convergence:

```text
{spacing[['series', 'spacing_A', 'ecut_analogue_eV', 'Ef_vac_eV']].to_string(index=False)}
```

DFTpy primitive supercell-size check:

```text
{size_done[['setting', 'repeat_n', 'N_pristine', 'N_vacancy', 'volume_A3', 'Ef_vac_eV']].to_string(index=False)}
```

## Important note about k=6

The local pulled archive at `latest_professor_pull_20260511` still contains an incomplete `k_06x06x06/vacancy_relax/relax.out`, so k=6 is not included in the archive-verified plot. The later terminal output reported `Ef_vac = 0.615195 eV` for k=6, but that completed output file is not present in this local archive copy.

## Interpretation

The discarded old vacancy nanostructure figure used a method-direction relative-energy normalization, which forced the largest-radius reference to zero. These new figures instead plot the physical vacancy formation energy:

```text
E_f^vac = E_vac^(N-1) - ((N-1)/N) E_pristine^N
```

The QE dense-k values are close to the literature vacancy-energy scale, while DFTpy/TFvW gives a much larger vacancy formation energy even after spacing and supercell-size checks. This is a physical/method discrepancy, not a plotting normalization artifact.
"""
    (OUTDIR / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    qe = load_qe()
    spacing = load_dftpy_spacing()
    size = load_dftpy_size()

    plot_qe_kmesh(qe, OUTDIR / "01_qe_vacancy_kmesh_convergence.png")
    dense = plot_qe_dense_ecut(qe, OUTDIR / "02_qe_dense_k5_ecut_convergence.png")
    plot_dftpy_spacing(spacing, OUTDIR / "03_dftpy_spacing_convergence.png")
    plot_dftpy_size(size, OUTDIR / "04_dftpy_supercell_size_convergence.png")
    plot_combined(qe, spacing, size, dense, OUTDIR / "vacancy_true_formation_summary_2x2.png")
    write_report(qe, spacing, size, dense)

    print("Wrote:", OUTDIR)
    for path in sorted(OUTDIR.glob("*")):
        print(path)


if __name__ == "__main__":
    main()
