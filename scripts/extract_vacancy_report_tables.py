from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PULL_ROOT = Path.home() / "Desktop" / "latest_professor_pull_20260511"
OUTDIR = ROOT / "outputs" / "vacancy_report_tables_20260519"

QE_SUMMARY = (
    PULL_ROOT
    / "qe_vacancy_convergence_20260506"
    / "processed_vacancy_convergence"
    / "qe_vacancy_all_recursive_summary.csv"
)
DFTPY_FIXED_QE_A0 = PULL_ROOT / "dftpy_vacancy_convergence_primitive4_qe_a0_20260508" / "summary.csv"
PROFESSOR_UPDATE = PULL_ROOT / "professor_updates_20260512" / "latest_results_analysis_report_20260512.md"


def _resolve(text: str) -> Path:
    path = Path(text)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _kmesh_n(kmesh: str) -> int:
    return int(str(kmesh).split("x")[0])


def _done(df: pd.DataFrame) -> pd.Series:
    return df["pristine_done"].astype(str).str.lower().eq("true") & df["vacancy_done"].astype(str).str.lower().eq("true")


def _read_qe() -> pd.DataFrame:
    df = pd.read_csv(QE_SUMMARY)
    df["done"] = _done(df)
    df["ecut_eV_round"] = df["ecut_eV"].round().astype(int)
    df["kmesh_n"] = df["kmesh"].map(_kmesh_n)
    return df


def _read_k6_from_report() -> float | None:
    if not PROFESSOR_UPDATE.exists():
        return None
    text = PROFESSOR_UPDATE.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"`?6x6x6`?\s*\|\s*([0-9.]+)\s*\|", text)
    return float(match.group(1)) if match else None


def make_qe_dense_k_table(qe: pd.DataFrame, include_report_k6: bool) -> pd.DataFrame:
    table = qe[(qe["mode"] == "kmesh") & qe["done"] & (qe["ecut_eV_round"] == 600)].copy()
    table = table.sort_values("kmesh_n")
    notes = {
        "1x1x1": "unphysical",
        "2x2x2": "reference",
        "3x3x3": "high",
        "4x4x4": "near exp 0.66",
        "5x5x5": "near Gillan 0.56",
        "6x6x6": "moderate",
    }
    out = table[["kmesh", "Ef_vac_eV"]].copy()
    if include_report_k6 and "6x6x6" not in set(out["kmesh"]):
        k6 = _read_k6_from_report()
        if k6 is not None:
            out = pd.concat(
                [out, pd.DataFrame([{"kmesh": "6x6x6", "Ef_vac_eV": k6}])],
                ignore_index=True,
            )
    out["kmesh_n"] = out["kmesh"].map(_kmesh_n)
    out = out.sort_values("kmesh_n")
    out["note"] = out["kmesh"].map(notes).fillna("")
    out["Ef_vac_eV"] = out["Ef_vac_eV"].astype(float)
    return out[["kmesh", "Ef_vac_eV", "note"]]


def make_qe_cutoff_tables(qe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_2x2 = qe[(qe["mode"] == "ecut") & qe["done"] & (qe["kmesh"] == "2x2x2")].copy()
    cutoff_2x2 = cutoff_2x2.sort_values("ecut_eV_round")
    cutoff_2x2 = cutoff_2x2[["ecut_eV_round", "kmesh", "Ef_vac_eV"]].rename(
        columns={"ecut_eV_round": "ecut_eV"}
    )

    dense_k5 = qe[(qe["mode"] == "dense_or_extra") & qe["done"] & (qe["kmesh"] == "5x5x5")].copy()
    k5_600 = qe[(qe["mode"] == "kmesh") & qe["done"] & (qe["kmesh"] == "5x5x5")].copy()
    dense_k5 = pd.concat([dense_k5, k5_600], ignore_index=True)
    dense_k5 = dense_k5.sort_values("ecut_eV_round")
    dense_k5 = dense_k5[["ecut_eV_round", "kmesh", "Ef_vac_eV"]].rename(
        columns={"ecut_eV_round": "ecut_eV"}
    )
    return cutoff_2x2, dense_k5


def make_dftpy_spacing_table() -> pd.DataFrame:
    df = pd.read_csv(DFTPY_FIXED_QE_A0)
    return df[
        [
            "spacing_A",
            "ecut_analogue_eV",
            "vacancy_formation_energy_eV",
        ]
    ].rename(columns={"vacancy_formation_energy_eV": "Ef_vac_eV"})


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the three PPT vacancy tables from processed CSV data."
    )
    parser.add_argument("--out-dir", default=str(OUTDIR))
    parser.add_argument(
        "--no-report-k6",
        action="store_true",
        help="Do not append the 6x6x6 value recorded in professor_updates_20260512.",
    )
    args = parser.parse_args()

    out_dir = _resolve(args.out_dir)
    qe = _read_qe()

    dense_k = make_qe_dense_k_table(qe, include_report_k6=not args.no_report_k6)
    cutoff_2x2, dense_k5 = make_qe_cutoff_tables(qe)
    dftpy_spacing = make_dftpy_spacing_table()

    _write_table(dense_k, out_dir / "table_1_qe_dense_k_600eV.csv")
    _write_table(cutoff_2x2, out_dir / "table_2a_qe_cutoff_2x2x2.csv")
    _write_table(dense_k5, out_dir / "table_2b_qe_dense_k5_cutoff.csv")
    _write_table(dftpy_spacing, out_dir / "table_3_dftpy_spacing_fixed_qe_a0.csv")

    readme = f"""# Vacancy Report Tables

These CSV files reproduce the three PPT tables from processed data.

## Table 1: QE dense-k table at 600 eV

Output:

```text
table_1_qe_dense_k_600eV.csv
```

Main source:

```text
{QE_SUMMARY}
```

Note: the `6x6x6 = 0.615195 eV` point is appended from:

```text
{PROFESSOR_UPDATE}
```

because the local pulled `qe_vacancy_all_recursive_summary.csv` copy contains only completed archive points through `5x5x5`.

## Table 2: QE cutoff convergence

Outputs:

```text
table_2a_qe_cutoff_2x2x2.csv
table_2b_qe_dense_k5_cutoff.csv
```

Source:

```text
{QE_SUMMARY}
```

## Table 3: DFTpy spacing convergence

Output:

```text
table_3_dftpy_spacing_fixed_qe_a0.csv
```

Source:

```text
{DFTPY_FIXED_QE_A0}
```

This is a DFTpy real-space grid-spacing table, not a QE table and not a plotted figure.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"[output] {out_dir}")
    for path in sorted(out_dir.glob("*.csv")):
        print(f"[table]  {path}")


if __name__ == "__main__":
    main()
