from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path


RY_TO_EV = 13.605693122994
DFTPY_SERIES = {
    "TFVW": "dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_20260529",
    "SM": "dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_SM_spacing_20260529",
    "WT": "dftpy_vacancy_vcrelax_conv3x3x3_centered_lda_WT_spacing_20260529",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def truth(value: object) -> bool:
    return str(value).strip().lower() == "true"


def finite(value: object) -> bool:
    return not math.isnan(number(value))


def format_float(value: object, digits: int = 6) -> str:
    result = number(value)
    return "n/a" if math.isnan(result) else f"{result:.{digits}f}"


def kmesh_n(value: object) -> int:
    try:
        return int(str(value).split("x")[0])
    except (TypeError, ValueError):
        return 999


def copy_summary(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def collect_qe(local_base: Path, processed: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source = local_base / "raw" / "QE" / "processed_vcrelax" / "qe_vcrelax_vacancy_summary.csv"
    rows = read_csv(source)
    completed: list[dict[str, object]] = []
    provisional: list[dict[str, object]] = []

    for row in rows:
        output = dict(row)
        output["completed"] = truth(row.get("pristine_done")) and truth(row.get("vacancy_done"))
        ep = number(row.get("E_pristine_Ry"))
        ev = number(row.get("E_vacancy_Ry"))
        n_pristine = number(row.get("N_pristine"))
        n_vacancy = number(row.get("N_vacancy"))
        ef_provisional = math.nan
        if all(not math.isnan(value) for value in [ep, ev, n_pristine, n_vacancy]) and n_pristine:
            ef_provisional = (ev - (n_vacancy / n_pristine) * ep) * RY_TO_EV
        output["Ef_vac_provisional_eV"] = ef_provisional
        if output["completed"] and finite(row.get("Ef_vac_eV")):
            completed.append(output)
        else:
            provisional.append(output)

    write_csv(processed / "qe_completed_results.csv", completed)
    write_csv(processed / "qe_incomplete_provisional_results.csv", provisional)

    completed_kmesh = sorted(
        [row for row in completed if row.get("mode") == "kmesh_scan"],
        key=lambda row: kmesh_n(row.get("kmesh")),
    )
    completed_ecut = sorted(
        [row for row in completed if row.get("mode") == "ecut_scan"],
        key=lambda row: number(row.get("ecut_eV")),
    )
    write_csv(processed / "qe_completed_kmesh_scan.csv", completed_kmesh)
    write_csv(processed / "qe_completed_ecut_scan.csv", completed_ecut)
    return completed, provisional


def collect_dftpy(local_base: Path, processed: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    range_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []

    for kedf, series in DFTPY_SERIES.items():
        series_root = local_base / "raw" / f"DFTpy_{kedf}" / series
        source = series_root / "dftpy_conventional_spacing_summary.csv"
        rows = read_csv(source)
        copy_summary(source, processed / f"dftpy_{kedf}_spacing_summary.csv")
        copy_summary(
            series_root / "dftpy_vcrelax_fmax_summary.csv",
            processed / f"dftpy_{kedf}_fmax_summary.csv",
        )

        completed = [row for row in rows if truth(row.get("done")) and finite(row.get("Ef_vac_eV"))]
        values = [number(row["Ef_vac_eV"]) for row in completed]
        selected = min(completed, key=lambda row: abs(number(row.get("spacing_A")) - 0.20)) if completed else {}
        range_rows.append(
            {
                "method": "DFTpy",
                "xc": "LDA",
                "kedf": kedf,
                "completed_spacing_points": len(completed),
                "Ef_min_eV": min(values) if values else math.nan,
                "Ef_max_eV": max(values) if values else math.nan,
                "Ef_spread_eV": (max(values) - min(values)) if values else math.nan,
                "Ef_at_0p20A_eV": number(selected.get("Ef_vac_eV")),
                "status": "complete" if len(completed) == 6 else "incomplete",
            }
        )
        if selected:
            comparison_rows.append(
                {
                    "method": "DFTpy",
                    "xc": str(selected.get("xc", "LDA")),
                    "kedf": kedf,
                    "cell": str(selected.get("conventional_repeat_label", "conv_03x03x03")),
                    "N_pristine": selected.get("N_pristine", ""),
                    "N_vacancy": selected.get("N_vacancy", ""),
                    "vacancy_concentration_percent": selected.get("vacancy_concentration_percent", ""),
                    "reference_setting": "spacing_0p20A",
                    "Ef_vac_eV": selected.get("Ef_vac_eV", ""),
                    "status": "completed",
                }
            )

    write_csv(processed / "dftpy_kedf_range_summary.csv", range_rows)
    write_csv(processed / "dftpy_kedf_comparison_0p20A.csv", comparison_rows)
    return range_rows, comparison_rows


def write_final_method_comparison(
    processed: Path,
    qe_completed: list[dict[str, object]],
    qe_provisional: list[dict[str, object]],
    dftpy_comparison: list[dict[str, object]],
) -> None:
    rows: list[dict[str, object]] = []

    for row in sorted(
        [item for item in qe_completed if item.get("mode") == "kmesh_scan"],
        key=lambda item: kmesh_n(item.get("kmesh")),
    ):
        rows.append(
            {
                "method": "QE",
                "xc": "PBE",
                "pseudo": "Al_PAW_PBE.UPF",
                "kedf": "",
                "reference_setting": f"k={row['kmesh']}, ecut={number(row['ecut_eV']):.0f} eV",
                "Ef_vac_eV": row["Ef_vac_eV"],
                "status": "completed",
                "reporting_role": "QE completed k-mesh reference",
            }
        )

    for row in sorted(
        [item for item in qe_provisional if item.get("mode") == "kmesh_scan"],
        key=lambda item: kmesh_n(item.get("kmesh")),
    ):
        rows.append(
            {
                "method": "QE",
                "xc": "PBE",
                "pseudo": "Al_PAW_PBE.UPF",
                "kedf": "",
                "reference_setting": f"k={row['kmesh']}, ecut={number(row['ecut_eV']):.0f} eV",
                "Ef_vac_eV": row["Ef_vac_provisional_eV"],
                "status": "provisional_only",
                "reporting_role": "do not quote as final",
            }
        )

    for row in dftpy_comparison:
        rows.append(
            {
                "method": "DFTpy",
                "xc": row["xc"],
                "pseudo": "al.lda.recpot",
                "kedf": row["kedf"],
                "reference_setting": "spacing=0.20 A",
                "Ef_vac_eV": row["Ef_vac_eV"],
                "status": "completed",
                "reporting_role": "DFTpy KEDF comparison",
            }
        )

    write_csv(processed / "final_method_comparison.csv", rows)


def plot_dftpy(local_base: Path, processed: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    colors = {"TFVW": "#b94834", "WT": "#bd8b2e", "SM": "#287a65"}
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    plotted = False
    for kedf, series in DFTPY_SERIES.items():
        path = local_base / "raw" / f"DFTpy_{kedf}" / series / "dftpy_conventional_spacing_summary.csv"
        rows = [row for row in read_csv(path) if truth(row.get("done")) and finite(row.get("Ef_vac_eV"))]
        if not rows:
            continue
        rows.sort(key=lambda row: number(row.get("spacing_A")))
        ax.plot(
            [number(row["spacing_A"]) for row in rows],
            [number(row["Ef_vac_eV"]) for row in rows],
            "-o",
            label=kedf,
            color=colors[kedf],
            linewidth=1.8,
            markersize=5,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("DFTpy grid spacing (A)")
    ax.set_ylabel(r"$E_f^{vac}$ (eV)")
    ax.set_title("DFTpy LDA vacancy formation energy: KEDF dependence")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(processed / "dftpy_kedf_spacing_comparison.png", dpi=300)
    plt.close(fig)


def plot_qe(processed: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rows = read_csv(processed / "qe_completed_kmesh_scan.csv")
    rows = [row for row in rows if finite(row.get("Ef_vac_eV"))]
    if not rows:
        return
    rows.sort(key=lambda row: kmesh_n(row.get("kmesh")))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        [kmesh_n(row["kmesh"]) for row in rows],
        [number(row["Ef_vac_eV"]) for row in rows],
        "-o",
        color="#27648a",
        linewidth=1.8,
        markersize=5,
    )
    ax.set_xlabel("QE k-point mesh density (n x n x n)")
    ax.set_ylabel(r"$E_f^{vac}$ (eV)")
    ax.set_title("QE/PBE vc-relax: completed k-point results only")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(processed / "qe_completed_kmesh_scan.png", dpi=300)
    plt.close(fig)


def make_report(
    local_base: Path,
    processed: Path,
    qe_completed: list[dict[str, object]],
    qe_provisional: list[dict[str, object]],
    dftpy_ranges: list[dict[str, object]],
) -> None:
    qe_kmesh = sorted(
        [row for row in qe_completed if row.get("mode") == "kmesh_scan"],
        key=lambda row: kmesh_n(row.get("kmesh")),
    )
    qe_pending = [row for row in qe_provisional if row.get("mode") == "kmesh_scan"]
    dftpy_by_kedf = {str(row["kedf"]): row for row in dftpy_ranges}

    lines = [
        "# Latest Vacancy Benchmark Results",
        "",
        "This folder contains the latest pulled QE and DFTpy vacancy benchmark data.",
        "All structures use a conventional cubic fcc 3x3x3 centered-vacancy cell:",
        "`108 pristine atoms -> 107 vacancy atoms`, vacancy concentration `1/108 = 0.925926%`.",
        "",
        "## Raw Data Layout",
        "",
        "- `raw/QE`: QE/PBE PAW `vc-relax` inputs, outputs, structures, and logs.",
        "- `raw/DFTpy_TFVW`: DFTpy/LDA + TFVW full atom+cell relaxation data.",
        "- `raw/DFTpy_SM`: DFTpy/LDA + SM full atom+cell relaxation data.",
        "- `raw/DFTpy_WT`: DFTpy/LDA + WT full atom+cell relaxation data.",
        "- `raw/DFTpy_support`: DFTpy LDA pseudopotential and remote runner scripts.",
        "- `processed`: compact tables and plots generated from the pulled raw data.",
        "- `organizer_scripts`: local collection and packaging scripts.",
        "",
        "## QE/PBE vc-relax: Completed k-point Results",
        "",
        "| k-mesh | ecut (eV) | Ef_vac (eV) | status |",
        "|---|---:|---:|---|",
    ]
    for row in qe_kmesh:
        lines.append(
            f"| {row['kmesh']} | {format_float(row['ecut_eV'], 1)} | "
            f"{format_float(row['Ef_vac_eV'])} | completed |"
        )
    for row in qe_pending:
        lines.append(
            f"| {row['kmesh']} | {format_float(row['ecut_eV'], 1)} | "
            f"{format_float(row['Ef_vac_provisional_eV'])} | provisional only; output incomplete |"
        )

    lines += [
        "",
        "## DFTpy/LDA KEDF Dependence",
        "",
        "| KEDF | completed spacing points | Ef range (eV) | Ef at 0.20 A (eV) | status |",
        "|---|---:|---:|---:|---|",
    ]
    for kedf in ["TFVW", "WT", "SM"]:
        row = dftpy_by_kedf.get(kedf, {})
        lines.append(
            f"| {kedf} | {row.get('completed_spacing_points', 0)} | "
            f"{format_float(row.get('Ef_min_eV'))} - {format_float(row.get('Ef_max_eV'))} | "
            f"{format_float(row.get('Ef_at_0p20A_eV'))} | {row.get('status', 'missing')} |"
        )

    lines += [
        "",
        "## Current Interpretation",
        "",
        "- DFTpy vacancy formation energy is strongly KEDF-dependent.",
        "- With the same LDA local pseudopotential, same 3x3x3 centered-vacancy cell, and same full atom+cell relaxation workflow, TFVW strongly overestimates the vacancy energy.",
        "- WT substantially reduces the value but remains above the QE reference range.",
        "- SM gives a stable value near the completed QE/PBE vc-relax results.",
        "- QE incomplete cases remain explicitly provisional and must not be quoted as final values.",
        "",
        "## Key Review Files",
        "",
        "- `processed/dftpy_kedf_range_summary.csv`",
        "- `processed/dftpy_kedf_comparison_0p20A.csv`",
        "- `processed/qe_completed_kmesh_scan.csv`",
        "- `processed/qe_incomplete_provisional_results.csv`",
        "- `processed/dftpy_kedf_spacing_comparison.png`",
        "- `processed/qe_completed_kmesh_scan.png`",
        "",
    ]
    (local_base / "README_LATEST_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the latest pulled QE/DFTpy vacancy benchmark.")
    parser.add_argument("--rootdir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_base = Path(args.rootdir).expanduser().resolve()
    processed = local_base / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    qe_completed, qe_provisional = collect_qe(local_base, processed)
    dftpy_ranges, dftpy_comparison = collect_dftpy(local_base, processed)
    write_final_method_comparison(processed, qe_completed, qe_provisional, dftpy_comparison)
    plot_dftpy(local_base, processed)
    plot_qe(processed)
    make_report(local_base, processed, qe_completed, qe_provisional, dftpy_ranges)

    print("============================================================")
    print("Latest vacancy benchmark summary completed")
    print("============================================================")
    print(f"Root      : {local_base}")
    print(f"Processed : {processed}")
    print(f"Report    : {local_base / 'README_LATEST_RESULTS.md'}")


if __name__ == "__main__":
    main()
