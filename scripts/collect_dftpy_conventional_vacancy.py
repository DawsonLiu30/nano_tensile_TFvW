from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def collect_scan(rootdir: Path, scan_name: str) -> list[dict[str, object]]:
    scan_root = rootdir / scan_name
    rows: list[dict[str, object]] = []
    if not scan_root.exists():
        return rows

    for case_dir in sorted(path for path in scan_root.iterdir() if path.is_dir()):
        manifest_path = case_dir / "point_manifest.json"
        result_path = case_dir / "result.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        result = read_json(result_path) if result_path.exists() else {}
        n_pristine = int(manifest["pristine_n_atoms"])
        ef = result.get("vacancy_formation_energy_eV", math.nan)
        repeat_label = str(manifest.get("conventional_repeat_label", ""))
        if not repeat_label:
            if "conventional_repeat_n" in manifest:
                n = int(manifest["conventional_repeat_n"])
                repeat_label = f"conv_{n:02d}x{n:02d}x{n:02d}"
            else:
                repeat = manifest.get("conventional_repeat", ["?", "?", "?"])
                repeat_label = "conv_" + "x".join(f"{int(v):02d}" for v in repeat)
        rows.append(
            {
                "scan": scan_name.replace("_scan", ""),
                "setting": str(manifest["setting"]),
                "done": result_path.exists(),
                "cell_basis": str(manifest["cell_basis"]),
                "conventional_repeat_label": repeat_label,
                "N_pristine": n_pristine,
                "N_vacancy": int(manifest["vacancy_n_atoms"]),
                "vacancy_concentration_percent": 100.0 / float(n_pristine),
                "spacing_A": float(manifest["spacing_A"]),
                "ecut_analogue_eV": float(manifest["ecut_analogue_eV"]),
                "fmax_eV_A": float(manifest["fmax_eV_per_A"]),
                "pristine_energy_eV": result.get("pristine_energy_eV", math.nan),
                "vacancy_energy_eV": result.get("vacancy_energy_eV", math.nan),
                "Ef_vac_eV": ef,
                "case_dir": str(case_dir),
            }
        )
    return rows


def add_deltas(rows: list[dict[str, object]], key: str) -> None:
    rows.sort(key=lambda row: float(row[key]))
    previous = math.nan
    for row in rows:
        ef = float(row["Ef_vac_eV"]) if row["done"] else math.nan
        if math.isnan(previous) or math.isnan(ef):
            row["delta_from_previous_eV"] = math.nan
        else:
            row["delta_from_previous_eV"] = abs(ef - previous)
        if not math.isnan(ef):
            previous = ef


def plot_scan(path: Path, rows: list[dict[str, object]], *, xkey: str, xlabel: str, title: str) -> None:
    done = [row for row in rows if row["done"] and not math.isnan(float(row["Ef_vac_eV"]))]
    if not done:
        return
    done.sort(key=lambda row: float(row[xkey]))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        [float(row[xkey]) for row in done],
        [float(row["Ef_vac_eV"]) for row in done],
        "-o",
        linewidth=1.8,
        markersize=6,
        color="#1f77b4",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$E_f^{vac}$ (eV)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect DFTpy conventional fcc vacancy spacing/size scans."
    )
    parser.add_argument("--rootdir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rootdir = Path(args.rootdir).expanduser().resolve()
    if not rootdir.exists():
        raise FileNotFoundError(f"Missing rootdir: {rootdir}")

    spacing_rows = collect_scan(rootdir, "spacing_scan")
    size_rows = collect_scan(rootdir, "size_scan")

    if spacing_rows:
        # Larger spacing first in the table mirrors the usual convergence scan order.
        spacing_rows.sort(key=lambda row: float(row["spacing_A"]), reverse=True)
        previous = math.nan
        for row in spacing_rows:
            ef = float(row["Ef_vac_eV"]) if row["done"] else math.nan
            row["delta_from_previous_eV"] = (
                math.nan if math.isnan(previous) or math.isnan(ef) else abs(ef - previous)
            )
            if not math.isnan(ef):
                previous = ef
        write_csv(rootdir / "dftpy_conventional_spacing_summary.csv", spacing_rows)
        plot_scan(
            rootdir / "dftpy_conventional_spacing_Ef.png",
            spacing_rows,
            xkey="spacing_A",
            xlabel="grid spacing (A)",
            title="DFTpy conventional fcc vacancy: spacing convergence",
        )

    if size_rows:
        add_deltas(size_rows, "N_pristine")
        write_csv(rootdir / "dftpy_conventional_size_summary.csv", size_rows)
        plot_scan(
            rootdir / "dftpy_conventional_size_Ef.png",
            size_rows,
            xkey="N_pristine",
            xlabel="pristine atom count",
            title="DFTpy conventional fcc vacancy: size/concentration convergence",
        )

    all_rows = spacing_rows + size_rows
    write_csv(rootdir / "dftpy_conventional_all_summary.csv", all_rows)

    print("============================================================")
    print("DFTpy conventional vacancy collection completed")
    print("============================================================")
    print(f"Root: {rootdir}")
    print(f"Spacing rows: {len(spacing_rows)}")
    print(f"Size rows   : {len(size_rows)}")
    print(f"All summary : {rootdir / 'dftpy_conventional_all_summary.csv'}")


if __name__ == "__main__":
    main()
