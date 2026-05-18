from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Collect DFTpy primitive-size vacancy results into CSV and plots.")
    ap.add_argument("--rootdir", required=True)
    ap.add_argument("--target-delta", type=float, default=0.03)
    return ap.parse_args()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def _plot(path: Path, xs: list[float], ys: list[float], *, xlabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(xs, ys, "-o", linewidth=1.8, markersize=6, color="#1f77b4")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$E_f^{vac}$ (eV)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rootdir = Path(args.rootdir).expanduser().resolve()
    scan_dir = rootdir / "size_scan"
    rows: list[dict[str, object]] = []

    for case_dir in sorted(scan_dir.glob("prim_*")):
        manifest_path = case_dir / "point_manifest.json"
        result_path = case_dir / "result.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

        row = {
            "setting": case_dir.name,
            "repeat_n": int(manifest.get("repeat_n", 0)) if manifest else 0,
            "N_pristine": int(manifest.get("pristine_n_atoms", 0)) if manifest else 0,
            "N_vacancy": int(manifest.get("vacancy_n_atoms", 0)) if manifest else 0,
            "volume_A3": float(manifest.get("volume_A3", math.nan)) if manifest else math.nan,
            "spacing_A": float(manifest.get("spacing_A", math.nan)) if manifest else math.nan,
            "fmax_eV_A": float(manifest.get("fmax_eV_per_A", math.nan)) if manifest else math.nan,
            "done": result_path.exists(),
            "pristine_energy_eV": math.nan,
            "vacancy_energy_eV": math.nan,
            "Ef_vac_eV": math.nan,
            "delta_from_previous_eV": math.nan,
            "within_target": False,
        }
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            row["pristine_energy_eV"] = float(result["pristine_energy_eV"])
            row["vacancy_energy_eV"] = float(result["vacancy_energy_eV"])
            row["Ef_vac_eV"] = float(result["vacancy_formation_energy_eV"])
        rows.append(row)

    rows.sort(key=lambda r: int(r["repeat_n"]))
    prev = math.nan
    for row in rows:
        ef = float(row["Ef_vac_eV"])
        if not math.isnan(ef) and not math.isnan(prev):
            delta = abs(ef - prev)
            row["delta_from_previous_eV"] = delta
            row["within_target"] = bool(delta <= float(args.target_delta))
        if not math.isnan(ef):
            prev = ef

    outcsv = rootdir / "dftpy_primitive_size_summary.csv"
    _write_csv(outcsv, rows)

    done_rows = [row for row in rows if not math.isnan(float(row["Ef_vac_eV"]))]
    if done_rows:
        _plot(
            rootdir / "ef_vac_vs_repeat_n.png",
            [float(row["repeat_n"]) for row in done_rows],
            [float(row["Ef_vac_eV"]) for row in done_rows],
            xlabel="primitive repeat n (n x n x n)",
            title="DFTpy vacancy formation energy vs primitive supercell size",
        )
        _plot(
            rootdir / "ef_vac_vs_inverse_natoms.png",
            [1.0 / float(row["N_pristine"]) for row in done_rows],
            [float(row["Ef_vac_eV"]) for row in done_rows],
            xlabel="1 / N_pristine",
            title="DFTpy vacancy formation energy vs inverse system size",
        )

    lines = [
        "DFTpy primitive supercell-size convergence summary",
        f"target_delta_eV={float(args.target_delta):.6f}",
        "",
    ]
    for row in rows:
        lines.append(
            f"{row['setting']}: repeat={row['repeat_n']}, N={row['N_pristine']} -> {row['N_vacancy']}, "
            f"V={row['volume_A3']}, E_f^vac={row['Ef_vac_eV']}, delta_prev={row['delta_from_previous_eV']}"
        )
    (rootdir / "dftpy_primitive_size_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("============================================================")
    print("DFTpy primitive supercell-size collection completed")
    print("============================================================")
    print(f"Output: {outcsv}")
    for row in rows:
        print(
            f"{row['setting']:14s} "
            f"N={int(row['N_pristine']):5d} -> {int(row['N_vacancy']):5d}  "
            f"V={float(row['volume_A3']):10.1f} A^3  "
            f"Ef={float(row['Ef_vac_eV']):10.6f} eV  "
            f"dEf={row['delta_from_previous_eV']}  done={row['done']}"
        )


if __name__ == "__main__":
    main()
