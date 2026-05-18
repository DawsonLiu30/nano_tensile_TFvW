from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_bulk_eos import _fit_metrics


RY_TO_EV = 13.605693122994
ENERGY_RE = re.compile(r"!\s+total energy\s+=\s+([-0-9.]+)\s+Ry")


def _parse_total_energy_ev(path: Path) -> float | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = ENERGY_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1]) * RY_TO_EV


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _collect_scan_points(setting_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    a0_vals: list[float] = []
    e_vals: list[float] = []
    for point_dir in sorted(setting_dir.glob("a0_*")):
        try:
            a0_ang = float(point_dir.name.split("_", 1)[1])
        except Exception:
            continue
        energy_ev = _parse_total_energy_ev(point_dir / "scf.out")
        if energy_ev is None:
            continue
        a0_vals.append(a0_ang)
        e_vals.append(energy_ev)
    if len(a0_vals) < 4:
        raise ValueError(f"Need at least four completed scan points in {setting_dir}")
    return np.asarray(a0_vals, dtype=float), np.asarray(e_vals, dtype=float)


def _setting_sort_key(label: str) -> tuple[int, int]:
    if label.startswith("ecut_"):
        value = int(label.replace("ecut_", "").replace("eV", ""))
        return (0, value)
    if label.startswith("k_"):
        parts = label.replace("k_", "").split("x")
        return (1, int(parts[0]))
    return (9, 0)


def _plot_metric(path: Path, xs: list[float], ys: list[float], *, xlabel: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(xs, ys, "-o", linewidth=1.8, markersize=6, color="#1f77b4")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Collect QE bulk B0 convergence results.")
    ap.add_argument("--rootdir", default=str(ROOT / "results" / "qe_bulk_b_convergence_20260506"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rootdir = Path(args.rootdir).expanduser().resolve()
    if not rootdir.exists():
        raise FileNotFoundError(f"Root directory not found: {rootdir}")

    summary_rows: list[dict[str, object]] = []
    for mode_dir, mode in ((rootdir / "ecut_scan", "ecut_scan"), (rootdir / "kmesh_scan", "kmesh_scan")):
        if not mode_dir.exists():
            continue
        setting_rows: list[dict[str, object]] = []
        for setting_dir in sorted(mode_dir.iterdir(), key=lambda p: _setting_sort_key(p.name)):
            if not setting_dir.is_dir():
                continue
            a0_scan, e_scan = _collect_scan_points(setting_dir)
            metrics = _fit_metrics(setting_dir.name, a0_scan, e_scan)
            row: dict[str, object] = {
                "mode": mode,
                "setting": setting_dir.name,
                "a0_A": float(metrics["a0_A"]),
                "bulk_modulus_GPa": float(metrics["bulk_modulus_GPa"]),
                "bulk_modulus_prime": float(metrics["bulk_modulus_prime"]),
                "n_scan_points": int(a0_scan.size),
                "delta_a0_from_previous_A": "",
                "delta_B0_from_previous_GPa": "",
            }
            if setting_dir.name.startswith("ecut_"):
                row["ecut_eV"] = int(setting_dir.name.replace("ecut_", "").replace("eV", ""))
                row["kmesh"] = ""
            else:
                row["ecut_eV"] = ""
                row["kmesh"] = setting_dir.name.replace("k_", "")

            point_rows = [
                {"a0_ang": f"{float(a0):.6f}", "energy_ev_per_atom": f"{float(e):.12f}"}
                for a0, e in sorted(zip(a0_scan, e_scan), key=lambda pair: pair[0])
            ]
            _write_csv(setting_dir / "energy_scan.csv", point_rows)
            setting_rows.append(row)

        prev_a0 = None
        prev_b0 = None
        for row in setting_rows:
            if prev_a0 is not None:
                row["delta_a0_from_previous_A"] = f"{abs(float(row['a0_A']) - prev_a0):.12f}"
            if prev_b0 is not None:
                row["delta_B0_from_previous_GPa"] = f"{abs(float(row['bulk_modulus_GPa']) - prev_b0):.12f}"
            prev_a0 = float(row["a0_A"])
            prev_b0 = float(row["bulk_modulus_GPa"])

        summary_rows.extend(setting_rows)

        if mode == "ecut_scan" and setting_rows:
            xs = [float(row["ecut_eV"]) for row in setting_rows]
            _plot_metric(rootdir / "b0_vs_ecut.png", xs, [float(row["bulk_modulus_GPa"]) for row in setting_rows], xlabel="ecutwfc (eV)", ylabel="B0 (GPa)", title="QE bulk B0 convergence vs ecut")
            _plot_metric(rootdir / "a0_vs_ecut.png", xs, [float(row["a0_A"]) for row in setting_rows], xlabel="ecutwfc (eV)", ylabel="a0 (A)", title="QE bulk a0 convergence vs ecut")
        if mode == "kmesh_scan" and setting_rows:
            xs = [int(str(row["kmesh"]).split("x")[0]) for row in setting_rows]
            _plot_metric(rootdir / "b0_vs_kmesh.png", xs, [float(row["bulk_modulus_GPa"]) for row in setting_rows], xlabel="k-mesh (NxNxN)", ylabel="B0 (GPa)", title="QE bulk B0 convergence vs k-mesh")
            _plot_metric(rootdir / "a0_vs_kmesh.png", xs, [float(row["a0_A"]) for row in setting_rows], xlabel="k-mesh (NxNxN)", ylabel="a0 (A)", title="QE bulk a0 convergence vs k-mesh")

    _write_csv(rootdir / "summary.csv", summary_rows)
    print(f"[collect-qe-bulk-B] Wrote summary: {rootdir / 'summary.csv'}")


if __name__ == "__main__":
    main()
