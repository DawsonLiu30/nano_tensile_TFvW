from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    path: Path,
    xs: list[float],
    ys: list[float],
    *,
    xlabel: str,
    title: str,
    gillan_ev: float,
    experiment_ev: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(xs, ys, "-o", linewidth=1.8, markersize=6, color="#1f77b4", label="Present DFTpy workflow")
    ax.axhline(gillan_ev, color="#d62728", linestyle="--", linewidth=1.0, label=f"Gillan 1989 = {gillan_ev:.2f} eV")
    ax.axhline(experiment_ev, color="#2ca02c", linestyle=":", linewidth=1.0, label=f"Experiment ~ {experiment_ev:.2f} eV")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$E_f^{vac}$ (eV)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Collect DFTpy primitive-vacancy convergence results.")
    ap.add_argument("--rootdir", required=True)
    ap.add_argument("--target-delta", type=float, default=0.03)
    ap.add_argument("--gillan-ev", type=float, default=0.56)
    ap.add_argument("--experiment-ev", type=float, default=0.66)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rootdir = Path(args.rootdir).expanduser().resolve()
    if not rootdir.exists():
        raise FileNotFoundError(f"Missing rootdir: {rootdir}")

    scan_root = rootdir / "spacing_scan"
    if not scan_root.exists():
        scan_root = rootdir / "ecut_scan"

    rows: list[dict[str, object]] = []
    for case_dir in sorted(scan_root.iterdir()):
        if not case_dir.is_dir():
            continue
        result_path = case_dir / "result.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "setting": str(result["setting"]),
                "grid_parameter": str(result.get("grid_parameter", "spacing")),
                "spacing_A": float(result["spacing_A"]),
                "ecut_analogue_eV": float(result.get("ecut_analogue_eV", result.get("ecut_eV", 0.0))),
                "pristine_energy_eV": float(result["pristine_energy_eV"]),
                "vacancy_energy_eV": float(result["vacancy_energy_eV"]),
                "vacancy_formation_energy_eV": float(result["vacancy_formation_energy_eV"]),
                "delta_from_previous_eV": "",
                "within_target": False,
            }
        )

    rows.sort(key=lambda row: float(row["spacing_A"]), reverse=True)
    prev = None
    for row in rows:
        ef = float(row["vacancy_formation_energy_eV"])
        if prev is not None:
            delta = abs(ef - prev)
            row["delta_from_previous_eV"] = f"{delta:.12f}"
            row["within_target"] = bool(delta <= float(args.target_delta))
        prev = ef

    _write_csv(rootdir / "summary.csv", rows)
    _plot(
        rootdir / "ef_vac_vs_spacing.png",
        [float(row["spacing_A"]) for row in rows],
        [float(row["vacancy_formation_energy_eV"]) for row in rows],
        xlabel="grid spacing (A)",
        title="DFTpy primitive 4x4x4 vacancy convergence vs spacing",
        gillan_ev=float(args.gillan_ev),
        experiment_ev=float(args.experiment_ev),
    )
    _plot(
        rootdir / "ef_vac_vs_ecut_analogue.png",
        [float(row["ecut_analogue_eV"]) for row in rows],
        [float(row["vacancy_formation_energy_eV"]) for row in rows],
        xlabel="ecut-like label (eV)",
        title="DFTpy primitive 4x4x4 vacancy convergence vs ecut-like label",
        gillan_ev=float(args.gillan_ev),
        experiment_ev=float(args.experiment_ev),
    )
    lines = [
        "DFTpy primitive 4x4x4 vacancy convergence summary",
        "primary_grid_parameter=spacing",
        f"target_delta_eV={float(args.target_delta):.6f}",
        f"gillan_1989_eV={float(args.gillan_ev):.6f}",
        f"experiment_reference_eV={float(args.experiment_ev):.6f}",
        "",
    ]
    for row in rows:
        lines.append(
            f"{row['setting']}: spacing={float(row['spacing_A']):.6f} A, "
            f"ecut_analogue={float(row['ecut_analogue_eV']):.2f} eV, "
            f"E_f^vac={float(row['vacancy_formation_energy_eV']):.6f} eV, "
            f"delta_prev={row['delta_from_previous_eV'] or 'n/a'}"
        )
    (rootdir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[collect-dftpy-vacancy] Wrote summary: {rootdir / 'summary.csv'}")


if __name__ == "__main__":
    main()
