from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


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


def _plot(path: Path, xs: list[float], ys: list[float], *, xlabel: str, title: str, gillan_ev: float, experiment_ev: float) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(xs, ys, "-o", linewidth=1.8, markersize=6, color="#1f77b4", label="Present QE workflow")
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
    ap = argparse.ArgumentParser(description="Collect QE vacancy convergence results.")
    ap.add_argument("--rootdir", required=True)
    ap.add_argument("--target-delta", type=float, default=0.03, help="Convergence target for |ΔE_f^vac| in eV.")
    ap.add_argument("--gillan-ev", type=float, default=0.56)
    ap.add_argument("--experiment-ev", type=float, default=0.66)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rootdir = Path(args.rootdir).expanduser().resolve()
    if not rootdir.exists():
        raise FileNotFoundError(f"Root directory not found: {rootdir}")

    manifest = json.loads((rootdir / "manifest.json").read_text(encoding="utf-8"))
    pristine_atoms = int(manifest["pristine_n_atoms"])

    summary_rows: list[dict[str, object]] = []
    for mode_dir, mode in ((rootdir / "ecut_scan", "ecut_scan"), (rootdir / "kmesh_scan", "kmesh_scan")):
        if not mode_dir.exists():
            continue
        rows_for_mode: list[dict[str, object]] = []
        for setting_dir in sorted(mode_dir.iterdir()):
            if not setting_dir.is_dir():
                continue
            pristine_ev = _parse_total_energy_ev(setting_dir / "pristine_scf" / "scf.out")
            vacancy_ev = _parse_total_energy_ev(setting_dir / "vacancy_relax" / "relax.out")
            if pristine_ev is None or vacancy_ev is None:
                continue
            ef_vac = float(vacancy_ev - ((pristine_atoms - 1) / pristine_atoms) * pristine_ev)
            row: dict[str, object] = {
                "mode": mode,
                "setting": setting_dir.name,
                "pristine_energy_eV": pristine_ev,
                "vacancy_energy_eV": vacancy_ev,
                "vacancy_formation_energy_eV": ef_vac,
                "delta_from_previous_eV": "",
                "within_target": False,
            }
            if setting_dir.name.startswith("ecut_"):
                row["ecut_eV"] = int(setting_dir.name.replace("ecut_", "").replace("eV", ""))
                row["kmesh"] = ""
            else:
                row["ecut_eV"] = ""
                row["kmesh"] = setting_dir.name.replace("k_", "")
            rows_for_mode.append(row)

        if mode == "ecut_scan":
            rows_for_mode.sort(key=lambda row: int(row["ecut_eV"]))
        else:
            rows_for_mode.sort(key=lambda row: int(str(row["kmesh"]).split("x")[0]))

        prev = None
        for row in rows_for_mode:
            ef = float(row["vacancy_formation_energy_eV"])
            if prev is not None:
                delta = abs(ef - prev)
                row["delta_from_previous_eV"] = f"{delta:.12f}"
                row["within_target"] = bool(delta <= float(args.target_delta))
            prev = ef

        summary_rows.extend(rows_for_mode)

        if rows_for_mode and mode == "ecut_scan":
            _plot(
                rootdir / "ef_vac_vs_ecut.png",
                [float(row["ecut_eV"]) for row in rows_for_mode],
                [float(row["vacancy_formation_energy_eV"]) for row in rows_for_mode],
                xlabel="ecutwfc (eV)",
                title="QE vacancy formation energy convergence vs ecut",
                gillan_ev=float(args.gillan_ev),
                experiment_ev=float(args.experiment_ev),
            )
        if rows_for_mode and mode == "kmesh_scan":
            _plot(
                rootdir / "ef_vac_vs_kmesh.png",
                [int(str(row["kmesh"]).split("x")[0]) for row in rows_for_mode],
                [float(row["vacancy_formation_energy_eV"]) for row in rows_for_mode],
                xlabel="k-mesh (NxNxN)",
                title="QE vacancy formation energy convergence vs k-mesh",
                gillan_ev=float(args.gillan_ev),
                experiment_ev=float(args.experiment_ev),
            )

    _write_csv(rootdir / "summary.csv", summary_rows)
    lines = [
        "QE vacancy convergence summary",
        f"supercell_n={manifest['supercell_n']}",
        f"pristine_n_atoms={manifest['pristine_n_atoms']}",
        f"vacancy_n_atoms={manifest['vacancy_n_atoms']}",
        f"target_delta_eV={float(args.target_delta):.6f}",
        f"gillan_1989_eV={float(args.gillan_ev):.6f}",
        f"experiment_reference_eV={float(args.experiment_ev):.6f}",
        "",
    ]
    for row in summary_rows:
        lines.append(
            f"{row['mode']} {row['setting']}: "
            f"E_f^vac={float(row['vacancy_formation_energy_eV']):.6f} eV, "
            f"delta_prev={row['delta_from_previous_eV'] or 'n/a'}"
        )
    (rootdir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[collect-qe-vacancy] Wrote summary: {rootdir / 'summary.csv'}")


if __name__ == "__main__":
    main()
