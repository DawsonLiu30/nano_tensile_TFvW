from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_dftpy_out(path: Path) -> tuple[float, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 5:
        raise ValueError(f"Unexpected DFTpy output format in {path}")
    energy = float(lines[0].split(":")[-1].strip())
    stress = np.array(
        [
            [float(x) for x in lines[2].split()],
            [float(x) for x in lines[3].split()],
            [float(x) for x in lines[4].split()],
        ],
        dtype=float,
    )
    return energy, stress


def _formation_energy_from_total(pristine_energy_eV: float, vacancy_energy_eV: float, pristine_atoms: int) -> float:
    mu = float(pristine_energy_eV) / float(pristine_atoms)
    return float(vacancy_energy_eV - (pristine_atoms - 1) * mu)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_convergence(path: Path, rows: list[dict[str, object]], *, delta_threshold_eV: float) -> None:
    sizes = np.asarray([int(row["supercell_n"]) for row in rows], dtype=float)
    ef = np.asarray([float(row["vacancy_formation_energy_eV"]) for row in rows], dtype=float)
    deltas = np.asarray(
        [float(row["delta_from_previous_eV"]) if row["delta_from_previous_eV"] not in ("", None) else np.nan for row in rows],
        dtype=float,
    )

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8.6, 8.2), sharex=True)
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.10, top=0.93, hspace=0.18)

    ax_top.plot(sizes, ef, "-o", linewidth=2.0, markersize=6, color="#0b5d7a")
    for x, y in zip(sizes, ef):
        ax_top.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax_top.set_ylabel(r"$E_f^{vac}$ (eV / vacancy)")
    ax_top.set_title("Bulk fcc Al vacancy formation energy vs supercell size")
    ax_top.grid(True, alpha=0.3)

    ax_bottom.plot(sizes, deltas, "-o", linewidth=1.8, markersize=5, color="#c84c09")
    ax_bottom.axhline(float(delta_threshold_eV), color="0.4", linestyle="--", linewidth=1.0, label=f"target = {float(delta_threshold_eV):.3f} eV/vacancy")
    ax_bottom.set_xlabel("Supercell replication N (NxNxN)")
    ax_bottom.set_ylabel(r"$\Delta E_f^{vac}$ vs previous size (eV)")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="upper right")

    fig.savefig(path, dpi=300)
    plt.close(fig)


def _write_summary_txt(path: Path, rows: list[dict[str, object]], *, a0_A: float, delta_threshold_eV: float, fmax: float) -> None:
    lines = [
        "Bulk fcc Al vacancy supercell convergence summary",
        f"a0_A = {float(a0_A):.12f}",
        f"relaxation_fmax_eV_per_A = {float(fmax):.6f}",
        f"convergence_target_delta_eV_per_vacancy = {float(delta_threshold_eV):.6f}",
        "",
        r"Formation energy definition: E_f^vac = E_vac^(N-1) - ((N-1)/N) E_pristine^N",
        "",
    ]
    converged = None
    for row in rows:
        lines.append(
            f"N={int(row['supercell_n'])}: "
            f"E_f^vac = {float(row['vacancy_formation_energy_eV']):.6f} eV, "
            f"delta_prev = {row['delta_from_previous_eV'] if row['delta_from_previous_eV'] != '' else 'n/a'}"
        )
        if converged is None and str(row["within_delta_target"]).lower() == "true":
            converged = int(row["supercell_n"])
    lines.append("")
    if converged is not None:
        lines.append(f"First size meeting |ΔE_f^vac| <= {float(delta_threshold_eV):.6f} eV/vacancy: N = {converged}")
    else:
        lines.append(f"No tested size reached |ΔE_f^vac| <= {float(delta_threshold_eV):.6f} eV/vacancy yet.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Collect completed bulk vacancy supercell sub-results into one summary.")
    ap.add_argument("--rootdir", required=True, help="Root directory containing sc_8x8x8 etc.")
    ap.add_argument("--delta-threshold", type=float, default=0.02)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rootdir = Path(args.rootdir).expanduser().resolve()
    if not rootdir.exists():
        raise FileNotFoundError(f"Missing rootdir: {rootdir}")

    rows: list[dict[str, object]] = []
    a0_ref = None
    fmax = None
    for subdir in sorted(rootdir.glob("sc_*x*x*")):
        manifest_path = subdir / "manifest.json"
        pristine_out = subdir / "pristine_dftpy.out"
        vacancy_out = subdir / "vacancy_dftpy.out"
        if not all(path.exists() for path in [manifest_path, pristine_out, vacancy_out]):
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pristine_energy_eV, pristine_stress = _read_dftpy_out(pristine_out)
        vacancy_energy_eV, vacancy_stress = _read_dftpy_out(vacancy_out)
        pristine_n_atoms = int(manifest["pristine_n_atoms"])
        n = int(manifest["supercell_n"])
        a0_ref = float(manifest["a0_A"])
        fmax = float(manifest["fmax_eV_per_A"])

        vacancy_formation_energy_eV = _formation_energy_from_total(
            pristine_energy_eV=float(pristine_energy_eV),
            vacancy_energy_eV=float(vacancy_energy_eV),
            pristine_atoms=pristine_n_atoms,
        )
        mu_eV_per_atom = float(pristine_energy_eV) / float(pristine_n_atoms)
        row = {
            "supercell_n": n,
            "label": f"{n}x{n}x{n}",
            "pristine_n_atoms": pristine_n_atoms,
            "vacancy_n_atoms": pristine_n_atoms - 1,
            "cell_lx_A": f"{float(manifest['a0_A']) * n:.12f}",
            "cell_ly_A": f"{float(manifest['a0_A']) * n:.12f}",
            "cell_lz_A": f"{float(manifest['a0_A']) * n:.12f}",
            "volume_A3": f"{(float(manifest['a0_A']) ** 3) * (n ** 3):.12f}",
            "volume_per_atom_A3": f"{(float(manifest['a0_A']) ** 3) / 4.0:.12f}",
            "pristine_energy_eV": f"{float(pristine_energy_eV):.12f}",
            "pristine_energy_per_atom_eV": f"{float(pristine_energy_eV) / pristine_n_atoms:.12f}",
            "pristine_sigma_xx_GPa": f"{float(pristine_stress[0,0]):.12f}",
            "pristine_sigma_yy_GPa": f"{float(pristine_stress[1,1]):.12f}",
            "pristine_sigma_zz_GPa": f"{float(pristine_stress[2,2]):.12f}",
            "vacancy_energy_eV": f"{float(vacancy_energy_eV):.12f}",
            "vacancy_energy_per_atom_eV": f"{float(vacancy_energy_eV) / (pristine_n_atoms - 1):.12f}",
            "vacancy_sigma_xx_GPa": f"{float(vacancy_stress[0,0]):.12f}",
            "vacancy_sigma_yy_GPa": f"{float(vacancy_stress[1,1]):.12f}",
            "vacancy_sigma_zz_GPa": f"{float(vacancy_stress[2,2]):.12f}",
            "vacancy_mu_eV_per_atom": f"{mu_eV_per_atom:.12f}",
            "vacancy_formation_energy_eV": f"{float(vacancy_formation_energy_eV):.12f}",
            "vacancy_site_index": int(manifest["vacancy_site"]["index"]),
            "vacancy_site_distance_to_center_A": f"{float(manifest['vacancy_site']['distance_to_center_A']):.12f}",
            "relaxed_vacancy_max_force_eV_per_A": "",
            "delta_from_previous_eV": "",
            "within_delta_target": False,
        }
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No completed sub-results found under {rootdir}")

    rows.sort(key=lambda row: int(row["supercell_n"]))
    prev_ef = None
    for row in rows:
        ef = float(row["vacancy_formation_energy_eV"])
        if prev_ef is None:
            row["delta_from_previous_eV"] = ""
            row["within_delta_target"] = False
        else:
            delta = abs(ef - prev_ef)
            row["delta_from_previous_eV"] = f"{delta:.12f}"
            row["within_delta_target"] = bool(delta <= float(args.delta_threshold))
        prev_ef = ef

    _write_csv(rootdir / "summary.csv", rows)
    _plot_convergence(rootdir / "vacancy_formation_convergence.png", rows, delta_threshold_eV=float(args.delta_threshold))
    _write_summary_txt(
        rootdir / "summary.txt",
        rows,
        a0_A=float(a0_ref if a0_ref is not None else math.nan),
        delta_threshold_eV=float(args.delta_threshold),
        fmax=float(fmax if fmax is not None else math.nan),
    )
    print(f"[collect-bulk-vacancy] Wrote {rootdir / 'summary.csv'}")


if __name__ == "__main__":
    main()
