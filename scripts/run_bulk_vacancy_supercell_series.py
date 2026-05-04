from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.build import bulk
from ase.io import read, write

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.aluminum_defaults import AL_FCC_A0_TFVW_ANG
from app.dft_engine import evaluate_atoms, normalize_kedf_name, relax_atoms


HA_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903


def ecut_to_spacing_angstrom(ecut_ev: float) -> float:
    if ecut_ev <= 0:
        raise ValueError(f"ecut must be positive, got {ecut_ev}")
    ecut_ha = ecut_ev / HA_TO_EV
    h_bohr = math.pi / math.sqrt(2.0 * ecut_ha)
    return h_bohr * BOHR_TO_ANG


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_int_list(text: str, *, label: str) -> list[int]:
    values: list[int] = []
    for chunk in str(text).split(","):
        token = chunk.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError(f"No valid {label} values were provided.")
    unique = sorted(set(values))
    if any(v <= 0 for v in unique):
        raise ValueError(f"All {label} values must be positive: {unique}")
    return unique


def _read_a0_from_summary(path: Path) -> float:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("a0_ref_A="):
            return float(line.split("=", 1)[1].strip())
    raise ValueError(f"Could not find a0_ref_A in {path}")


def _latest_bulk_summary_path() -> Path:
    candidates = sorted(
        (ROOT / "results").glob("bulk_Al_fcc_TFVW*/summary.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No bulk TFVW summary.txt was found under results/.")
    return candidates[0]


def _write_structure_pair(base: Path, atoms) -> None:
    write(str(base.with_suffix(".xyz")), atoms)
    write(str(base.with_suffix(".vasp")), atoms, direct=True, vasp5=True)


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


def _build_pristine_supercell(a0: float, repeats: int):
    return bulk("Al", "fcc", a=float(a0), cubic=True).repeat((int(repeats), int(repeats), int(repeats)))


def _choose_vacancy_index(atoms) -> tuple[int, dict[str, float]]:
    pos = atoms.get_positions()
    cell = atoms.get_cell().array
    center = 0.5 * (cell[0] + cell[1] + cell[2])
    dists = np.linalg.norm(pos - center[None, :], axis=1)
    idx = int(np.argmin(dists))
    return idx, {
        "index": idx,
        "x_A": float(pos[idx, 0]),
        "y_A": float(pos[idx, 1]),
        "z_A": float(pos[idx, 2]),
        "distance_to_center_A": float(dists[idx]),
    }


def _remove_atom(atoms, atom_index: int):
    defect = atoms.copy()
    del defect[int(atom_index)]
    return defect


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_summary_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _formation_energy_from_total(pristine_energy_eV: float, vacancy_energy_eV: float, pristine_atoms: int) -> float:
    mu = float(pristine_energy_eV) / float(pristine_atoms)
    return float(vacancy_energy_eV - (pristine_atoms - 1) * mu)


def _plot_convergence(path: Path, rows: list[dict[str, object]], *, delta_threshold_eV: float) -> None:
    sizes = np.asarray([int(row["supercell_n"]) for row in rows], dtype=float)
    ef = np.asarray([float(row["vacancy_formation_energy_eV"]) for row in rows], dtype=float)
    deltas = np.asarray([float(row["delta_from_previous_eV"]) if row["delta_from_previous_eV"] not in ("", None) else np.nan for row in rows], dtype=float)

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
    ap = argparse.ArgumentParser(description="Run bulk fcc Al vacancy formation energy convergence over supercell sizes.")
    ap.add_argument("--series", default="", help="Optional series name. Defaults to a timestamped results directory.")
    ap.add_argument("--supercells", default="8,10,12,14", help="Comma-separated cubic repetition sizes, e.g. 8,10,12,14")
    ap.add_argument("--bulk-summary", default="", help="Optional bulk summary.txt used to read a0_ref_A.")
    ap.add_argument(
        "--a0",
        type=float,
        default=None,
        help="Override bulk lattice constant in Angstrom. Defaults to latest TFvW bulk summary a0_ref_A.",
    )
    ap.add_argument("--pp", default="al.gga.recpot", help="Pseudopotential file path.")
    ap.add_argument("--ecut", type=float, default=1000.0, help="Kinetic energy cutoff (eV).")
    ap.add_argument("--spacing", type=float, default=None, help="Override grid spacing (Angstrom).")
    ap.add_argument("--kedf", default="TFVW", help="DFTpy KEDF name. Examples: TFVW, SM, WT.")
    ap.add_argument("--fmax", type=float, default=0.02, help="Force convergence threshold for vacancy relaxation (eV/Angstrom).")
    ap.add_argument("--relax-steps", type=int, default=200, help="Maximum relaxation steps for vacancy structures.")
    ap.add_argument("--delta-threshold", type=float, default=0.02, help="Target |ΔE_f^vac| threshold between neighboring supercell sizes (eV/vacancy).")
    ap.add_argument("--prepare-only", action="store_true", help="Only generate pristine/vacancy structures and manifests; do not run DFT calculations.")
    ap.add_argument("--plot-only", action="store_true", help="Rebuild convergence plots and summary from an existing summary.csv.")
    ap.add_argument("--outdir", default="", help="Optional output directory. Defaults to results/<series>.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    pp_path = Path(args.pp).expanduser().resolve()
    if not args.plot_only and not pp_path.exists():
        raise FileNotFoundError(f"pp file not found: {pp_path}")

    sizes = _parse_int_list(args.supercells, label="supercell sizes")
    kedf_name = normalize_kedf_name(args.kedf)
    if args.spacing is not None:
        spacing = float(args.spacing)
        print(f"[bulk-vacancy] Manual spacing override: {spacing:.6f} A")
    else:
        spacing = ecut_to_spacing_angstrom(float(args.ecut))
        print(f"[bulk-vacancy] ecut={float(args.ecut):.2f} eV -> spacing={spacing:.6f} A")

    bulk_summary = (
        _latest_bulk_summary_path()
        if not str(args.bulk_summary).strip()
        else Path(args.bulk_summary).expanduser().resolve()
    )
    a0_A = float(args.a0) if args.a0 is not None else _read_a0_from_summary(bulk_summary)

    series_name = args.series or f"bulk_fcc_vacancy_supercell_tfvw_{_ts()}"
    if args.outdir:
        outdir = Path(args.outdir).expanduser().resolve()
    else:
        outdir = (ROOT / "results" / series_name).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    summary_csv = outdir / "summary.csv"

    print("========================================")
    print("[bulk-vacancy] fcc Al vacancy supercell convergence")
    print(f"[bulk-vacancy] outdir          : {outdir}")
    print(f"[bulk-vacancy] supercells      : {', '.join(str(v) for v in sizes)}")
    print(f"[bulk-vacancy] bulk summary    : {bulk_summary}")
    print(f"[bulk-vacancy] a0_ref_A        : {a0_A:.12f}")
    print(f"[bulk-vacancy] KEDF            : {args.kedf} -> {kedf_name}")
    print(f"[bulk-vacancy] delta target    : {float(args.delta_threshold):.6f} eV/vacancy")
    print(f"[bulk-vacancy] relax fmax      : {float(args.fmax):.6f} eV/Angstrom")
    print("========================================")

    if args.plot_only:
        if not summary_csv.exists():
            raise FileNotFoundError(f"plot-only mode requires existing summary.csv: {summary_csv}")
        rows = _read_summary_csv(summary_csv)
        typed_rows = []
        for row in rows:
            typed_rows.append(
                {
                    "supercell_n": int(row["supercell_n"]),
                    "vacancy_formation_energy_eV": float(row["vacancy_formation_energy_eV"]),
                    "delta_from_previous_eV": row["delta_from_previous_eV"],
                    "within_delta_target": row["within_delta_target"],
                }
            )
        _plot_convergence(outdir / "vacancy_formation_convergence.png", typed_rows, delta_threshold_eV=float(args.delta_threshold))
        _write_summary_txt(
            outdir / "summary.txt",
            typed_rows,
            a0_A=float(a0_A),
            delta_threshold_eV=float(args.delta_threshold),
            fmax=float(args.fmax),
        )
        return

    rows: list[dict[str, object]] = []
    for n in sizes:
        subdir = outdir / f"sc_{n}x{n}x{n}"
        subdir.mkdir(parents=True, exist_ok=True)

        pristine = _build_pristine_supercell(a0=float(a0_A), repeats=int(n))
        pristine_n_atoms = int(len(pristine))
        cell_lengths = pristine.get_cell().lengths()
        vacancy_index, vacancy_site = _choose_vacancy_index(pristine)
        vacancy_start = _remove_atom(pristine, vacancy_index)

        _write_structure_pair(subdir / "pristine_raw", pristine)
        _write_structure_pair(subdir / "vacancy_start", vacancy_start)

        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "series": series_name,
            "supercell_n": int(n),
            "repeats": [int(n), int(n), int(n)],
            "a0_A": float(a0_A),
            "pp": str(pp_path),
            "kedf": kedf_name,
            "spacing_A": float(spacing),
            "fmax_eV_per_A": float(args.fmax),
            "relax_steps": int(args.relax_steps),
            "delta_target_eV_per_vacancy": float(args.delta_threshold),
            "pristine_n_atoms": pristine_n_atoms,
            "vacancy_site": vacancy_site,
        }
        (subdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if args.prepare_only:
            print(f"[bulk-vacancy] prepared only: {subdir}")
            continue

        pristine_relaxed_vasp = subdir / "pristine_relaxed.vasp"
        pristine_dftpy_out = subdir / "pristine_dftpy.out"
        vacancy_relaxed_vasp = subdir / "vacancy_relaxed.vasp"
        vacancy_dftpy_out = subdir / "vacancy_dftpy.out"

        if pristine_relaxed_vasp.exists() and pristine_dftpy_out.exists():
            pristine_relaxed = read(str(pristine_relaxed_vasp))
            pristine_energy_eV, pristine_stress = _read_dftpy_out(pristine_dftpy_out)
        else:
            pristine_relaxed, pristine_energy_eV, pristine_stress = evaluate_atoms(
                pristine.copy(),
                pp_file=str(pp_path),
                spacing=float(spacing),
                kedf=kedf_name,
                dftpy_outfile=str(pristine_dftpy_out),
            )
            _write_structure_pair(subdir / "pristine_relaxed", pristine_relaxed)

        if vacancy_relaxed_vasp.exists() and vacancy_dftpy_out.exists():
            vacancy_relaxed = read(str(vacancy_relaxed_vasp))
            vacancy_energy_eV, vacancy_stress = _read_dftpy_out(vacancy_dftpy_out)
        else:
            vacancy_relaxed, vacancy_energy_eV, vacancy_stress = relax_atoms(
                vacancy_start.copy(),
                pp_file=str(pp_path),
                spacing=float(spacing),
                fixed_idx=np.array([], dtype=int),
                kedf=kedf_name,
                fmax=float(args.fmax),
                steps=int(args.relax_steps),
                logfile=str(subdir / "vacancy_relax.log"),
                trajfile=str(subdir / "vacancy_relax.traj"),
                dftpy_outfile=str(vacancy_dftpy_out),
                debug_fixed=False,
            )
            _write_structure_pair(subdir / "vacancy_relaxed", vacancy_relaxed)

        vacancy_formation_energy_eV = _formation_energy_from_total(
            pristine_energy_eV=float(pristine_energy_eV),
            vacancy_energy_eV=float(vacancy_energy_eV),
            pristine_atoms=pristine_n_atoms,
        )
        vacancy_mu_eV_per_atom = float(pristine_energy_eV) / float(pristine_n_atoms)
        max_force_eVA = float(np.linalg.norm(vacancy_relaxed.get_forces(), axis=1).max())

        row: dict[str, object] = {
            "supercell_n": int(n),
            "label": f"{n}x{n}x{n}",
            "pristine_n_atoms": pristine_n_atoms,
            "vacancy_n_atoms": int(len(vacancy_relaxed)),
            "cell_lx_A": float(cell_lengths[0]),
            "cell_ly_A": float(cell_lengths[1]),
            "cell_lz_A": float(cell_lengths[2]),
            "volume_A3": float(pristine.get_volume()),
            "volume_per_atom_A3": float(pristine.get_volume() / pristine_n_atoms),
            "pristine_energy_eV": float(pristine_energy_eV),
            "pristine_energy_per_atom_eV": float(pristine_energy_eV / pristine_n_atoms),
            "pristine_sigma_xx_GPa": float(pristine_stress[0, 0]),
            "pristine_sigma_yy_GPa": float(pristine_stress[1, 1]),
            "pristine_sigma_zz_GPa": float(pristine_stress[2, 2]),
            "vacancy_energy_eV": float(vacancy_energy_eV),
            "vacancy_energy_per_atom_eV": float(vacancy_energy_eV / len(vacancy_relaxed)),
            "vacancy_sigma_xx_GPa": float(vacancy_stress[0, 0]),
            "vacancy_sigma_yy_GPa": float(vacancy_stress[1, 1]),
            "vacancy_sigma_zz_GPa": float(vacancy_stress[2, 2]),
            "vacancy_mu_eV_per_atom": vacancy_mu_eV_per_atom,
            "vacancy_formation_energy_eV": float(vacancy_formation_energy_eV),
            "vacancy_site_index": int(vacancy_site["index"]),
            "vacancy_site_distance_to_center_A": float(vacancy_site["distance_to_center_A"]),
            "relaxed_vacancy_max_force_eV_per_A": max_force_eVA,
            "delta_from_previous_eV": "",
            "within_delta_target": False,
        }
        rows.append(row)
        print(
            f"[bulk-vacancy] {row['label']}: "
            f"E_f^vac={float(row['vacancy_formation_energy_eV']):.6f} eV, "
            f"max|F|={float(row['relaxed_vacancy_max_force_eV_per_A']):.6f} eV/A"
        )

    if args.prepare_only:
        print(f"[bulk-vacancy] Structures prepared under {outdir}")
        return

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

    _write_csv(summary_csv, rows)
    _plot_convergence(outdir / "vacancy_formation_convergence.png", rows, delta_threshold_eV=float(args.delta_threshold))
    _write_summary_txt(
        outdir / "summary.txt",
        rows,
        a0_A=float(a0_A),
        delta_threshold_eV=float(args.delta_threshold),
        fmax=float(args.fmax),
    )

    print(f"[bulk-vacancy] Wrote summary csv : {summary_csv}")
    print(f"[bulk-vacancy] Wrote convergence : {outdir / 'vacancy_formation_convergence.png'}")
    print(f"[bulk-vacancy] Wrote summary txt : {outdir / 'summary.txt'}")


if __name__ == "__main__":
    main()
