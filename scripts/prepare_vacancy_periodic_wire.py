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
from ase.io import read, write

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ase_nanocrystal import build_circular_nanowire
from app.dft_engine import normalize_kedf_name, relax_atoms


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


def _parse_float_list(text: str, *, label: str) -> list[float]:
    values = []
    for chunk in str(text).split(","):
        token = chunk.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError(f"No valid {label} values were provided.")
    return values


def _scale_tag(scale: float) -> str:
    return f"{float(scale):.4f}".replace(".", "p")


def _write_structure_pair(base: Path, atoms) -> None:
    write(str(base.with_suffix(".xyz")), atoms)
    write(str(base.with_suffix(".vasp")), atoms, vasp5=True, direct=True)


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_dftpy_out(path: Path) -> tuple[float, np.ndarray]:
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


def _pick_equilibrium_scale(rows: list[dict[str, float]]) -> tuple[float, str]:
    scales = np.asarray([row["scale"] for row in rows], dtype=float)
    sigma = np.asarray([row["sigma_axis_GPa"] for row in rows], dtype=float)
    energy = np.asarray([row["energy_per_atom_eV"] for row in rows], dtype=float)
    order = np.argsort(scales)
    scales = scales[order]
    sigma = sigma[order]
    energy = energy[order]

    best_sigma_idx = int(np.argmin(np.abs(sigma)))
    best_energy_idx = int(np.argmin(energy))
    if best_sigma_idx == best_energy_idx:
        return float(scales[best_sigma_idx]), "best_scan_point_energy_and_sigma"
    if abs(float(sigma[best_sigma_idx])) <= 0.05:
        return float(scales[best_sigma_idx]), "best_scan_point_near_zero_sigma"

    for i in range(len(scales) - 1):
        s0, s1 = float(scales[i]), float(scales[i + 1])
        g0, g1 = float(sigma[i]), float(sigma[i + 1])
        if g0 == 0.0:
            return s0, "exact_zero_stress_scan_point"
        if g0 * g1 < 0.0:
            scale = s0 + (0.0 - g0) * (s1 - s0) / (g1 - g0)
            return float(scale), "linear_zero_stress_interpolation"

    best = int(np.argmin(np.abs(sigma)))
    return float(scales[best]), "min_abs_sigma_scan_point"


def _read_bulk_mu_from_validation(path: Path) -> tuple[float, float]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    zero_row = min(rows, key=lambda row: abs(float(row["strain"])))
    return float(zero_row["energy_per_atom_eV"]), float(zero_row["strain"])


def _latest_bulk_validation_csv() -> Path:
    candidates = sorted(
        (ROOT / "results").glob("bulk_Al_fcc_TFVW*/bulk_validation.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No bulk TFVW bulk_validation.csv was found under results/.")
    return candidates[0]


def _build_base_short_wire(
    *,
    a0: float,
    diameter_nm: float,
    vacuum: float,
    orientation: str,
):
    # length_z < one repeat keeps the primitive periodic wire along z.
    return build_circular_nanowire(
        a0=float(a0),
        diameter_nm=float(diameter_nm),
        length_z=1.0,
        vacuum=float(vacuum),
        orientation=str(orientation),
    )


def _extend_short_wire(base_atoms, *, min_short_lz: float, short_repeat_z: int) -> tuple[object, int]:
    base_lz = float(base_atoms.get_cell().lengths()[2])
    required_repeat_z = max(1, int(short_repeat_z))
    if required_repeat_z <= 1 and float(min_short_lz) > 0.0:
        required_repeat_z = max(1, int(math.ceil(float(min_short_lz) / max(base_lz, 1e-12))))
    extended = base_atoms.repeat((1, 1, required_repeat_z))
    return extended, required_repeat_z


def _scale_wire_z(atoms, scale: float):
    scaled = atoms.copy()
    cell = scaled.get_cell().array.copy()
    cell[2] *= float(scale)
    scaled.set_cell(cell, scale_atoms=True)
    return scaled


def _xy_center(atoms) -> tuple[float, float]:
    cell = atoms.get_cell().array
    return float(cell[0, 0] * 0.5), float(cell[1, 1] * 0.5)


def _choose_surface_vacancy_index(atoms, *, z_window_fraction: float) -> tuple[int, dict[str, float]]:
    pos = atoms.get_positions()
    cx, cy = _xy_center(atoms)
    z_center = float(np.mean(pos[:, 2]))
    radial = np.hypot(pos[:, 0] - cx, pos[:, 1] - cy)
    z_offset = np.abs(pos[:, 2] - z_center)
    lz = float(atoms.get_cell().lengths()[2])
    z_window = max(float(z_window_fraction) * lz, 1e-6)

    center_mask = z_offset <= z_window
    candidate_idx = np.where(center_mask)[0]
    if candidate_idx.size == 0:
        candidate_idx = np.arange(len(atoms), dtype=int)

    chosen = int(candidate_idx[int(np.argmax(radial[candidate_idx]))])
    site = {
        "index": chosen,
        "x_A": float(pos[chosen, 0]),
        "y_A": float(pos[chosen, 1]),
        "z_A": float(pos[chosen, 2]),
        "radial_distance_A": float(radial[chosen]),
        "z_center_offset_A": float(z_offset[chosen]),
        "z_window_A": float(z_window),
    }
    return chosen, site


def _remove_atom(atoms, atom_index: int):
    defect = atoms.copy()
    del defect[int(atom_index)]
    return defect


def _plot_axial_scan(path: Path, rows: list[dict[str, float]], *, kedf_name: str, eq_scale: float) -> None:
    scale_arr = np.asarray([row["scale"] for row in rows], dtype=float)
    energy_arr = np.asarray([row["energy_per_atom_eV"] for row in rows], dtype=float)
    sigma_arr = np.asarray([row["sigma_axis_GPa"] for row in rows], dtype=float)
    order = np.argsort(scale_arr)
    scale_arr = scale_arr[order]
    energy_arr = energy_arr[order]
    sigma_arr = sigma_arr[order]

    fig, (ax_energy, ax_sigma) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    ax_energy.plot(scale_arr, energy_arr, "-o", linewidth=1.4, markersize=4)
    ax_energy.axvline(eq_scale, color="0.5", linewidth=0.8)
    ax_energy.set_ylabel("Energy / atom (eV)")
    ax_energy.set_title(f"Vacancy branch pristine axial scan ({kedf_name})")
    ax_energy.grid(True, alpha=0.3)

    ax_sigma.plot(scale_arr, sigma_arr, "-o", linewidth=1.4, markersize=4)
    ax_sigma.axhline(0.0, color="0.5", linewidth=0.8)
    ax_sigma.axvline(eq_scale, color="0.5", linewidth=0.8, label=f"eq scale {eq_scale:.4f}")
    ax_sigma.set_xlabel("Axial scale factor of the elongated short unit cell")
    ax_sigma.set_ylabel("sigma_zz (GPa)")
    ax_sigma.grid(True, alpha=0.3)
    ax_sigma.legend()

    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Prepare an elongated periodic short nanowire (z > 10 A), pick a near-surface vacancy, "
            "and relax the vacancy structure for periodic tensile loading."
        )
    )
    ap.add_argument("--case", default="", help="Optional case name. Defaults to a timestamped name.")
    ap.add_argument("--diameter-nm", type=float, required=True, help="Nanowire diameter in nm.")
    ap.add_argument("--orientation", choices=["111", "100", "110"], default="111")
    ap.add_argument("--a0", type=float, default=4.05, help="Starting bulk lattice constant in Angstrom.")
    ap.add_argument("--vacuum", type=float, default=10.0, help="Vacuum padding in Angstrom.")
    ap.add_argument("--min-short-lz", type=float, default=10.0, help="Minimum short-cell z length in Angstrom.")
    ap.add_argument(
        "--short-repeat-z",
        type=int,
        default=0,
        help="Optional explicit z-repeat count for the short cell. If <= 0, choose the smallest repeat count that satisfies --min-short-lz.",
    )
    ap.add_argument(
        "--target-long-lz",
        type=float,
        default=200.0,
        help="Optional target long pristine z length in Angstrom for a reference replicated structure.",
    )
    ap.add_argument(
        "--scan-scales",
        default="0.94,0.95,0.96,0.97,0.98,0.99,1.00,1.01,1.02",
        help="Comma-separated axial scales for the pristine elongated short-cell quasistatic scan.",
    )
    ap.add_argument(
        "--vacancy-z-window-fraction",
        type=float,
        default=0.25,
        help="Restrict vacancy selection to atoms within this fraction of Lz around the short-cell z-center before choosing the outermost atom.",
    )
    ap.add_argument("--pp", default="al.gga.recpot", help="Pseudopotential file path.")
    ap.add_argument("--ecut", type=float, default=1000.0, help="Kinetic energy cutoff (eV).")
    ap.add_argument("--spacing", type=float, default=None, help="Override grid spacing (Angstrom).")
    ap.add_argument("--kedf", default="TFVW", help="DFTpy KEDF name. Examples: TFVW, SM, WT.")
    ap.add_argument("--fmax", type=float, default=0.02, help="Force convergence for each relaxation.")
    ap.add_argument("--relax-steps", type=int, default=120, help="Maximum relaxation steps per scan point.")
    ap.add_argument(
        "--bulk-validation-csv",
        default="",
        help="Optional bulk_validation.csv used to estimate bulk chemical potential for vacancy formation energy.",
    )
    ap.add_argument("--outdir", default="", help="Optional output directory. Defaults to cases/<case>/inputs.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    pp_path = Path(args.pp).expanduser().resolve()
    if not pp_path.exists():
        raise FileNotFoundError(f"pp file not found: {pp_path}")

    kedf_name = normalize_kedf_name(args.kedf)
    if args.spacing is not None:
        spacing = float(args.spacing)
        print(f"[vacancy-wire] Manual spacing override: {spacing:.6f} A")
    else:
        spacing = ecut_to_spacing_angstrom(float(args.ecut))
        print(f"[vacancy-wire] ecut={float(args.ecut):.2f} eV -> spacing={spacing:.6f} A")

    scan_scales = _parse_float_list(args.scan_scales, label="scan scale")
    bulk_validation_csv = (
        _latest_bulk_validation_csv()
        if not str(args.bulk_validation_csv).strip()
        else Path(args.bulk_validation_csv).expanduser().resolve()
    )
    if not bulk_validation_csv.exists():
        raise FileNotFoundError(f"bulk validation csv not found: {bulk_validation_csv}")
    bulk_mu_eV_per_atom, bulk_mu_strain = _read_bulk_mu_from_validation(bulk_validation_csv)

    case_name = args.case or f"paper_periodic_{args.orientation}_{float(args.diameter_nm):.1f}nm_vacancy_tfvw_{_ts()}"
    if args.outdir:
        outdir = Path(args.outdir).expanduser().resolve()
    else:
        outdir = (ROOT / "cases" / case_name / "inputs").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("[vacancy-wire] Preparing vacancy branch geometry")
    print(f"[vacancy-wire] case            : {case_name}")
    print(f"[vacancy-wire] outdir          : {outdir}")
    print(f"[vacancy-wire] diameter        : {float(args.diameter_nm):.4f} nm")
    print(f"[vacancy-wire] orientation     : [{args.orientation}]")
    print(f"[vacancy-wire] vacuum          : {float(args.vacuum):.4f} A")
    print(f"[vacancy-wire] min short lz    : {float(args.min_short_lz):.4f} A")
    print(f"[vacancy-wire] KEDF            : {args.kedf} -> {kedf_name}")
    print(f"[vacancy-wire] scan scales     : {', '.join(f'{x:.4f}' for x in scan_scales)}")
    print(f"[vacancy-wire] bulk mu source  : {bulk_validation_csv}")
    print("========================================")

    base_short = _build_base_short_wire(
        a0=float(args.a0),
        diameter_nm=float(args.diameter_nm),
        vacuum=float(args.vacuum),
        orientation=str(args.orientation),
    )
    base_short_lz = float(base_short.get_cell().lengths()[2])
    short_raw, short_repeat_z = _extend_short_wire(
        base_short,
        min_short_lz=float(args.min_short_lz),
        short_repeat_z=max(1, int(args.short_repeat_z)) if int(args.short_repeat_z) > 0 else 0,
    )
    short_raw_lz = float(short_raw.get_cell().lengths()[2])
    _write_structure_pair(outdir / "pristine_short_raw", short_raw)

    fixed_idx = np.array([], dtype=int)
    rows: list[dict[str, float]] = []
    relaxed_by_scale: dict[float, object] = {}

    for scale in scan_scales:
        tag = _scale_tag(scale)
        relaxed_vasp = outdir / f"pristine_scan_{tag}_relaxed.vasp"
        dftpy_out = outdir / f"pristine_scan_{tag}_dftpy.out"
        if relaxed_vasp.exists() and dftpy_out.exists():
            atoms_relaxed = read(str(relaxed_vasp))
            energy_ev, stress_gpa = _parse_dftpy_out(dftpy_out)
            print(f"[vacancy-wire][scan] scale={float(scale):.4f} resumed from existing files")
        else:
            atoms_guess = _scale_wire_z(short_raw, scale)
            _write_structure_pair(outdir / f"pristine_scan_{tag}_start", atoms_guess)

            atoms_relaxed, energy_ev, stress_gpa = relax_atoms(
                atoms_guess,
                pp_file=str(pp_path),
                spacing=float(spacing),
                fixed_idx=fixed_idx,
                kedf=kedf_name,
                fmax=float(args.fmax),
                steps=int(args.relax_steps),
                logfile=str(outdir / f"pristine_scan_{tag}_relax.log"),
                trajfile=str(outdir / f"pristine_scan_{tag}_relax.traj"),
                dftpy_outfile=str(dftpy_out),
                debug_fixed=False,
            )
            _write_structure_pair(outdir / f"pristine_scan_{tag}_relaxed", atoms_relaxed)

        relaxed_by_scale[float(scale)] = atoms_relaxed.copy()
        row = {
            "scale": float(scale),
            "lz_A": float(atoms_relaxed.get_cell().lengths()[2]),
            "energy_eV": float(energy_ev),
            "energy_per_atom_eV": float(energy_ev / len(atoms_relaxed)),
            "sigma_xx_GPa": float(stress_gpa[0, 0]),
            "sigma_yy_GPa": float(stress_gpa[1, 1]),
            "sigma_zz_GPa": float(stress_gpa[2, 2]),
            "sigma_axis_GPa": float(stress_gpa[2, 2]),
            "n_atoms": int(len(atoms_relaxed)),
        }
        rows.append(row)
        print(
            f"[vacancy-wire][scan] scale={row['scale']:.4f} "
            f"lz={row['lz_A']:.6f} A E/atom={row['energy_per_atom_eV']:.6f} eV "
            f"sigma_zz={row['sigma_zz_GPa']:+.6f} GPa"
        )

    _write_csv(outdir / "pristine_short_axial_scan.csv", rows)
    eq_scale, eq_source = _pick_equilibrium_scale(rows)

    pristine_eq_energy = None
    pristine_eq_stress = None
    short_pristine_eq = None
    eq_scale_rounded = None
    for scale in scan_scales:
        if math.isclose(eq_scale, float(scale), rel_tol=0.0, abs_tol=1e-12):
            eq_scale_rounded = float(scale)
            break

    if eq_scale_rounded is not None:
        short_pristine_eq = relaxed_by_scale[eq_scale_rounded].copy()
        eq_row = next(row for row in rows if math.isclose(float(row["scale"]), eq_scale_rounded, rel_tol=0.0, abs_tol=1e-12))
        pristine_eq_energy = float(eq_row["energy_eV"])
        pristine_eq_stress = np.diag(
            [
                float(eq_row["sigma_xx_GPa"]),
                float(eq_row["sigma_yy_GPa"]),
                float(eq_row["sigma_zz_GPa"]),
            ]
        )
    else:
        atoms_guess = _scale_wire_z(short_raw, eq_scale)
        _write_structure_pair(outdir / "pristine_short_equilibrium_start", atoms_guess)
        short_pristine_eq, pristine_eq_energy, pristine_eq_stress = relax_atoms(
            atoms_guess,
            pp_file=str(pp_path),
            spacing=float(spacing),
            fixed_idx=fixed_idx,
            kedf=kedf_name,
            fmax=float(args.fmax),
            steps=int(args.relax_steps),
            logfile=str(outdir / "pristine_short_equilibrium_relax.log"),
            trajfile=str(outdir / "pristine_short_equilibrium_relax.traj"),
            dftpy_outfile=str(outdir / "pristine_short_equilibrium_dftpy.out"),
            debug_fixed=False,
        )
        rows.append(
            {
                "scale": float(eq_scale),
                "lz_A": float(short_pristine_eq.get_cell().lengths()[2]),
                "energy_eV": float(pristine_eq_energy),
                "energy_per_atom_eV": float(pristine_eq_energy / len(short_pristine_eq)),
                "sigma_xx_GPa": float(pristine_eq_stress[0, 0]),
                "sigma_yy_GPa": float(pristine_eq_stress[1, 1]),
                "sigma_zz_GPa": float(pristine_eq_stress[2, 2]),
                "sigma_axis_GPa": float(pristine_eq_stress[2, 2]),
                "n_atoms": int(len(short_pristine_eq)),
            }
        )
        _write_csv(outdir / "pristine_short_axial_scan.csv", rows)
        print(f"[vacancy-wire] Refined pristine equilibrium scale at {eq_scale:.6f} ({eq_source})")

    _write_structure_pair(outdir / "pristine_short_equilibrium", short_pristine_eq)
    short_pristine_eq_lz = float(short_pristine_eq.get_cell().lengths()[2])

    long_repeat_z = max(1, int(math.ceil(float(args.target_long_lz) / max(short_pristine_eq_lz, 1e-12))))
    long_pristine_eq = short_pristine_eq.repeat((1, 1, long_repeat_z))
    _write_structure_pair(outdir / "pristine_long_equilibrium", long_pristine_eq)
    long_pristine_eq_lz = float(long_pristine_eq.get_cell().lengths()[2])

    vacancy_index, vacancy_site = _choose_surface_vacancy_index(
        short_pristine_eq,
        z_window_fraction=float(args.vacancy_z_window_fraction),
    )
    vacancy_start = _remove_atom(short_pristine_eq, vacancy_index)
    _write_structure_pair(outdir / "vacancy_short_start", vacancy_start)

    vacancy_eq, vacancy_energy, vacancy_stress = relax_atoms(
        vacancy_start,
        pp_file=str(pp_path),
        spacing=float(spacing),
        fixed_idx=fixed_idx,
        kedf=kedf_name,
        fmax=float(args.fmax),
        steps=int(args.relax_steps),
        logfile=str(outdir / "vacancy_short_relax.log"),
        trajfile=str(outdir / "vacancy_short_relax.traj"),
        dftpy_outfile=str(outdir / "vacancy_short_dftpy.out"),
        debug_fixed=False,
    )
    _write_structure_pair(outdir / "vacancy_short_equilibrium", vacancy_eq)
    _write_structure_pair(outdir / "vacancy_equilibrium", vacancy_eq)

    vacancy_formation_energy = float(vacancy_energy - pristine_eq_energy + bulk_mu_eV_per_atom)

    _plot_axial_scan(
        outdir / "pristine_short_axial_scan.png",
        rows,
        kedf_name=kedf_name,
        eq_scale=float(eq_scale),
    )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "case": case_name,
        "branch": "vacancy_periodic_short",
        "geometry": {
            "builder": "paper_circular_vacancy_short_branch",
            "orientation": args.orientation,
            "diameter_nm": float(args.diameter_nm),
            "vacuum_A": float(args.vacuum),
            "a0_input_A": float(args.a0),
            "base_short_lz_A": base_short_lz,
            "min_short_lz_target_A": float(args.min_short_lz),
            "short_repeat_z": int(short_repeat_z),
            "pristine_short_raw_lz_A": short_raw_lz,
            "pristine_short_equilibrium_lz_A": short_pristine_eq_lz,
            "pristine_long_repeat_z": int(long_repeat_z),
            "pristine_long_equilibrium_lz_A": long_pristine_eq_lz,
            "expected_layers_short": (3 * int(short_repeat_z)) if args.orientation == "111" else None,
            "expected_layers_long": (3 * int(short_repeat_z) * int(long_repeat_z)) if args.orientation == "111" else None,
        },
        "dft": {
            "pp": str(pp_path),
            "kedf": kedf_name,
            "spacing_A": float(spacing),
            "fmax": float(args.fmax),
            "relax_steps": int(args.relax_steps),
        },
        "equilibrium_selection": {
            "source": eq_source,
            "scale": float(eq_scale),
        },
        "pristine": {
            "n_atoms": int(len(short_pristine_eq)),
            "energy_eV": float(pristine_eq_energy),
            "energy_per_atom_eV": float(pristine_eq_energy / len(short_pristine_eq)),
            "sigma_zz_GPa": float(pristine_eq_stress[2, 2]),
        },
        "vacancy": {
            "n_atoms": int(len(vacancy_eq)),
            "energy_eV": float(vacancy_energy),
            "energy_per_atom_eV": float(vacancy_energy / len(vacancy_eq)),
            "sigma_zz_GPa": float(vacancy_stress[2, 2]),
            "formation_energy_bulk_mu_eV": float(vacancy_formation_energy),
            "bulk_mu_eV_per_atom": float(bulk_mu_eV_per_atom),
            "bulk_mu_source_csv": str(bulk_validation_csv),
            "bulk_mu_reference_strain": float(bulk_mu_strain),
            "selected_site": vacancy_site,
            "selection_rule": "largest radial distance among atoms within the central z-window",
        },
        "artifacts": {
            "pristine_short_raw": str(outdir / "pristine_short_raw.vasp"),
            "pristine_short_equilibrium": str(outdir / "pristine_short_equilibrium.vasp"),
            "pristine_long_equilibrium": str(outdir / "pristine_long_equilibrium.vasp"),
            "vacancy_short_start": str(outdir / "vacancy_short_start.vasp"),
            "vacancy_short_equilibrium": str(outdir / "vacancy_short_equilibrium.vasp"),
            "vacancy_equilibrium": str(outdir / "vacancy_equilibrium.vasp"),
            "scan_csv": str(outdir / "pristine_short_axial_scan.csv"),
            "scan_plot": str(outdir / "pristine_short_axial_scan.png"),
        },
    }
    (outdir / "vacancy_branch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[vacancy-wire] Wrote pristine short raw         : {outdir / 'pristine_short_raw.vasp'}")
    print(f"[vacancy-wire] Wrote pristine short equilibrium : {outdir / 'pristine_short_equilibrium.vasp'}")
    print(f"[vacancy-wire] Wrote pristine long equilibrium  : {outdir / 'pristine_long_equilibrium.vasp'}")
    print(f"[vacancy-wire] Wrote vacancy start             : {outdir / 'vacancy_short_start.vasp'}")
    print(f"[vacancy-wire] Wrote vacancy equilibrium       : {outdir / 'vacancy_short_equilibrium.vasp'}")
    print(f"[vacancy-wire] short eq lz                     : {short_pristine_eq_lz:.6f} A")
    print(f"[vacancy-wire] long eq lz                      : {long_pristine_eq_lz:.6f} A")
    print(f"[vacancy-wire] vacancy site index             : {vacancy_index}")
    print(f"[vacancy-wire] vacancy radial distance        : {vacancy_site['radial_distance_A']:.6f} A")
    print(f"[vacancy-wire] vacancy formation energy       : {vacancy_formation_energy:.6f} eV (bulk mu)")


if __name__ == "__main__":
    main()
