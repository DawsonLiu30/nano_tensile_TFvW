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

from app.aluminum_defaults import AL_FCC_A0_TFVW_ANG
from app.ase_nanocrystal import build_periodic_prism, cross_section_area_A2
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


def _build_short_periodic_wire(
    *,
    a0: float,
    diameter_nm: float,
    vacuum: float,
    orientation: str,
    cross_section_shape: str,
    shape_rotation_deg: float,
):
    # Any positive length shorter than one axial repeat produces nz=1. The
    # returned object is still infinite along z through periodic boundary
    # conditions; length_z only sets the simulation-cell repeat.
    return build_periodic_prism(
        a0=float(a0),
        diameter_nm=float(diameter_nm),
        length_z=1.0,
        vacuum=float(vacuum),
        orientation=str(orientation),
        cross_section_shape=str(cross_section_shape),
        shape_rotation_deg=float(shape_rotation_deg),
    )


def _scale_wire_z(atoms, scale: float):
    scaled = atoms.copy()
    cell = scaled.get_cell().array.copy()
    cell[2] *= float(scale)
    scaled.set_cell(cell, scale_atoms=True)
    return scaled


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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Prepare an axially periodic Al prism. Nanocolumns use a circular xy cross-section; "
            "nanocrystals use polygonal xy cross-sections such as hexagon or triangle."
        )
    )
    ap.add_argument("--case", default="", help="Optional case name. Defaults to a timestamped name.")
    ap.add_argument("--diameter-nm", type=float, required=True, help="Circumscribed cross-section diameter in nm.")
    ap.add_argument(
        "--cross-section-shape",
        choices=["circle", "hexagon", "triangle"],
        default="circle",
        help="xy shape: circle for nanocolumn; hexagon/triangle for nanocrystal.",
    )
    ap.add_argument("--shape-rotation-deg", type=float, default=0.0, help="Polygon rotation in the xy plane.")
    ap.add_argument("--orientation", choices=["111", "100", "110"], default="111")
    ap.add_argument(
        "--a0",
        type=float,
        default=AL_FCC_A0_TFVW_ANG,
        help="Starting bulk lattice constant in Angstrom (TFvW bulk equilibrium a0).",
    )
    ap.add_argument("--vacuum", type=float, default=10.0, help="Vacuum padding in Angstrom.")
    ap.add_argument("--replicate-z", type=int, default=30, help="Number of short-cell replications along z.")
    ap.add_argument(
        "--scan-scales",
        default="0.94,0.95,0.96,0.97,0.98,0.99,1.00,1.01,1.02",
        help="Comma-separated axial scales for the short-cell quasistatic scan.",
    )
    ap.add_argument("--pp", default="al.gga.recpot", help="Pseudopotential file path.")
    ap.add_argument("--ecut", type=float, default=1000.0, help="Kinetic energy cutoff (eV).")
    ap.add_argument("--spacing", type=float, default=None, help="Override grid spacing (Angstrom).")
    ap.add_argument("--kedf", default="TFVW", help="DFTpy KEDF name. Examples: TFVW, SM, WT.")
    ap.add_argument("--fmax", type=float, default=0.002, help="Force convergence for each short-cell relaxation.")
    ap.add_argument("--relax-steps", type=int, default=120, help="Maximum relaxation steps per scan point.")
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
        print(f"[paper-wire] Manual spacing override: {spacing:.6f} A")
    else:
        spacing = ecut_to_spacing_angstrom(float(args.ecut))
        print(f"[paper-wire] ecut={float(args.ecut):.2f} eV -> spacing={spacing:.6f} A")

    if int(args.replicate_z) <= 0:
        raise ValueError(f"replicate-z must be positive, got {args.replicate_z}")

    scan_scales = _parse_float_list(args.scan_scales, label="scan scale")
    shape_tag = str(args.cross_section_shape).lower()
    default_family = "nanocolumn" if shape_tag == "circle" else "nanocrystal"
    case_name = (
        args.case
        or f"{default_family}_{shape_tag}_periodic_{args.orientation}_{float(args.diameter_nm):.1f}nm_{kedf_name.lower()}_{_ts()}"
    )
    if args.outdir:
        outdir = Path(args.outdir).expanduser().resolve()
    else:
        outdir = (ROOT / "cases" / case_name / "inputs").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("[periodic-prism] Preparing axially periodic prism geometry")
    print(f"[paper-wire] case       : {case_name}")
    print(f"[paper-wire] outdir     : {outdir}")
    print(f"[paper-wire] shape      : {shape_tag}")
    print(f"[paper-wire] diameter   : {float(args.diameter_nm):.4f} nm (circumscribed)")
    print(f"[paper-wire] orientation: [{args.orientation}]")
    print(f"[paper-wire] vacuum     : {float(args.vacuum):.4f} A")
    print(f"[paper-wire] replicate  : {int(args.replicate_z)}")
    print(f"[paper-wire] KEDF       : {args.kedf} -> {kedf_name}")
    print(f"[paper-wire] scan scales: {', '.join(f'{x:.4f}' for x in scan_scales)}")
    print("========================================")

    short_raw = _build_short_periodic_wire(
        a0=float(args.a0),
        diameter_nm=float(args.diameter_nm),
        vacuum=float(args.vacuum),
        orientation=str(args.orientation),
        cross_section_shape=shape_tag,
        shape_rotation_deg=float(args.shape_rotation_deg),
    )
    _write_structure_pair(outdir / "short_raw", short_raw)

    fixed_idx = np.array([], dtype=int)
    rows: list[dict[str, float]] = []
    relaxed_by_scale: dict[float, object] = {}
    short_lz_raw = float(short_raw.get_cell().lengths()[2])

    for scale in scan_scales:
        tag = _scale_tag(scale)
        relaxed_vasp = outdir / f"scan_{tag}_relaxed.vasp"
        dftpy_out = outdir / f"scan_{tag}_dftpy.out"
        if relaxed_vasp.exists() and dftpy_out.exists():
            atoms_relaxed = read(str(relaxed_vasp))
            energy_ev, stress_gpa = _parse_dftpy_out(dftpy_out)
            print(f"[paper-wire][scan] scale={float(scale):.4f} resumed from existing files")
        else:
            atoms_guess = _scale_wire_z(short_raw, scale)
            _write_structure_pair(outdir / f"scan_{tag}_start", atoms_guess)

            atoms_relaxed, energy_ev, stress_gpa = relax_atoms(
                atoms_guess,
                pp_file=str(pp_path),
                spacing=float(spacing),
                fixed_idx=fixed_idx,
                kedf=kedf_name,
                fmax=float(args.fmax),
                steps=int(args.relax_steps),
                logfile=str(outdir / f"scan_{tag}_relax.log"),
                trajfile=str(outdir / f"scan_{tag}_relax.traj"),
                dftpy_outfile=str(dftpy_out),
                debug_fixed=False,
            )
            _write_structure_pair(outdir / f"scan_{tag}_relaxed", atoms_relaxed)

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
            f"[paper-wire][scan] scale={row['scale']:.4f} "
            f"lz={row['lz_A']:.6f} A "
            f"E/atom={row['energy_per_atom_eV']:.6f} eV "
            f"sigma_zz={row['sigma_zz_GPa']:+.6f} GPa"
        )

    _write_csv(outdir / "short_axial_scan.csv", rows)

    eq_scale, eq_source = _pick_equilibrium_scale(rows)
    eq_scale_rounded = None
    for scale in scan_scales:
        if math.isclose(eq_scale, float(scale), rel_tol=0.0, abs_tol=1e-12):
            eq_scale_rounded = float(scale)
            break

    if eq_scale_rounded is not None:
        short_eq = relaxed_by_scale[eq_scale_rounded].copy()
    else:
        atoms_guess = _scale_wire_z(short_raw, eq_scale)
        _write_structure_pair(outdir / "short_equilibrium_start", atoms_guess)
        short_eq, energy_eq, stress_eq = relax_atoms(
            atoms_guess,
            pp_file=str(pp_path),
            spacing=float(spacing),
            fixed_idx=fixed_idx,
            kedf=kedf_name,
            fmax=float(args.fmax),
            steps=int(args.relax_steps),
            logfile=str(outdir / "short_equilibrium_relax.log"),
            trajfile=str(outdir / "short_equilibrium_relax.traj"),
            dftpy_outfile=str(outdir / "short_equilibrium_dftpy.out"),
            debug_fixed=False,
        )
        rows.append(
            {
                "scale": float(eq_scale),
                "lz_A": float(short_eq.get_cell().lengths()[2]),
                "energy_eV": float(energy_eq),
                "energy_per_atom_eV": float(energy_eq / len(short_eq)),
                "sigma_xx_GPa": float(stress_eq[0, 0]),
                "sigma_yy_GPa": float(stress_eq[1, 1]),
                "sigma_zz_GPa": float(stress_eq[2, 2]),
                "sigma_axis_GPa": float(stress_eq[2, 2]),
                "n_atoms": int(len(short_eq)),
            }
        )
        print(f"[paper-wire] Refined equilibrium scale at {eq_scale:.6f} ({eq_source})")

    _write_csv(outdir / "short_axial_scan.csv", rows)
    _write_structure_pair(outdir / "short_equilibrium", short_eq)
    short_lz_eq = float(short_eq.get_cell().lengths()[2])

    long_eq = short_eq.repeat((1, 1, int(args.replicate_z)))
    _write_structure_pair(outdir / "long_equilibrium", long_eq)
    long_lz_eq = float(long_eq.get_cell().lengths()[2])

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
    ax_energy.set_title(f"Short periodic wire axial scan ({kedf_name})")
    ax_energy.grid(True, alpha=0.3)

    ax_sigma.plot(scale_arr, sigma_arr, "-o", linewidth=1.4, markersize=4)
    ax_sigma.axhline(0.0, color="0.5", linewidth=0.8)
    ax_sigma.axvline(eq_scale, color="0.5", linewidth=0.8, label=f"eq scale {eq_scale:.4f}")
    ax_sigma.set_xlabel("Axial scale factor of the short unit cell")
    ax_sigma.set_ylabel("sigma_zz (GPa)")
    ax_sigma.grid(True, alpha=0.3)
    ax_sigma.legend()

    fig.tight_layout()
    fig.savefig(str(outdir / "short_axial_scan.png"), dpi=150)
    plt.close(fig)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "case": case_name,
        "geometry": {
            "builder": "periodic_prism_short_to_reference_supercell",
            "orientation": args.orientation,
            "cross_section_shape": shape_tag,
            "shape_rotation_deg": float(args.shape_rotation_deg),
            "diameter_nm": float(args.diameter_nm),
            "cross_section_area_model_A2": cross_section_area_A2(shape_tag, 0.5 * float(args.diameter_nm) * 10.0),
            "vacuum_A": float(args.vacuum),
            "a0_input_A": float(args.a0),
            "short_raw_lz_A": short_lz_raw,
            "short_equilibrium_lz_A": short_lz_eq,
            "replicate_z": int(args.replicate_z),
            "long_equilibrium_lz_A": long_lz_eq,
            "axial_boundary_condition": "periodic_infinite_z",
            "expected_layers_short": 3 if args.orientation == "111" else None,
            "expected_layers_long": 3 * int(args.replicate_z) if args.orientation == "111" else None,
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
        "artifacts": {
            "short_raw": str(outdir / "short_raw.vasp"),
            "short_equilibrium": str(outdir / "short_equilibrium.vasp"),
            "long_equilibrium": str(outdir / "long_equilibrium.vasp"),
            "scan_csv": str(outdir / "short_axial_scan.csv"),
            "scan_plot": str(outdir / "short_axial_scan.png"),
        },
    }
    (outdir / "paper_periodic_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[paper-wire] Wrote short raw         : {outdir / 'short_raw.vasp'}")
    print(f"[paper-wire] Wrote short equilibrium : {outdir / 'short_equilibrium.vasp'}")
    print(f"[paper-wire] Wrote long equilibrium  : {outdir / 'long_equilibrium.vasp'}")
    print(f"[paper-wire] short raw lz           : {short_lz_raw:.6f} A")
    print(f"[paper-wire] short equilibrium lz   : {short_lz_eq:.6f} A")
    print(f"[paper-wire] long equilibrium lz    : {long_lz_eq:.6f} A")
    if args.orientation == "111":
        print(f"[paper-wire] expected long layers    : {3 * int(args.replicate_z)}")


if __name__ == "__main__":
    main()
