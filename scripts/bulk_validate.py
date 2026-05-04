from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.build import bulk
from ase.io import write
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.aluminum_defaults import AL_FCC_A0_TFVW_ANG
from app.dft_engine import evaluate_atoms, normalize_kedf_name


HA_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903
EV_PER_A3_TO_GPA = 160.21766208
GPA_TO_EV_PER_A3 = 1.0 / EV_PER_A3_TO_GPA


def ecut_to_spacing_angstrom(ecut_ev: float) -> float:
    if ecut_ev <= 0:
        raise ValueError(f"ecut must be positive, got {ecut_ev}")

    ecut_ha = ecut_ev / HA_TO_EV
    h_bohr = math.pi / math.sqrt(2.0 * ecut_ha)
    return h_bohr * BOHR_TO_ANG


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_float_list(text: str, *, label: str) -> np.ndarray:
    values = []
    for chunk in str(text).split(","):
        token = chunk.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError(f"No valid {label} points were provided.")
    arr = np.asarray(sorted(set(values)), dtype=float)
    return arr


def _parse_strains(text: str) -> np.ndarray:
    return _parse_float_list(text, label="strain")


def _strain_tag(strain: float) -> str:
    sign = "p" if strain >= 0.0 else "m"
    digits = f"{abs(strain):.4f}".replace(".", "")
    return f"{sign}{digits}"


def _build_bulk_atoms(a0: float, repeat: tuple[int, int, int]):
    return bulk("Al", "fcc", a=float(a0), cubic=True).repeat(repeat)


def _run_single_point(
    atoms,
    *,
    pp_path: Path,
    spacing: float,
    kedf_name: str,
    outfile: Path,
) -> tuple[float, np.ndarray]:
    _, energy_ev, stress_gpa = evaluate_atoms(
        atoms,
        pp_file=str(pp_path),
        spacing=float(spacing),
        kedf=kedf_name,
        dftpy_outfile=str(outfile),
    )
    return float(energy_ev), np.asarray(stress_gpa, dtype=float)


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, float]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({key: float(value) for key, value in row.items()})
    return rows


def _quadratic_minimum(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray] | tuple[None, None]:
    if x.size < 3:
        return None, None
    coeffs = np.polyfit(x, y, deg=2)
    a, b = coeffs[0], coeffs[1]
    if a <= 0.0:
        return None, coeffs
    x0 = float(-b / (2.0 * a))
    return x0, coeffs


def _sampled_minimum(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    idx = int(np.argmin(y))
    return float(x[idx]), float(y[idx])


def _read_summary(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def _fcc_volume_per_atom_from_a0(a0: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(a0, dtype=float)
    result = (arr**3) / 4.0
    if np.isscalar(a0):
        return float(result)
    return result


def _birch_murnaghan_energy(v_atom: np.ndarray, e0: float, v0: float, b0_eva3: float, b0_prime: float) -> np.ndarray:
    eta = (v0 / np.asarray(v_atom, dtype=float)) ** (2.0 / 3.0)
    term1 = (eta - 1.0) ** 3 * b0_prime
    term2 = (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
    return e0 + 9.0 * v0 * b0_eva3 / 16.0 * (term1 + term2)


def _bulk_modulus_guess_from_energy_vs_a(a0: np.ndarray, energy_per_atom: np.ndarray) -> float | None:
    if np.asarray(a0).size < 3:
        return None
    coeffs = np.polyfit(a0, energy_per_atom, deg=2)
    c2, c1 = coeffs[0], coeffs[1]
    if c2 <= 0.0:
        return None
    a0_fit = float(-c1 / (2.0 * c2))
    d2e_da2 = float(2.0 * c2)
    b0_eva3 = (4.0 / (9.0 * a0_fit)) * d2e_da2
    if b0_eva3 <= 0.0:
        return None
    return float(b0_eva3)


def _fit_birch_murnaghan_from_a0_scan(a0: np.ndarray, energy_per_atom: np.ndarray) -> dict[str, float] | None:
    if np.asarray(a0).size < 4:
        return None

    v_atom = np.asarray(_fcc_volume_per_atom_from_a0(a0), dtype=float)
    e_atom = np.asarray(energy_per_atom, dtype=float)
    idx_min = int(np.argmin(e_atom))

    e0_guess = float(e_atom[idx_min])
    v0_guess = float(v_atom[idx_min])
    b0_guess = _bulk_modulus_guess_from_energy_vs_a(np.asarray(a0, dtype=float), e_atom)
    if b0_guess is None:
        b0_guess = 0.5
    b0_prime_guess = 4.0

    lower = (-np.inf, float(v_atom.min()) * 0.95, 1.0e-4, 1.0)
    upper = (np.inf, float(v_atom.max()) * 1.05, 5.0, 12.0)
    p0 = (e0_guess, v0_guess, float(b0_guess), b0_prime_guess)

    try:
        params, _ = curve_fit(
            _birch_murnaghan_energy,
            v_atom,
            e_atom,
            p0=p0,
            bounds=(lower, upper),
            maxfev=20000,
        )
    except Exception:
        return None

    e0, v0, b0_eva3, b0_prime = [float(x) for x in params]
    a0_fit = float((4.0 * v0) ** (1.0 / 3.0))
    return {
        "e0_eV_per_atom": e0,
        "v0_atom_A3": v0,
        "a0_A": a0_fit,
        "bulk_modulus_eV_per_A3": b0_eva3,
        "bulk_modulus_GPa": b0_eva3 * EV_PER_A3_TO_GPA,
        "bulk_modulus_prime": b0_prime,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Low-memory bulk Al validation using DFTpy.")
    ap.add_argument("--pp", default="al.gga.recpot", help="Pseudopotential file path.")
    ap.add_argument("--ecut", type=float, default=1000.0, help="Kinetic energy cutoff (eV).")
    ap.add_argument("--spacing", type=float, default=None, help="Override grid spacing (Angstrom).")
    ap.add_argument("--kedf", default="TFVW", help="DFTpy KEDF name. Examples: TFVW, SM, WT.")
    ap.add_argument(
        "--a0",
        type=float,
        default=AL_FCC_A0_TFVW_ANG,
        help="Reference fcc lattice constant (Angstrom). Defaults to the TFvW bulk equilibrium value.",
    )
    ap.add_argument(
        "--a0-scan",
        default="",
        help="Optional comma-separated isotropic lattice scan before the strain scan.",
    )
    ap.add_argument("--repeat", nargs=3, type=int, default=[1, 1, 1], metavar=("NX", "NY", "NZ"))
    ap.add_argument("--axis", type=int, default=2, choices=[0, 1, 2], help="Strain axis.")
    ap.add_argument(
        "--strains",
        default="-0.01,-0.005,0.0,0.005,0.01",
        help="Comma-separated engineering strain points.",
    )
    ap.add_argument(
        "--fit-max-strain",
        type=float,
        default=0.005,
        help="Use |strain| <= this value for the linear stress fit.",
    )
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help="Rebuild plots from existing CSV files in --outdir without rerunning DFT calculations.",
    )
    ap.add_argument("--outdir", default="", help="Optional output directory.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    pp_path = Path(args.pp).expanduser().resolve()
    if not pp_path.exists():
        raise FileNotFoundError(f"pp file not found: {pp_path}")

    kedf_name = normalize_kedf_name(args.kedf)
    if args.spacing is not None:
        spacing = float(args.spacing)
        print(f"[bulk] Manual spacing override: {spacing:.6f} A")
    else:
        spacing = ecut_to_spacing_angstrom(float(args.ecut))
        print(f"[bulk] ecut={float(args.ecut):.2f} eV -> spacing={spacing:.6f} A")

    strains = _parse_strains(args.strains)
    a0_scan = _parse_float_list(args.a0_scan, label="a0") if str(args.a0_scan).strip() else np.array([], dtype=float)
    repeat = tuple(int(v) for v in args.repeat)
    if any(v <= 0 for v in repeat):
        raise ValueError(f"repeat must be positive, got {repeat}")

    if args.outdir:
        outdir = Path(args.outdir).expanduser().resolve()
    else:
        outdir = (Path("results") / f"bulk_Al_fcc_{kedf_name}_{_ts()}").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("[bulk] fcc Al validation")
    print(f"[bulk] PP      : {pp_path}")
    print(f"[bulk] KEDF    : {args.kedf} -> {kedf_name}")
    print(f"[bulk] a0      : {float(args.a0):.6f} A")
    if a0_scan.size:
        print(f"[bulk] a0 scan : {', '.join(f'{x:.4f}' for x in a0_scan)}")
    print(f"[bulk] repeat  : {repeat}")
    print(f"[bulk] axis    : {int(args.axis)}")
    print(f"[bulk] strains : {', '.join(f'{x:+.4f}' for x in strains)}")
    print(f"[bulk] outdir  : {outdir}")
    print("========================================")

    a0_fit = float("nan")
    eos_metrics: dict[str, float] | None = None
    a0_rows: list[dict[str, float]] = []
    rows: list[dict[str, float]] = []
    csv_path = outdir / "bulk_validation.csv"

    if args.plot_only:
        print("[bulk] Plot-only mode: reusing existing CSV outputs.")
        if (outdir / "a0_scan.csv").exists():
            a0_rows = _read_csv(outdir / "a0_scan.csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"plot-only mode requires existing CSV: {csv_path}")
        rows = _read_csv(csv_path)
        summary_map = _read_summary(outdir / "summary.txt")
        if "a0_ref_A" in summary_map:
            a0_fit = float(summary_map["a0_ref_A"])
        if all(key in summary_map for key in ["eos_v0_atom_A3", "eos_a0_A", "eos_bulk_modulus_GPa", "eos_bulk_modulus_prime"]):
            eos_metrics = {
                "v0_atom_A3": float(summary_map["eos_v0_atom_A3"]),
                "a0_A": float(summary_map["eos_a0_A"]),
                "bulk_modulus_GPa": float(summary_map["eos_bulk_modulus_GPa"]),
                "bulk_modulus_prime": float(summary_map["eos_bulk_modulus_prime"]),
                "bulk_modulus_eV_per_A3": float(summary_map.get("eos_bulk_modulus_eV_per_A3", "nan")),
                "e0_eV_per_atom": float(summary_map.get("eos_e0_eV_per_atom", "nan")),
            }
        elif a0_rows:
            a0_arr_tmp = np.asarray([row["a0_A"] for row in a0_rows], dtype=float)
            a0_energy_tmp = np.asarray([row["energy_per_atom_eV"] for row in a0_rows], dtype=float)
            a0_fit, _ = _quadratic_minimum(a0_arr_tmp, a0_energy_tmp)
            if not np.isfinite(a0_fit):
                a0_fit = float(_sampled_minimum(a0_arr_tmp, a0_energy_tmp)[0])
            eos_metrics = _fit_birch_murnaghan_from_a0_scan(a0_arr_tmp, a0_energy_tmp)
        else:
            a0_fit = float(args.a0)
    elif a0_scan.size:
        print("[bulk] Running isotropic a0 scan...")
        for a0_trial in a0_scan:
            atoms_a0 = _build_bulk_atoms(float(a0_trial), repeat)
            tag = f"a0_{float(a0_trial):.4f}".replace(".", "p")
            write(str(outdir / f"bulk_{tag}.vasp"), atoms_a0, direct=True, vasp5=True)
            energy_ev, stress_gpa = _run_single_point(
                atoms_a0,
                pp_path=pp_path,
                spacing=float(spacing),
                kedf_name=kedf_name,
                outfile=outdir / f"bulk_{tag}.dftpy.out",
            )
            hydro = float(np.trace(stress_gpa) / 3.0)
            row = {
                "a0_A": float(a0_trial),
                "energy_eV": float(energy_ev),
                "energy_per_atom_eV": float(energy_ev / len(atoms_a0)),
                "volume_A3": float(atoms_a0.get_volume()),
                "sigma_xx_GPa": float(stress_gpa[0, 0]),
                "sigma_yy_GPa": float(stress_gpa[1, 1]),
                "sigma_zz_GPa": float(stress_gpa[2, 2]),
                "hydrostatic_stress_GPa": hydro,
            }
            a0_rows.append(row)
            print(
                f"[bulk][a0] a0={row['a0_A']:.4f} "
                f"E/atom={row['energy_per_atom_eV']:.6f} eV "
                f"hydro={row['hydrostatic_stress_GPa']:+.6f} GPa"
            )

        _write_csv(outdir / "a0_scan.csv", a0_rows)
    if a0_rows:
        a0_arr = np.asarray([row["a0_A"] for row in a0_rows], dtype=float)
        a0_energy = np.asarray([row["energy_per_atom_eV"] for row in a0_rows], dtype=float)
        a0_fit, a0_coeffs = _quadratic_minimum(a0_arr, a0_energy)
        eos_metrics = _fit_birch_murnaghan_from_a0_scan(a0_arr, a0_energy)
        sampled_a0, sampled_energy = _sampled_minimum(a0_arr, a0_energy)
        if not np.isfinite(a0_fit) or not (float(a0_arr.min()) <= a0_fit <= float(a0_arr.max())):
            a0_fit = sampled_a0
            a0_fit_curve = None
            fit_energy = sampled_energy
        else:
            dense_a0 = np.linspace(float(a0_arr.min()), float(a0_arr.max()), 300)
            a0_fit_curve = np.polyval(a0_coeffs, dense_a0) if a0_coeffs is not None else None
            fit_energy = float(np.polyval(a0_coeffs, a0_fit)) if a0_coeffs is not None else sampled_energy

        fig, ax = plt.subplots(figsize=(8.4, 5.4))
        fig.subplots_adjust(left=0.12, right=0.96, bottom=0.14, top=0.90)
        ax.plot(a0_arr, a0_energy, "-o", linewidth=1.4, markersize=4, label="DFTpy")
        if a0_fit_curve is not None:
            ax.plot(dense_a0, a0_fit_curve, "--", linewidth=1.1, label="Quadratic fit")

        ax.axvline(a0_fit, color="0.5", linewidth=0.8)
        ax.scatter([sampled_a0], [sampled_energy], color="#c84c09", s=62, zorder=5)
        ax.scatter([a0_fit], [fit_energy], color="0.35", s=48, zorder=6)
        ax.set_xlabel("Lattice constant a0 (A)")
        ax.set_ylabel("Energy / atom (eV)")
        ax.set_title(f"Bulk fcc Al EOS scan ({kedf_name})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        ax.margins(x=0.05, y=0.08)
        fit_lines = [
            f"Sampled min = {sampled_a0:.4f} A",
            f"Quadratic min = {a0_fit:.4f} A",
        ]
        if eos_metrics is not None:
            fit_lines.append(f"EOS a0 = {eos_metrics['a0_A']:.4f} A")
            fit_lines.append(f"EOS B0 = {eos_metrics['bulk_modulus_GPa']:.1f} GPa")
        ax.text(
            0.61,
            0.86,
            "\n".join(fit_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.75", alpha=0.95),
        )
        fig.savefig(str(outdir / "a0_scan.png"), dpi=300)
        plt.close(fig)
        if not args.plot_only:
            if eos_metrics is not None:
                a0_fit = float(eos_metrics["a0_A"])
                print(
                    f"[bulk] Selected reference a0 from EOS fit: {a0_fit:.6f} A; "
                    f"B0 = {eos_metrics['bulk_modulus_GPa']:.6f} GPa"
                )
            else:
                print(f"[bulk] Selected reference a0 from quadratic scan fit: {a0_fit:.6f} A")
    else:
        a0_fit = float(args.a0)

    if not args.plot_only:
        atoms_ref = _build_bulk_atoms(float(a0_fit), repeat)
        write(str(outdir / "bulk_reference.vasp"), atoms_ref, direct=True, vasp5=True)
        write(str(outdir / "bulk_reference.xyz"), atoms_ref)

        rows = []
        for strain in strains:
            atoms = atoms_ref.copy()
            cell = atoms.get_cell().array.copy()
            cell[int(args.axis)] *= 1.0 + float(strain)
            atoms.set_cell(cell, scale_atoms=True)

            tag = _strain_tag(float(strain))
            write(str(outdir / f"bulk_strain_{tag}.vasp"), atoms, direct=True, vasp5=True)

            energy_ev, stress_gpa = _run_single_point(
                atoms,
                pp_path=pp_path,
                spacing=float(spacing),
                kedf_name=kedf_name,
                outfile=outdir / f"bulk_strain_{tag}.dftpy.out",
            )

            row = {
                "strain": float(strain),
                "energy_eV": float(energy_ev),
                "energy_per_atom_eV": float(energy_ev / len(atoms)),
                "volume_A3": float(atoms.get_volume()),
                "sigma_xx_GPa": float(stress_gpa[0, 0]),
                "sigma_yy_GPa": float(stress_gpa[1, 1]),
                "sigma_zz_GPa": float(stress_gpa[2, 2]),
                "sigma_axis_GPa": float(stress_gpa[int(args.axis), int(args.axis)]),
                "sigma_xy_GPa": float(stress_gpa[0, 1]),
                "sigma_xz_GPa": float(stress_gpa[0, 2]),
                "sigma_yz_GPa": float(stress_gpa[1, 2]),
            }
            rows.append(row)
            print(
                f"[bulk] strain={row['strain']:+.4f} "
                f"E/atom={row['energy_per_atom_eV']:.6f} eV "
                f"sigma_zz={row['sigma_zz_GPa']:+.6f} GPa "
                f"sigma_axis={row['sigma_axis_GPa']:+.6f} GPa"
            )

        _write_csv(csv_path, rows)
    else:
        atoms_ref = _build_bulk_atoms(float(a0_fit), repeat)

    strain_arr = np.asarray([row["strain"] for row in rows], dtype=float)
    energy_arr = np.asarray([row["energy_per_atom_eV"] for row in rows], dtype=float)
    sigma_axis_arr = np.asarray([row["sigma_axis_GPa"] for row in rows], dtype=float)

    fit_mask = np.abs(strain_arr) <= float(args.fit_max_strain)
    slope = float("nan")
    intercept = float("nan")
    fit_curve = None
    if int(np.sum(fit_mask)) >= 2:
        slope, intercept = np.polyfit(strain_arr[fit_mask], sigma_axis_arr[fit_mask], deg=1)
        fit_curve = slope * strain_arr + intercept

    energy_fit = None
    if strain_arr.size >= 3:
        coeffs = np.polyfit(strain_arr, energy_arr, deg=2)
        dense_strain = np.linspace(float(strain_arr.min()), float(strain_arr.max()), 300)
        energy_fit = np.polyval(coeffs, dense_strain)
        energy_fit_min = float(-coeffs[1] / (2.0 * coeffs[0])) if coeffs[0] > 0.0 else None
    else:
        dense_strain = None
        energy_fit_min = None

    fig, (ax_energy, ax_stress) = plt.subplots(2, 1, figsize=(8.6, 8.2), sharex=True)
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.10, top=0.92, hspace=0.14)
    ax_energy.plot(strain_arr, energy_arr, "-o", linewidth=1.4, markersize=4, label="DFTpy")
    if energy_fit is not None and dense_strain is not None:
        ax_energy.plot(dense_strain, energy_fit, "--", linewidth=1.1, label="Quadratic fit")
    energy_min_strain, energy_min_value = _sampled_minimum(strain_arr, energy_arr)
    ax_energy.scatter([energy_min_strain], [energy_min_value], color="#c84c09", s=55, zorder=5)
    if energy_fit_min is not None and float(strain_arr.min()) <= energy_fit_min <= float(strain_arr.max()):
        fit_min_energy = float(np.polyval(coeffs, energy_fit_min))
        ax_energy.scatter([energy_fit_min], [fit_min_energy], color="0.35", s=45, zorder=5)
    else:
        fit_min_energy = energy_min_value
    ax_energy.set_ylabel("Energy / atom (eV)")
    ax_energy.set_title(f"Bulk fcc Al validation ({kedf_name})")
    ax_energy.grid(True, alpha=0.3)
    ax_energy.legend(loc="upper left")
    ax_energy.margins(x=0.05, y=0.10)

    ax_stress.plot(strain_arr, sigma_axis_arr, "-o", linewidth=1.4, markersize=4, label="Axial stress")
    if fit_curve is not None:
        ax_stress.plot(strain_arr, fit_curve, "--", linewidth=1.1, label=f"Linear fit: {slope:.2f} GPa")
    ax_stress.axhline(0.0, color="0.5", linewidth=0.8)
    zero_cross = None
    if np.isfinite(intercept):
        zero_cross = float(-intercept / slope) if np.isfinite(slope) and abs(slope) > 1e-12 else None
        if zero_cross is not None:
            ax_stress.axvline(zero_cross, color="0.65", linewidth=0.8, linestyle=":")
            ax_stress.scatter([zero_cross], [0.0], color="0.35", s=42, zorder=6)
    ax_stress.set_xlabel("Applied engineering strain")
    ax_stress.set_ylabel("sigma_axis (GPa)")
    ax_stress.grid(True, alpha=0.3)
    ax_stress.legend(loc="upper left")
    ax_stress.margins(x=0.05, y=0.10)

    top_box_lines = [
        "Energy panel",
        f"Sampled min = {energy_min_strain:+.4f}",
    ]
    if energy_fit_min is not None and float(strain_arr.min()) <= energy_fit_min <= float(strain_arr.max()):
        top_box_lines.append(f"Fit min = {energy_fit_min:+.4f}")

    bottom_box_lines = ["Stress panel"]
    if zero_cross is not None:
        bottom_box_lines.append(f"Zero cross = {zero_cross:+.4f}")
    if np.isfinite(slope):
        bottom_box_lines.append(f"Slope = {slope:.2f} GPa")

    ax_energy.text(
        0.04,
        0.12,
        "\n".join(top_box_lines),
        transform=ax_energy.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.75", alpha=0.95),
    )
    ax_stress.text(
        0.63,
        0.08,
        "\n".join(bottom_box_lines),
        transform=ax_stress.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.75", alpha=0.95),
    )

    fig.savefig(str(outdir / "bulk_validation.png"), dpi=300)
    plt.close(fig)

    eos_a0 = float(eos_metrics["a0_A"]) if eos_metrics is not None else float(a0_fit)
    eos_v0 = float(eos_metrics["v0_atom_A3"]) if eos_metrics is not None else float("nan")
    eos_b0_eva3 = float(eos_metrics["bulk_modulus_eV_per_A3"]) if eos_metrics is not None else float("nan")
    eos_b0_gpa = float(eos_metrics["bulk_modulus_GPa"]) if eos_metrics is not None else float("nan")
    eos_b0_prime = float(eos_metrics["bulk_modulus_prime"]) if eos_metrics is not None else float("nan")
    eos_e0 = float(eos_metrics["e0_eV_per_atom"]) if eos_metrics is not None else float("nan")

    summary_lines = [
        f"kedf={kedf_name}",
        f"pp={pp_path}",
        f"spacing_A={spacing:.12f}",
        f"a0_input_A={float(args.a0):.12f}",
        f"a0_ref_A={eos_a0:.12f}",
        f"eos_e0_eV_per_atom={eos_e0:.12f}",
        f"eos_v0_atom_A3={eos_v0:.12f}",
        f"eos_a0_A={eos_a0:.12f}",
        f"eos_bulk_modulus_eV_per_A3={eos_b0_eva3:.12f}",
        f"eos_bulk_modulus_GPa={eos_b0_gpa:.12f}",
        f"eos_bulk_modulus_prime={eos_b0_prime:.12f}",
        f"repeat={repeat}",
        f"axis={int(args.axis)}",
        f"n_atoms={len(atoms_ref)}",
        f"fit_max_strain={float(args.fit_max_strain):.6f}",
        f"stress_slope_GPa={slope:.12f}",
        f"stress_intercept_GPa={intercept:.12f}",
    ]
    (outdir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[bulk] Wrote CSV  : {csv_path}")
    print(f"[bulk] Wrote plot : {outdir / 'bulk_validation.png'}")
    print(f"[bulk] Wrote note : {outdir / 'summary.txt'}")
    if eos_metrics is not None:
        print(
            f"[bulk] EOS fit: a0 = {eos_a0:.6f} A, "
            f"V0(atom) = {eos_v0:.6f} A^3, B0 = {eos_b0_gpa:.6f} GPa, B0' = {eos_b0_prime:.6f}"
        )
    if np.isfinite(slope):
        print(f"[bulk] Small-strain slope d(sigma_axis)/de = {slope:.6f} GPa")


if __name__ == "__main__":
    main()
