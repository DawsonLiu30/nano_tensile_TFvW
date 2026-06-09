from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import write
from scipy.optimize import curve_fit
from dftpy.mpi import utils as dftpy_mpi_utils


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dft_engine import evaluate_atoms_with_energy_components


EV_PER_A3_TO_GPA = 160.21766208


def birch_murnaghan(
    volume_per_atom: np.ndarray,
    e0: float,
    v0: float,
    b0_eva3: float,
    b0_prime: float,
) -> np.ndarray:
    eta = (v0 / np.asarray(volume_per_atom, dtype=float)) ** (2.0 / 3.0)
    return e0 + 9.0 * v0 * b0_eva3 / 16.0 * (
        (eta - 1.0) ** 3 * b0_prime
        + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
    )


def fit_equilibrium(rows: list[dict[str, object]]) -> dict[str, float | str | bool]:
    valid = sorted(
        [row for row in rows if row["status"] == "OK"],
        key=lambda row: float(row["a0_A"]),
    )
    if len(valid) < 3:
        raise RuntimeError(f"Only {len(valid)} valid a0 points; at least 3 are required.")

    a0 = np.asarray([float(row["a0_A"]) for row in valid], dtype=float)
    energy = np.asarray([float(row["total_energy_eV_per_atom"]) for row in valid], dtype=float)
    sampled_index = int(np.argmin(energy))
    sampled_a0 = float(a0[sampled_index])
    boundary_minimum = sampled_index in {0, len(a0) - 1}

    stress = np.asarray([float(row["hydrostatic_stress_GPa"]) for row in valid])
    brackets: list[tuple[float, int]] = []
    for index in range(len(valid) - 1):
        s1 = stress[index]
        s2 = stress[index + 1]
        if not np.isfinite(s1) or not np.isfinite(s2):
            continue
        if s1 == 0.0 or s2 == 0.0 or s1 * s2 < 0.0:
            midpoint_distance = abs((a0[index] + a0[index + 1]) / 2.0 - sampled_a0)
            brackets.append((midpoint_distance, index))

    if brackets:
        _, bracket_index = min(brackets)
        a1, a2 = a0[bracket_index : bracket_index + 2]
        s1, s2 = stress[bracket_index : bracket_index + 2]
        if s1 == 0.0:
            equilibrium_a0 = float(a1)
        elif s2 == 0.0:
            equilibrium_a0 = float(a2)
        else:
            equilibrium_a0 = float(a1 - s1 * (a2 - a1) / (s2 - s1))

        q_left = max(0, sampled_index - 1)
        q_right = min(len(a0), sampled_index + 2)
        q_a0 = a0[q_left:q_right]
        q_energy = energy[q_left:q_right]
        if q_a0.size == 3:
            coeffs = np.polyfit(q_a0, q_energy, deg=2)
            fitted = np.polyval(coeffs, q_a0)
            fit_e0 = float(np.polyval(coeffs, equilibrium_a0))
            rmse = float(np.sqrt(np.mean((fitted - q_energy) ** 2)))
        else:
            fit_e0 = float(energy[sampled_index])
            rmse = math.nan
        return {
            "fit_method": "zero-stress interpolation",
            "equilibrium_a0_A": equilibrium_a0,
            "fit_e0_eV_per_atom": fit_e0,
            "bulk_modulus_GPa": math.nan,
            "bulk_modulus_prime": math.nan,
            "fit_rmse_eV_per_atom": rmse,
            "sampled_minimum_a0_A": sampled_a0,
            "boundary_minimum": boundary_minimum,
            "fit_reliable": not boundary_minimum,
            "stress_bracket_low_a0_A": float(a1),
            "stress_bracket_high_a0_A": float(a2),
            "stress_bracket_low_GPa": float(s1),
            "stress_bracket_high_GPa": float(s2),
        }

    left = max(0, sampled_index - 4)
    right = min(len(a0), sampled_index + 5)
    fit_a0 = a0[left:right]
    fit_energy = energy[left:right]

    if fit_a0.size >= 4:
        volume = fit_a0**3 / 4.0
        e0_guess = float(fit_energy.min())
        v0_guess = float(volume[int(np.argmin(fit_energy))])
        lower = (-np.inf, float(volume.min()) * 0.95, 1.0e-4, 1.0)
        upper = (np.inf, float(volume.max()) * 1.05, 5.0, 12.0)
        try:
            params, _ = curve_fit(
                birch_murnaghan,
                volume,
                fit_energy,
                p0=(e0_guess, v0_guess, 0.5, 4.0),
                bounds=(lower, upper),
                maxfev=20000,
            )
            e0, v0, b0_eva3, b0_prime = [float(value) for value in params]
            equilibrium_a0 = float((4.0 * v0) ** (1.0 / 3.0))
            if float(a0.min()) <= equilibrium_a0 <= float(a0.max()):
                fitted = birch_murnaghan(volume, *params)
                rmse = float(np.sqrt(np.mean((fitted - fit_energy) ** 2)))
                return {
                    "fit_method": "Birch-Murnaghan",
                    "equilibrium_a0_A": equilibrium_a0,
                    "fit_e0_eV_per_atom": e0,
                    "bulk_modulus_GPa": b0_eva3 * EV_PER_A3_TO_GPA,
                    "bulk_modulus_prime": b0_prime,
                    "fit_rmse_eV_per_atom": rmse,
                    "sampled_minimum_a0_A": sampled_a0,
                    "boundary_minimum": boundary_minimum,
                    "fit_reliable": not boundary_minimum,
                }
        except Exception:
            pass

    q_left = max(0, sampled_index - 2)
    q_right = min(len(a0), sampled_index + 3)
    q_a0 = a0[q_left:q_right]
    q_energy = energy[q_left:q_right]
    coeffs = np.polyfit(q_a0, q_energy, deg=2)
    if coeffs[0] > 0.0:
        equilibrium_a0 = float(-coeffs[1] / (2.0 * coeffs[0]))
    else:
        equilibrium_a0 = sampled_a0
    if not (float(a0.min()) <= equilibrium_a0 <= float(a0.max())):
        equilibrium_a0 = sampled_a0
    fitted = np.polyval(coeffs, q_a0)
    return {
        "fit_method": "local quadratic",
        "equilibrium_a0_A": equilibrium_a0,
        "fit_e0_eV_per_atom": float(np.polyval(coeffs, equilibrium_a0)),
        "bulk_modulus_GPa": math.nan,
        "bulk_modulus_prime": math.nan,
        "fit_rmse_eV_per_atom": float(np.sqrt(np.mean((fitted - q_energy) ** 2))),
        "sampled_minimum_a0_A": sampled_a0,
        "boundary_minimum": boundary_minimum,
        "fit_reliable": not boundary_minimum and coeffs[0] > 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_point_isolated(
    *,
    case_dir: Path,
    tag: str,
    a0_A: float,
    repeat: tuple[int, int, int],
    pp_file: Path,
    spacing_A: float,
    xc: str,
    lambda_tf: float,
    mu_vw: float,
    opt_method: str,
    opt_maxiter: int,
    opt_maxfun: int,
) -> dict[str, object]:
    request_path = case_dir / f"{tag}.worker_request.json"
    result_path = case_dir / f"{tag}.worker_result.json"
    structure_path = case_dir / f"{tag}.vasp"
    request = {
        "a0_A": a0_A,
        "repeat": list(repeat),
        "pp_file": str(pp_file),
        "spacing_A": spacing_A,
        "xc": xc,
        "lambda_tf": lambda_tf,
        "mu_vw": mu_vw,
        "opt_method": opt_method,
        "opt_maxiter": opt_maxiter,
        "opt_maxfun": opt_maxfun,
        "dftpy_outfile": str(case_dir / f"{tag}.dftpy.out"),
        "scf_logfile": str(case_dir / f"{tag}.scf.log"),
        "structure_path": str(structure_path),
        "result_path": str(result_path),
    }
    request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    worker_script = ROOT / "scripts" / "evaluate_dftpy_tfvw_bulk_point.py"
    completed = subprocess.run(
        [sys.executable, str(worker_script), "--request", str(request_path)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (case_dir / f"{tag}.worker.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0 or not result_path.exists():
        raise RuntimeError(
            f"Isolated DFTpy point failed for {tag}; see {tag}.worker.log"
        )
    return json.loads(result_path.read_text(encoding="utf-8"))


def evaluate_point(
    *,
    a0_A: float,
    repeat: tuple[int, int, int],
    pp_file: Path,
    spacing_A: float,
    xc: str,
    lambda_tf: float,
    mu_vw: float,
    opt_method: str,
    opt_maxiter: int,
    opt_maxfun: int,
    outfile: Path,
    scf_logfile: Path,
) -> tuple[object, dict[str, object]]:
    atoms = bulk("Al", "fcc", a=float(a0_A), cubic=True).repeat(repeat)
    capture = io.StringIO()
    original_dftpy_stdout = dftpy_mpi_utils.environ.get("STDOUT")
    dftpy_mpi_utils.environ["STDOUT"] = capture
    try:
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            atoms, total_energy_eV, stress_gpa, terms = evaluate_atoms_with_energy_components(
                atoms,
                pp_file=pp_file,
                spacing=spacing_A,
                kedf="TFVW",
                xc=xc,
                kedf_x=lambda_tf,
                kedf_y=mu_vw,
                opt_method=opt_method,
                opt_maxiter=opt_maxiter,
                opt_maxfun=opt_maxfun,
                dftpy_outfile=str(outfile),
            )
    finally:
        dftpy_mpi_utils.environ["STDOUT"] = original_dftpy_stdout
    scf_text = capture.getvalue()
    scf_logfile.write_text(scf_text, encoding="utf-8")
    scf_converged = (
        "Optimization Converged" in scf_text
        and "NOT Converged" not in scf_text
        and "Not converged" not in scf_text
    )
    n_atoms = len(atoms)
    row = {
        "a0_A": float(a0_A),
        "n_atoms": n_atoms,
        "volume_A3": float(atoms.get_volume()),
        "total_energy_eV": total_energy_eV,
        "total_energy_eV_per_atom": total_energy_eV / n_atoms,
        "kinetic_energy_eV": float(terms["KEDF"]),
        "kinetic_energy_eV_per_atom": float(terms["KEDF"]) / n_atoms,
        "tf_energy_eV": float(terms.get("KEDF-TF", math.nan)),
        "tf_energy_eV_per_atom": float(terms.get("KEDF-TF", math.nan)) / n_atoms,
        "vw_energy_eV": float(terms.get("KEDF-VW", math.nan)),
        "vw_energy_eV_per_atom": float(terms.get("KEDF-VW", math.nan)) / n_atoms,
        "hydrostatic_stress_GPa": float(np.trace(stress_gpa) / 3.0),
        "scf_converged": scf_converged,
        "status": "OK" if scf_converged else "SCF_NOT_CONVERGED",
        "error": "" if scf_converged else "DFTpy density optimization did not converge.",
    }
    return atoms, row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one DFTpy TF+vW lambda-mu bulk EOS case.")
    parser.add_argument("--rootdir", required=True)
    parser.add_argument("--setting", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rootdir = Path(args.rootdir).expanduser().resolve()
    case_dir = rootdir / "lambda_mu_scan" / args.setting
    manifest_path = case_dir / "point_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lambda_tf = float(manifest["lambda_tf"])
    mu_vw = float(manifest["mu_vw"])
    repeat = tuple(int(value) for value in manifest["repeat"])
    pp_file = Path(manifest["pp_file"]).expanduser().resolve()
    spacing_A = float(manifest["spacing_A"])
    xc = str(manifest["xc"])
    opt_method = str(manifest.get("opt_method", "CG-HS"))
    opt_maxiter = int(manifest.get("opt_maxiter", 500))
    opt_maxfun = int(manifest.get("opt_maxfun", 500))

    rows: list[dict[str, object]] = []
    for a0_A in manifest["a0_scan_A"]:
        a0_A = float(a0_A)
        tag = f"a0_{a0_A:.4f}".replace(".", "p")
        try:
            atoms, row = evaluate_point(
                a0_A=a0_A,
                repeat=repeat,
                pp_file=pp_file,
                spacing_A=spacing_A,
                xc=xc,
                lambda_tf=lambda_tf,
                mu_vw=mu_vw,
                opt_method=opt_method,
                opt_maxiter=opt_maxiter,
                opt_maxfun=opt_maxfun,
                outfile=case_dir / f"{tag}.dftpy.out",
                scf_logfile=case_dir / f"{tag}.scf.log",
            )
            write(case_dir / f"{tag}.vasp", atoms, direct=True, vasp5=True)
            if row["status"] != "OK":
                row = evaluate_point_isolated(
                    case_dir=case_dir,
                    tag=tag,
                    a0_A=a0_A,
                    repeat=repeat,
                    pp_file=pp_file,
                    spacing_A=spacing_A,
                    xc=xc,
                    lambda_tf=lambda_tf,
                    mu_vw=mu_vw,
                    opt_method=opt_method,
                    opt_maxiter=opt_maxiter,
                    opt_maxfun=opt_maxfun,
                )
        except Exception as exc:
            row = {
                "a0_A": a0_A,
                "n_atoms": math.nan,
                "volume_A3": math.nan,
                "total_energy_eV": math.nan,
                "total_energy_eV_per_atom": math.nan,
                "kinetic_energy_eV": math.nan,
                "kinetic_energy_eV_per_atom": math.nan,
                "tf_energy_eV": math.nan,
                "tf_energy_eV_per_atom": math.nan,
                "vw_energy_eV": math.nan,
                "vw_energy_eV_per_atom": math.nan,
                "hydrostatic_stress_GPa": math.nan,
                "scf_converged": False,
                "status": "FAILED",
                "error": repr(exc),
            }
        rows.append(row)
        print(
            f"[{args.setting}] a0={a0_A:.4f} status={row['status']} "
            f"E/atom={row['total_energy_eV_per_atom']}"
        )

    write_csv(case_dir / "a0_scan.csv", rows)
    fit = fit_equilibrium(rows)
    equilibrium_a0 = float(fit["equilibrium_a0_A"])
    equilibrium = evaluate_point_isolated(
        case_dir=case_dir,
        tag="equilibrium",
        a0_A=equilibrium_a0,
        repeat=repeat,
        pp_file=pp_file,
        spacing_A=spacing_A,
        xc=xc,
        lambda_tf=lambda_tf,
        mu_vw=mu_vw,
        opt_method=opt_method,
        opt_maxiter=opt_maxiter,
        opt_maxfun=opt_maxfun,
    )
    if equilibrium["status"] != "OK":
        raise RuntimeError(
            f"Equilibrium SCF did not converge for {args.setting} at a0={equilibrium_a0:.8f} A"
        )
    refinement_steps = 0
    if fit["fit_method"] == "zero-stress interpolation":
        low_a0 = float(fit["stress_bracket_low_a0_A"])
        high_a0 = float(fit["stress_bracket_high_a0_A"])
        low_stress = float(fit["stress_bracket_low_GPa"])
        high_stress = float(fit["stress_bracket_high_GPa"])
        for step in range(1, 6):
            current_stress = float(equilibrium["hydrostatic_stress_GPa"])
            if abs(current_stress) <= 0.1:
                break
            if low_stress * current_stress <= 0.0:
                high_a0 = equilibrium_a0
                high_stress = current_stress
            else:
                low_a0 = equilibrium_a0
                low_stress = current_stress
            denominator = high_stress - low_stress
            if abs(denominator) < 1.0e-12:
                break
            next_a0 = low_a0 - low_stress * (high_a0 - low_a0) / denominator
            if abs(next_a0 - equilibrium_a0) < 1.0e-7:
                break
            equilibrium_a0 = float(next_a0)
            equilibrium = evaluate_point_isolated(
                case_dir=case_dir,
                tag="equilibrium",
                a0_A=equilibrium_a0,
                repeat=repeat,
                pp_file=pp_file,
                spacing_A=spacing_A,
                xc=xc,
                lambda_tf=lambda_tf,
                mu_vw=mu_vw,
                opt_method=opt_method,
                opt_maxiter=opt_maxiter,
                opt_maxfun=opt_maxfun,
            )
            if equilibrium["status"] != "OK":
                raise RuntimeError(
                    f"Refined equilibrium SCF failed for {args.setting} "
                    f"at a0={equilibrium_a0:.8f} A"
                )
            refinement_steps = step
        fit["equilibrium_a0_A"] = equilibrium_a0
        fit["fit_e0_eV_per_atom"] = float(equilibrium["total_energy_eV_per_atom"])
    result = {
        "setting": args.setting,
        "done": True,
        "lambda_tf": lambda_tf,
        "mu_vw": mu_vw,
        "dftpy_kedf_x": lambda_tf,
        "dftpy_kedf_y": mu_vw,
        "equation": "T_s = lambda_TF * T_TF + mu_vW * T_vW",
        "xc": xc,
        "opt_method": opt_method,
        "opt_maxiter": opt_maxiter,
        "opt_maxfun": opt_maxfun,
        "equilibrium_refinement_steps": refinement_steps,
        "spacing_A": spacing_A,
        "repeat": list(repeat),
        **fit,
        **{f"equilibrium_{key}": value for key, value in equilibrium.items()},
    }
    (case_dir / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print("============================================================")
    print(f"Completed: {args.setting}")
    print(f"lambda_TF = {lambda_tf:.6f}")
    print(f"mu_vW     = {mu_vw:.6f}")
    print(f"a0(eq)    = {equilibrium_a0:.8f} A")
    print(f"E_total/N = {equilibrium['total_energy_eV_per_atom']:.12f} eV/atom")
    print(f"T_s/N     = {equilibrium['kinetic_energy_eV_per_atom']:.12f} eV/atom")
    print("============================================================")


if __name__ == "__main__":
    main()
