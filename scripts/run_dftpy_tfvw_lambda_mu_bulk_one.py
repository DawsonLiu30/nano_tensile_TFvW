from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import write
from dftpy.mpi import utils as dftpy_mpi_utils


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dft_engine import evaluate_atoms_with_energy_components, relax_atoms_and_cell


BFGS_RE = re.compile(
    r"^\s*BFGS:\s+(?P<step>\d+)\s+\S+\s+(?P<energy>[-+0-9.eE]+)\s+"
    r"(?P<fmax>[-+0-9.eE]+)\s*$"
)


def capture_dftpy(path: Path):
    capture = io.StringIO()
    original = dftpy_mpi_utils.environ.get("STDOUT")

    @contextlib.contextmanager
    def manager():
        dftpy_mpi_utils.environ["STDOUT"] = capture
        try:
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                yield
        finally:
            dftpy_mpi_utils.environ["STDOUT"] = original
            path.write_text(capture.getvalue(), encoding="utf-8")

    return manager()


def final_bfgs_row(path: Path) -> tuple[int, float, float]:
    last = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = BFGS_RE.match(line)
        if match:
            last = (
                int(match.group("step")),
                float(match.group("energy")),
                float(match.group("fmax")),
            )
    if last is None:
        raise RuntimeError(f"No BFGS row found in {path}")
    return last


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one DFTpy lambda-mu vc-relax case.")
    parser.add_argument("--rootdir", required=True)
    parser.add_argument("--setting", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rootdir = Path(args.rootdir).expanduser().resolve()
    case_dir = rootdir / "lambda_mu_scan" / args.setting
    manifest = json.loads((case_dir / "point_manifest.json").read_text(encoding="utf-8"))

    lambda_tf = float(manifest["lambda_tf"])
    mu_vw = float(manifest["mu_vw"])
    initial_a0 = float(manifest["initial_a0_A"])
    spacing = float(manifest["spacing_A"])
    fmax = float(manifest["fmax_eV_A"])
    steps = int(manifest["relax_steps"])
    pp_file = Path(manifest["pp_file"]).resolve()
    opt_method = str(manifest["opt_method"])
    opt_maxiter = int(manifest["opt_maxiter"])
    opt_maxfun = int(manifest["opt_maxfun"])

    atoms = bulk("Al", "fcc", a=initial_a0, cubic=True)
    write(case_dir / "initial.vasp", atoms, direct=True, vasp5=True)

    relax_log = case_dir / "cell_relax.log"
    with capture_dftpy(case_dir / "cell_relax_dftpy.log"):
        relaxed, _, _ = relax_atoms_and_cell(
            atoms,
            pp_file=pp_file,
            spacing=spacing,
            kedf="TFVW",
            xc=str(manifest["xc"]),
            kedf_x=lambda_tf,
            kedf_y=mu_vw,
            fmax=fmax,
            steps=steps,
            logfile=str(relax_log),
            trajfile=str(case_dir / "cell_relax.traj"),
            dftpy_outfile=str(case_dir / "cell_relax.out"),
            scalar_pressure_gpa=0.0,
            hydrostatic_strain=True,
            opt_method=opt_method,
            opt_maxiter=opt_maxiter,
            opt_maxfun=opt_maxfun,
        )

    final_step, _, final_filter_fmax = final_bfgs_row(relax_log)
    write(case_dir / "relaxed.vasp", relaxed, direct=True, vasp5=True)

    with capture_dftpy(case_dir / "final_scf.log"):
        relaxed, total_energy, stress_gpa, terms = evaluate_atoms_with_energy_components(
            relaxed,
            pp_file=pp_file,
            spacing=spacing,
            kedf="TFVW",
            xc=str(manifest["xc"]),
            kedf_x=lambda_tf,
            kedf_y=mu_vw,
            opt_method=opt_method,
            opt_maxiter=opt_maxiter,
            opt_maxfun=opt_maxfun,
            dftpy_outfile=str(case_dir / "final_scf.out"),
        )

    final_scf_text = (case_dir / "final_scf.log").read_text(encoding="utf-8")
    scf_converged = (
        "Optimization Converged" in final_scf_text
        and "NOT Converged" not in final_scf_text
    )
    n_atoms = len(relaxed)
    volume = float(relaxed.get_volume())
    lattice_constant = volume ** (1.0 / 3.0)
    atomic_forces = relaxed.get_forces()
    atomic_fmax = float(np.linalg.norm(atomic_forces, axis=1).max())
    max_abs_stress = float(np.abs(stress_gpa).max())
    relaxation_converged = bool(final_filter_fmax <= fmax and scf_converged)
    stable_fcc_equilibrium = bool(
        relaxation_converged
        and float(manifest["min_solid_a0_A"])
        <= lattice_constant
        <= float(manifest["max_solid_a0_A"])
    )
    status = (
        "OK"
        if stable_fcc_equilibrium
        else "NO_STABLE_FCC_EQUILIBRIUM"
        if relaxation_converged
        else "UNCONVERGED"
    )

    result = {
        "setting": args.setting,
        "done": stable_fcc_equilibrium,
        "status": status,
        "relaxation_converged": relaxation_converged,
        "stable_fcc_equilibrium": stable_fcc_equilibrium,
        "relaxation_mode": "full_atom_and_hydrostatic_cell_relaxation_vc_relax_equivalent",
        "lambda_tf": lambda_tf,
        "mu_vw": mu_vw,
        "equation": "T_s = lambda_TF * T_TF + mu_vW * T_vW",
        "initial_a0_A": initial_a0,
        "lattice_constant_A": lattice_constant,
        "final_volume_A3": volume,
        "final_cell_lengths_A": [float(value) for value in relaxed.cell.lengths()],
        "final_cell_angles_deg": [float(value) for value in relaxed.cell.angles()],
        "total_energy_eV": total_energy,
        "total_energy_eV_per_atom": total_energy / n_atoms,
        "kinetic_energy_eV": float(terms["KEDF"]),
        "kinetic_energy_eV_per_atom": float(terms["KEDF"]) / n_atoms,
        "tf_energy_eV": float(terms.get("KEDF-TF", math.nan)),
        "tf_energy_eV_per_atom": float(terms.get("KEDF-TF", math.nan)) / n_atoms,
        "vw_energy_eV": float(terms.get("KEDF-VW", math.nan)),
        "vw_energy_eV_per_atom": float(terms.get("KEDF-VW", math.nan)) / n_atoms,
        "final_relax_step": final_step,
        "final_filter_fmax_eV_A": final_filter_fmax,
        "final_atomic_fmax_eV_A": atomic_fmax,
        "target_fmax_eV_A": fmax,
        "final_stress_GPa": stress_gpa.tolist(),
        "final_max_abs_stress_GPa": max_abs_stress,
        "final_scf_converged": scf_converged,
    }
    (case_dir / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(
        f"{args.setting} status={status} a0={lattice_constant:.8f} "
        f"E/N={total_energy / n_atoms:.12f} T/N={terms['KEDF'] / n_atoms:.12f} "
        f"filter_fmax={final_filter_fmax:.6f}"
    )
    if not relaxation_converged:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
