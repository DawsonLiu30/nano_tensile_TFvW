"""Run one audited, fresh ideal-FCC, fixed-cell OF-DFT vacancy calculation.

Default --dry-check reads geometry/configuration only; --execute is required
to calculate. Each attempt is new, outside Git, and never resumes old density.
Designed for /var/tmp/al-defects-runtime-20260907/env/bin/python in WSL.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback


REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / "campaigns/al_bulk_vacancy_relax_20260924"
EXPECTED_PP_SHA256 = "c2fd4300aa39aa2466149a315e1c7cd87aac6615032db5ded8f237e9c40df803"
CASE_IDS = ("0V", "1V", "2V_1NN_D110", "2V_2NN_D100", "2V_3NN_D211",
            "2V_4NN_D110", "2V_5NN_D310", "2V_6NN_D111")
PROTOCOL = {
    "a_ref_A": 3.9545804060131293, "repeat": [3, 3, 3], "parent_sites": 108,
    "grid": [78, 78, 78], "xc": "LDA", "kedf": "TFVW",
    "lambda_tf": 0.9, "mu_vw": 0.1,
    "equation": "Ts = lambda_tf * T_TF + mu_vw * T_vW",
    "electronic": {"method": "LBFGS", "econv_Ha_per_atom": 1e-9, "ncheck": 3,
                   "maxiter": 1000, "maxls": 50, "maxfun": 100,
                   "densityini": "heg", "reuse": True},
    "ionic": {"optimizer": "LBFGS", "fmax_eV_A": 0.001, "max_steps": 1000,
              "maxstep_A": 0.15},
    "cell_protocol": "fixed identical 3x3x3 cubic cell; all atoms free; no cell filter",
    "initial_state": "fresh ideal FCC geometry; HEG density; no prior geometry/density restart",
    "scientific_scope": "finite 108-site result; not evidence of isolated-defect, grid, or KS-DFT accuracy",
}
HA_TO_EV = 27.211386245988
EV_A3_TO_GPA = 160.21766208


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                    allow_nan=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_token(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or ".." in value:
        raise ValueError(f"Unsafe path component: {value!r}")
    return value


def contained_path(root, relative):
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise ValueError(f"Path escapes campaign: {relative}")
    return candidate


def validate_output_root(path, repo=REPO):
    path, repo = Path(path).resolve(), Path(repo).resolve()
    if path == repo or path.is_relative_to(repo):
        raise ValueError("Production output must be outside the source repository")
    if any((p / ".git").exists() for p in [path, *path.parents]):
        raise ValueError("Production output cannot be inside any Git checkout")
    return path


def preflight(campaign, case_id, pp):
    """Validate immutable scientific inputs without constructing a calculator."""
    import numpy as np
    from ase.io import read

    if case_id not in CASE_IDS:
        raise ValueError(f"Unknown case: {case_id}")
    campaign, pp = Path(campaign).resolve(), Path(pp).resolve()
    geom_path = campaign / "geometry.json"
    geometry = json.loads(geom_path.read_text(encoding="utf-8-sig"))
    if geometry["case_order"] != list(CASE_IDS) or set(geometry["cases"]) != set(CASE_IDS):
        raise ValueError("Campaign case inventory differs from the eight-case protocol")
    if geometry["n_parent_sites"] != 108 or not math.isclose(
            geometry["a_ref_A"], PROTOCOL["a_ref_A"], abs_tol=1e-12, rel_tol=0):
        raise ValueError("Campaign lattice parameter or site count mismatch")
    expected_cell = np.eye(3) * (3 * PROTOCOL["a_ref_A"])
    if not np.allclose(geometry["cell_A"], expected_cell, atol=1e-10, rtol=0):
        raise ValueError("Campaign is not the specified fixed cubic cell")
    if sha256(pp) != EXPECTED_PP_SHA256:
        raise ValueError("al.lda.recpot differs from the controlled LDA pseudopotential")
    case = geometry["cases"][case_id]
    q = 0 if case_id == "0V" else 1 if case_id == "1V" else 2
    if case["vacancy_count"] != q or case["n_atoms"] != 108 - q:
        raise ValueError("Case atom/vacancy count mismatch")
    removed = case["removed_parent_ids"]
    remaining = case["remaining_parent_ids"]
    if (len(removed) != q or len(set(removed)) != q or
            remaining != [i for i in range(108) if i not in removed] or
            any(not isinstance(i, int) or not 0 <= i < 108 for i in removed)):
        raise ValueError("Invalid parent-site mapping")
    files = {key: contained_path(campaign, rel)
             for key, rel in case["structure_files"].items()}
    for key in ("extxyz", "poscar"):
        if sha256(files[key]) != case["structure_sha256"][key]:
            raise ValueError(f"Structure hash mismatch: {key}")
    atoms = read(files["extxyz"], format="extxyz")
    if (len(atoms) != 108 - q or set(atoms.get_chemical_symbols()) != {"Al"} or
            not np.all(atoms.pbc) or atoms.constraints):
        raise ValueError("Expected unconstrained all-periodic Al structure")
    if not np.allclose(atoms.cell.array, expected_cell, atol=1e-10, rtol=0):
        raise ValueError("Structure cell disagrees with fixed-cell protocol")
    if not np.allclose(atoms.positions, case["positions_A"], atol=1e-7, rtol=0):
        raise ValueError("Structure ordering/positions disagree with geometry manifest")
    if not np.all(np.isfinite(atoms.positions)):
        raise ValueError("Nonfinite input positions")
    # All cases must be literal deletions of the same translated ideal FCC.
    parent = geometry["cases"]["0V"]
    parent_positions = np.asarray(parent["positions_A"], dtype=float)
    if parent_positions.shape != (108, 3):
        raise ValueError("Parent geometry shape mismatch")
    if not np.allclose(atoms.positions, parent_positions[remaining], atol=1e-7, rtol=0):
        raise ValueError("Input is not the ideal parent with specified sites removed")
    halfgrid = 2 * parent_positions / PROTOCOL["a_ref_A"]
    if not np.allclose(halfgrid, np.round(halfgrid), atol=1e-7, rtol=0):
        raise ValueError("Parent is not aligned to the translated ideal FCC lattice")
    parity = np.round(halfgrid).astype(int).sum(axis=1) % 2
    if len(set(parity.tolist())) != 1:
        raise ValueError("Parent sites violate FCC parity")
    if len(set(map(tuple, (np.round(halfgrid).astype(int) % 6).tolist()))) != 108:
        raise ValueError("Duplicate parent sites under periodic boundaries")
    atoms.set_array("parent_site_id", np.asarray(remaining, dtype=int))
    hashes = {"geometry.json": sha256(geom_path), "structure.extxyz": sha256(files["extxyz"]),
              "POSCAR": sha256(files["poscar"]), "al.lda.recpot": sha256(pp),
              "runner.py": sha256(__file__), "dft_engine.py": sha256(REPO / "app/dft_engine.py")}
    report = {"status": "dry_check_passed_no_calculation", "case": case_id,
              "n_atoms": len(atoms), "vacancy_count": q, "expected_electrons": 3 * len(atoms),
              "protocol": PROTOCOL, "actual_grid_spacing_A": 3 * PROTOCOL["a_ref_A"] / 78,
              "input_sha256": hashes, "geometry_case": case,
              "grid_points": 78**3, "one_float64_real_field_MiB": 78**3 * 8 / 2**20,
              "memory_note": "one-field size is a lower bound, not total peak process memory"}
    return report, atoms, files, geometry


def make_config(atoms, pp):
    sys.path.insert(0, str(REPO))
    # Installs the project's existing SciPy line-search compatibility patch.
    from app.dft_engine import _build_dftpy_config
    electronic = PROTOCOL["electronic"]
    config = _build_dftpy_config(
        pp, 3 * PROTOCOL["a_ref_A"] / 78, atoms,
        kedf="TFVW", xc="LDA", kedf_x=0.9, kedf_y=0.1,
        opt_method="LBFGS", opt_maxiter=electronic["maxiter"],
        opt_maxfun=electronic["maxfun"])
    config["GRID"]["nr"] = PROTOCOL["grid"].copy()
    config["OPT"].update(econv=electronic["econv_Ha_per_atom"],
                         ncheck=electronic["ncheck"], maxls=electronic["maxls"])
    config["DENSITY"]["densityini"] = "heg"
    config["MATH"]["reuse"] = True
    config["MATH"]["multistep"] = 1
    return config


def validate_electronic_event(event, n_atoms):
    if event["grid_shape"] != PROTOCOL["grid"]:
        raise RuntimeError("Electronic evaluation grid changed")
    if not math.isfinite(event["density_integral_electrons"]) or abs(
            event["density_integral_electrons"] - 3 * n_atoms) > 1e-6:
        raise RuntimeError("Electronic density integral mismatch")
    if event["converged_flag"] != 0 or not math.isfinite(event["energy_Ha"]):
        raise RuntimeError("Electronic minimization did not converge; ionic forces rejected")


@contextlib.contextmanager
def capture(path):
    """Capture DFTpy's stdout handle, Python, and native process output."""
    from dftpy.mpi import utils as mpi_utils
    sys.stdout.flush()
    sys.stderr.flush()
    with Path(path).open("w", encoding="utf-8", buffering=1) as stream:
        saved_fd = os.dup(1), os.dup(2)
        original_stdout = mpi_utils.environ.get("STDOUT")
        try:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            mpi_utils.environ["STDOUT"] = stream
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                yield
        finally:
            stream.flush()
            mpi_utils.environ["STDOUT"] = original_stdout
            os.dup2(saved_fd[0], 1)
            os.dup2(saved_fd[1], 2)
            os.close(saved_fd[0])
            os.close(saved_fd[1])


def energy_components(calc):
    terms = {str(name): float(value.energy) * HA_TO_EV
             for name, value in calc.dftpy_results["energypotential"].items()
             if getattr(value, "energy", None) is not None}
    components = [v for k, v in terms.items() if k.startswith("KEDF-")]
    if components:
        terms["KEDF"] = sum(components)
    if not all(math.isfinite(value) for value in terms.values()):
        raise RuntimeError("Nonfinite energy component")
    return terms


def run(args, checked, atoms, files, config):
    import fcntl
    import resource
    import numpy as np
    from ase.io import write
    from ase.optimize import LBFGS
    from dftpy.api.api4ase import DFTpyCalculator
    from dftpy.optimization.optimization import Optimization

    root = validate_output_root(args.output_root)
    safe_token(args.attempt)
    root.mkdir(parents=True, exist_ok=True)
    # One worker across this output root. flock releases after interruption.
    with (root / ".execution.lock").open("a") as execution_lock:
        fcntl.flock(execution_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        case_dir = root / args.case
        for prior in case_dir.glob("*/result.json"):
            prior_result = json.loads(prior.read_text(encoding="utf-8"))
            if prior_result.get("status") == "converged":
                raise FileExistsError(f"Already converged; refusing duplicate execution: {prior}")
        attempt = case_dir / args.attempt
        attempt.mkdir(parents=True, exist_ok=False)
        inputs = attempt / "inputs"
        inputs.mkdir()
        snapshots = {"geometry.json": Path(args.campaign) / "geometry.json",
                     "structure.extxyz": files["extxyz"], "POSCAR": files["poscar"],
                     "al.lda.recpot": Path(args.pp), "runner.py": Path(__file__),
                     "dft_engine.py": REPO / "app/dft_engine.py"}
        for name, source in snapshots.items():
            shutil.copyfile(source, inputs / name)
            if sha256(inputs / name) != checked["input_sha256"][name]:
                raise ValueError(f"Input changed between validation and snapshot: {name}")
        config["PATH"]["pppath"] = str(inputs)
        config["PP"]["Al"] = "al.lda.recpot"
        write(attempt / "initial.extxyz", atoms, format="extxyz")
        write(attempt / "initial.traj", atoms, format="traj")
        dump(attempt / "config.json", config)
        provenance = {**checked, "status": "execution_inputs_frozen", "started_utc": utc(),
                      "python": sys.version, "executable": sys.executable,
                      "packages": {name: version(name) for name in ("dftpy", "ase", "numpy", "scipy")},
                      "threads": {key: os.environ[key] for key in THREAD_KEYS},
                      "source_repo": str(REPO), "campaign": str(Path(args.campaign).resolve()),
                      "command": sys.argv, "config_sha256": sha256(attempt / "config.json")}
        dump(attempt / "provenance.json", provenance)
        calc = DFTpyCalculator(config=config)
        atoms.calc = calc
        reference_cell = atoms.cell.array.copy()
        original_optimize = Optimization.optimize_rho
        events = []
        dyn = None

        def monitored(opt, *positional, **keyword):
            start = time.monotonic()
            event = {"evaluation": len(events) + 1, "utc": utc()}
            try:
                rho = original_optimize(opt, *positional, **keyword)
                event.update(converged_flag=int(opt.converged),
                             flag_meaning="0 converged; 1 failed line search; 2 max iterations",
                             effective_econv_Ha_total=float(opt.optimization_options["econv"]),
                             density_integral_electrons=float(np.sum(rho.integral())),
                             grid_shape=[int(v) for v in rho.grid.nr],
                             energy_Ha=float(opt.functional.energy))
                validate_electronic_event(event, len(atoms))
                event["accepted"] = True
                return rho
            except BaseException as exc:
                event.update(accepted=False, error=repr(exc))
                raise
            finally:
                event["elapsed_s"] = time.monotonic() - start
                events.append(event)
                # Explicitly keep pathological numeric values as diagnostic strings.
                serializable = {k: (str(v) if isinstance(v, float) and not math.isfinite(v) else v)
                                for k, v in event.items()}
                with (attempt / "electronic_evaluations.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(serializable, allow_nan=False) + "\n")

        started = time.monotonic()
        result = {"case": args.case, "attempt": args.attempt, "status": "running",
                  "accepted": False, "n_atoms": len(atoms), "started_utc": utc(),
                  "scientific_scope": PROTOCOL["scientific_scope"]}
        dump(attempt / "result.json", result)
        Optimization.optimize_rho = monitored
        try:
            with capture(attempt / "dftpy_full_stdout.log"):
                ionic = PROTOCOL["ionic"]
                dyn = LBFGS(atoms, trajectory=str(attempt / "relax.traj"),
                            logfile=str(attempt / "relax.log"), maxstep=ionic["maxstep_A"])

                def checkpoint():
                    energy = float(atoms.get_potential_energy())
                    forces = np.asarray(atoms.get_forces())
                    if not np.all(np.isfinite(forces)) or not math.isfinite(energy):
                        raise RuntimeError("Nonfinite ionic energy/forces")
                    if not np.allclose(atoms.cell.array, reference_cell, atol=1e-12, rtol=0):
                        raise RuntimeError("Cell changed during fixed-cell relaxation")
                    write(attempt / "checkpoint.tmp.traj", atoms, format="traj")
                    (attempt / "checkpoint.tmp.traj").replace(attempt / "checkpoint.traj")
                    event = {"utc": utc(), "ionic_step": int(dyn.nsteps), "energy_eV": energy,
                             "fmax_eV_A": float(np.linalg.norm(forces, axis=1).max()),
                             "checkpoint_sha256": sha256(attempt / "checkpoint.traj")}
                    dump(attempt / "checkpoint.json", event)
                    with (attempt / "ionic_evaluations.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(event, allow_nan=False) + "\n")

                dyn.attach(checkpoint, interval=1)
                converged = bool(dyn.run(fmax=ionic["fmax_eV_A"], steps=ionic["max_steps"]))
                energy = float(atoms.get_potential_energy())
                forces = np.asarray(atoms.get_forces())
                stress = np.asarray(atoms.get_stress(voigt=False))
                terms = energy_components(calc)
                if not np.all(np.isfinite(stress)):
                    raise RuntimeError("Nonfinite final stress")
                final_fmax = float(np.linalg.norm(forces, axis=1).max())
                all_electronic_ok = bool(events) and all(e.get("accepted", False) for e in events)
                accepted = converged and final_fmax <= ionic["fmax_eV_A"] and all_electronic_ok
                write(attempt / "final.extxyz", atoms, format="extxyz")
                write(attempt / "final.traj", atoms, format="traj")
                write(attempt / "final.vasp", atoms, format="vasp", direct=True, vasp5=True)
                rho = calc.dftpy_results["density"]
                np.savez_compressed(attempt / "final_density.npz", density_electrons_bohr3=np.asarray(rho),
                                    lattice_bohr=np.asarray(rho.grid.lattice),
                                    grid=np.asarray(rho.grid.nr), origin_bohr=np.asarray(rho.grid.origin))
                np.save(attempt / "final_forces_eV_A.npy", forces)
                result.update(status="converged" if accepted else "ionic_unconverged", accepted=accepted,
                              energy_eV=energy, energy_components_eV=terms,
                              kinetic_energy_eV=terms.get("KEDF"),
                              tf_energy_eV=terms.get("KEDF-TF"), vw_energy_eV=terms.get("KEDF-VW"),
                              ionic_converged=converged, final_fmax_eV_A=final_fmax,
                              ionic_steps=int(dyn.nsteps), all_electronic_calls_converged=all_electronic_ok,
                              stress_eV_A3=stress.tolist(), stress_GPa=(stress * EV_A3_TO_GPA).tolist(),
                              stress_convention="ASE tensile-positive; verify with energy-strain derivative before dipole analysis",
                              cell_A=atoms.cell.array.tolist(), volume_A3=float(atoms.get_volume()),
                              density_integral_electrons=float(np.sum(rho.integral())))
        except BaseException as exc:
            result.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                          accepted=False, error=repr(exc), traceback=traceback.format_exc())
        finally:
            Optimization.optimize_rho = original_optimize
            result.update(finished_utc=utc(), elapsed_s=time.monotonic() - started,
                          electronic_calls=len(events), peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                          artifact_sha256={str(p.relative_to(attempt)): sha256(p)
                                           for p in sorted(attempt.rglob("*")) if p.is_file()
                                           and p.name not in ("result.json", "result.json.tmp")})
            dump(attempt / "result.json", result)
        print(json.dumps({"case": args.case, "status": result["status"],
                          "attempt_dir": str(attempt), "elapsed_s": result["elapsed_s"]}, indent=2))
        return 0 if result["accepted"] else 1


THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN)
    parser.add_argument("--case", required=True, choices=CASE_IDS)
    parser.add_argument("--pp", type=Path, default=REPO / "al.lda.recpot")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--attempt", default="attempt_001")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--full-report", action="store_true",
                        help="Include full atom mapping and resolved DFTpy config in dry-check stdout")
    args = parser.parse_args(argv)
    if args.execute and args.output_root is None:
        parser.error("--execute requires an explicit outside-Git --output-root")
    if args.execute and sys.platform != "linux":
        parser.error("Execute with the verified WSL Linux Python runtime")
    for key in THREAD_KEYS:
        os.environ[key] = str(args.threads)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    checked, atoms, files, _ = preflight(args.campaign, args.case, args.pp)
    config = make_config(atoms, args.pp)
    checked["resolved_config"] = config
    checked["packages"] = {name: version(name) for name in ("dftpy", "ase", "numpy", "scipy")}
    if args.output_root:
        checked["output_root"] = str(validate_output_root(args.output_root))
    if not args.execute:
        display = checked if args.full_report else {
            key: value for key, value in checked.items() if key not in ("geometry_case", "resolved_config")}
        print(json.dumps(display, indent=2, allow_nan=False, default=str))
        return 0
    return run(args, checked, atoms, files, config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileExistsError, FileNotFoundError, BlockingIOError) as error:
        print(f"[REJECTED] {error}", file=sys.stderr)
        raise SystemExit(2)
