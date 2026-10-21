"""Audited nanocolumn pilot: atoms and axial zz cell relaxation, fixed vacuum.

Default dry-check never constructs a calculator. Execute only with --execute;
each case has its own lock and each attempt must be new, outside Git. Scientific
settings come from dftpy_manifest.json; elapsed-time limits never imply success.
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
import signal
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / "campaigns/al_luder_validation_nano_20260930"
CASE_IDS = ("R15_0V", "R15_1V", "R20_0V", "R20_1V")
THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
EV_A3_TO_GPA = 160.21766208


class BudgetExpired(RuntimeError):
    pass


class Terminated(RuntimeError):
    pass


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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_token(value):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) and ".." not in value,
            "Unsafe path component")
    return value


def contained_path(root, relative):
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    require(candidate.is_relative_to(root) and candidate != root, "Path escapes campaign")
    return candidate


def validate_output_root(path):
    path = Path(path).resolve()
    require(not any((p / ".git").exists() for p in [path, *path.parents]),
            "Production output must be outside any Git checkout")
    require(path != REPO and not path.is_relative_to(REPO), "Output is inside source repository")
    return path


def positive(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(value) and value > 0, "Invalid " + label)


def validate_manifest(manifest):
    require(manifest["schema_version"] == 1, "Unsupported manifest schema")
    require(manifest["xc"] in ("LDA", "PBE"), "Unsupported XC")
    grid = manifest["grid"]
    require(len(grid) == 3 and all(type(n) is int and n > 1 for n in grid), "Invalid grid")
    positive(manifest["lambda_tf"], "lambda_tf")
    positive(manifest["mu_vw"], "mu_vw")
    electronic, ionic, cell = (manifest[key] for key in ("electronic", "ionic", "cell"))
    require(electronic["method"] == "LBFGS" and electronic["densityini"] == "heg" and
            electronic["reuse"] is True, "Expected fresh HEG / reusable LBFGS electronic protocol")
    positive(electronic["econv_Ha_per_atom"], "electronic threshold")
    for key in ("ncheck", "maxiter", "maxls", "maxfun"):
        require(type(electronic[key]) is int and electronic[key] > 0, "Invalid electronic " + key)
    require(ionic["optimizer"] == "LBFGS", "Unsupported ionic optimizer")
    for key in ("fmax_eV_A", "filter_fmax_eV_A", "maxstep_A"):
        positive(ionic[key], key)
    require(type(ionic["max_steps"]) is int and ionic["max_steps"] > 0, "Invalid max_steps")
    require(cell["filter"] == "FrechetCellFilter" and cell["scalar_pressure_GPa"] == 0 and
            cell["hydrostatic_strain"] is False and cell["constant_volume"] is False and
            cell["exp_cell_factor"] == "n_atoms", "Expected zero-pressure protocol")
    positive(cell["stress_max_abs_GPa"], "stress threshold")
    positive(manifest["runtime"]["max_walltime_s"], "runtime budget")


def preflight(campaign, case_id):
    import numpy as np
    from ase.io import read
    campaign=Path(campaign).resolve()
    manifest_path=campaign / "dftpy_cases.json"
    cases=json.loads(manifest_path.read_text())
    require(case_id in cases, "Unknown nano case")
    case=cases[case_id]["geometry"]; manifest=cases[case_id]["protocol"]
    validate_manifest(manifest)
    require(manifest["cell"]["mask"]==[0,0,1,0,0,0], "Nano must hold transverse cell fixed")
    structure=contained_path(campaign,case["structure_file"])
    pp=contained_path(campaign,manifest["pseudo"]["file"])
    require(sha256(structure)==case["structure_sha256"], "Structure hash mismatch")
    require(sha256(pp)==manifest["pseudo"]["sha256"], "PP hash mismatch")
    atoms=read(structure,format="extxyz")
    require(len(atoms)==case["n_atoms"] and set(atoms.get_chemical_symbols())=={"Al"}, "Invalid atoms")
    require(np.all(atoms.pbc) and not atoms.constraints, "Expected fully PBC calculation with vacuum")
    require(atoms.arrays["parent_site_id"].tolist()==case["remaining_parent_ids"], "Parent IDs changed")
    require(np.allclose(atoms.cell.array,case["cell_A"],atol=1e-8,rtol=0) and np.allclose(atoms.positions,case["positions_A"],atol=1e-7,rtol=0), "Geometry mismatch")
    require(case["vacancy_parent_CN"]==12 and case["vacancy_count"] in [0,1], "Wrong core vacancy")
    require(min(case["minimum_axis_projected_image_gaps_A"])>=10, "Insufficient declared initial vacuum")
    snapshots={"dftpy_cases.json":manifest_path,"structure.extxyz":structure,pp.name:pp,"runner.py":Path(__file__),"dft_engine.py":REPO/"app/dft_engine.py"}
    checked={"status":"dry_check_passed_no_calculation","case":case_id,"n_atoms":len(atoms),"vacancy_count":case["vacancy_count"],"expected_electrons":3*len(atoms),"protocol":manifest,"geometry_case":case,"input_sha256":{n:sha256(p) for n,p in snapshots.items()},"initial_grid_spacing_A":(atoms.cell.lengths()/manifest["grid"]).tolist(),"grid_points":int(np.prod(manifest["grid"])),"scientific_scope":"Single core vacancy in axial periodic nanocolumn; pilot, not vacuum/grid/axial-period converged; zz strain only"}
    return checked,atoms,snapshots


def make_config(atoms, pp, manifest):
    sys.path.insert(0, str(REPO))
    # Reuse the verified project SciPy line-search compatibility layer.
    from app.dft_engine import _build_dftpy_config
    electronic = manifest["electronic"]
    config = _build_dftpy_config(
        pp, float(atoms.cell.lengths()[0] / manifest["grid"][0]), atoms,
        kedf="TFVW", xc=manifest["xc"], kedf_x=manifest["lambda_tf"], kedf_y=manifest["mu_vw"],
        opt_method=electronic["method"], opt_maxiter=electronic["maxiter"], opt_maxfun=electronic["maxfun"])
    config["GRID"]["nr"] = manifest["grid"].copy()
    config["OPT"].update(econv=electronic["econv_Ha_per_atom"], ncheck=electronic["ncheck"],
                         maxls=electronic["maxls"])
    config["DENSITY"]["densityini"] = electronic["densityini"]
    config["MATH"].update(reuse=electronic["reuse"], multistep=1)
    return config


def validate_electronic_event(event, n_atoms, manifest):
    if event["grid_shape"] != manifest["grid"]:
        raise RuntimeError("Electronic grid changed")
    if not math.isfinite(event["density_integral_electrons"]) or abs(
            event["density_integral_electrons"] - 3 * n_atoms) > 1e-6:
        raise RuntimeError("Electronic density integral mismatch")
    if event["converged_flag"] != 0 or not math.isfinite(event["energy_Ha"]):
        raise RuntimeError("Electronic minimization failed; forces and stress rejected")


def acceptance(converged, metrics, events, manifest):
    return bool(converged and events and all(e.get("accepted") is True for e in events) and
                metrics["fmax_eV_A"] <= manifest["ionic"]["fmax_eV_A"] and
                metrics["filter_fmax_eV_A"] <= manifest["ionic"]["filter_fmax_eV_A"] and
                abs(metrics["stress_GPa"][2][2]) <= manifest["cell"]["stress_max_abs_GPa"])


@contextlib.contextmanager
def capture(path):
    from dftpy.mpi import utils as mpi_utils
    sys.stdout.flush()
    sys.stderr.flush()
    with Path(path).open("w", encoding="utf-8", buffering=1) as stream:
        saved = os.dup(1), os.dup(2)
        original = mpi_utils.environ.get("STDOUT")
        try:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            mpi_utils.environ["STDOUT"] = stream
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                yield
        finally:
            stream.flush()
            mpi_utils.environ["STDOUT"] = original
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


@contextlib.contextmanager
def budget_guard(seconds):
    def alarm(signum, frame):
        raise BudgetExpired("Wall-time budget reached; last completed checkpoint retained")

    def terminated(signum, frame):
        raise Terminated("Termination signal received; last completed checkpoint retained")

    old_alarm, old_term = signal.getsignal(signal.SIGALRM), signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGALRM, alarm)
    signal.signal(signal.SIGTERM, terminated)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        signal.signal(signal.SIGTERM, old_term)


def energy_components(calc):
    from ase.units import Hartree
    terms = {str(name): float(value.energy) * Hartree
             for name, value in calc.dftpy_results["energypotential"].items()
             if getattr(value, "energy", None) is not None}
    parts = [v for key, v in terms.items() if key.startswith("KEDF-")]
    if parts:
        terms["KEDF"] = sum(parts)
    if not all(math.isfinite(value) for value in terms.values()):
        raise RuntimeError("Nonfinite energy component")
    return terms


def save_density(path, rho):
    import numpy as np
    path = Path(path)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, density_electrons_bohr3=np.asarray(rho),
                        lattice_bohr=np.asarray(rho.grid.lattice),
                        grid=np.asarray(rho.grid.nr), origin_bohr=np.asarray(rho.grid.origin))
    temporary.replace(path)


def run(args, checked, atoms, snapshots, config):
    import fcntl
    import resource
    import numpy as np
    from ase.filters import FrechetCellFilter
    from ase.io import write
    from ase.io.trajectory import Trajectory
    from staged_optimizer import make_optimizer
    from ase.units import Bohr, Hartree
    from dftpy.api.api4ase import DFTpyCalculator
    from dftpy.optimization.optimization import Optimization

    manifest = checked["protocol"]
    root = validate_output_root(args.output_root)
    safe_token(args.attempt)
    budget = min(float(args.max_runtime_s if args.max_runtime_s is not None else
                       manifest["runtime"]["max_walltime_s"]), float(manifest["runtime"]["max_walltime_s"]))
    positive(budget, "runtime budget")
    case_dir = root / args.case
    case_dir.mkdir(parents=True, exist_ok=True)
    # Independent cases can run simultaneously; the same case cannot.
    with (case_dir / ".execution.lock").open("a") as execution_lock:
        fcntl.flock(execution_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for prior in case_dir.glob("*/result.json"):
            if json.loads(prior.read_text(encoding="utf-8")).get("accepted") is True:
                raise FileExistsError("Already converged; refusing duplicate execution: " + str(prior))
        attempt = case_dir / args.attempt
        attempt.mkdir(exist_ok=False)
        inputs = attempt / "inputs"
        inputs.mkdir()
        for name, source in snapshots.items():
            shutil.copyfile(source, inputs / name)
            require(sha256(inputs / name) == checked["input_sha256"][name], "Input changed during snapshot")
        pp_name = Path(manifest["pseudo"]["file"]).name
        config["PATH"]["pppath"] = str(inputs)
        config["PP"]["Al"] = pp_name
        dump(attempt / "config.json", config)
        write(attempt / "initial.extxyz", atoms, format="extxyz")
        write(attempt / "initial.traj", atoms, format="traj")
        provenance = {**checked, "status": "execution_inputs_frozen", "started_utc": utc(),
                      "python": sys.version, "executable": sys.executable,
                      "packages": {key: version(key) for key in ("dftpy", "ase", "numpy", "scipy")},
                      "threads": {key: os.environ.get(key) for key in THREAD_KEYS},
                      "source_repo": str(REPO), "campaign": str(Path(args.campaign).resolve()),
                      "command": sys.argv, "resolved_config": config,
                      "config_sha256": sha256(attempt / "config.json"),
                      "effective_runtime_budget_s": budget,
                      "units": {"Hartree_eV": Hartree, "Bohr_A": Bohr, "eV_A3_to_GPa": EV_A3_TO_GPA}}
        dump(attempt / "provenance.json", provenance)
        calc = DFTpyCalculator(config=config)
        atoms.calc = calc
        cell_filter = FrechetCellFilter(atoms, scalar_pressure=0.0, hydrostatic_strain=False,
                                       constant_volume=False, exp_cell_factor=float(len(atoms)), mask=manifest["cell"]["mask"])
        original_optimize = Optimization.optimize_rho
        events = []
        dyn = None
        trajectory = None
        started = time.monotonic()

        def check_deadline():
            if time.monotonic() - started >= budget:
                raise BudgetExpired("Wall-time budget reached")

        def monitored(opt, *positional, **keyword):
            check_deadline()
            start = time.monotonic()
            event = {"evaluation": len(events) + 1, "utc": utc()}
            try:
                rho = original_optimize(opt, *positional, **keyword)
                event.update(converged_flag=int(opt.converged),
                             flag_meaning="0 converged; 1 failed line search; 2 max iterations",
                             effective_econv_Ha_total=float(opt.optimization_options["econv"]),
                             density_integral_electrons=float(np.sum(rho.integral())),
                             grid_shape=[int(v) for v in rho.grid.nr], energy_Ha=float(opt.functional.energy),
                             cell_A=atoms.cell.array.tolist())
                validate_electronic_event(event, len(atoms), manifest)
                if not np.allclose(np.asarray(rho.grid.lattice) * Bohr, atoms.cell.array, atol=1e-8, rtol=0):
                    raise RuntimeError("Density lattice does not follow current cell")
                event["accepted"] = True
                return rho
            except BaseException as exc:
                event.update(accepted=False, error=repr(exc))
                raise
            finally:
                event["elapsed_s"] = time.monotonic() - start
                events.append(event)
                serializable = {key: (str(v) if isinstance(v, float) and not math.isfinite(v) else v)
                                for key, v in event.items()}
                with (attempt / "electronic_evaluations.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(serializable, allow_nan=False) + "\n")

        def metrics():
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces())
            stress = np.asarray(atoms.get_stress(voigt=False))
            generalized = np.asarray(cell_filter.get_forces())
            cell = atoms.cell.array
            frozen=np.ones((3,3),dtype=bool); frozen[2,2]=False
            if not np.allclose(cell[frozen],np.asarray(checked["geometry_case"]["cell_A"])[frozen],atol=1e-8,rtol=0):
                raise RuntimeError("Transverse vacuum cell or tilt changed")
            volume = float(atoms.get_volume())
            if (not math.isfinite(energy) or volume <= 0 or not np.all(np.isfinite(cell)) or
                    not np.all(np.isfinite(forces)) or not np.all(np.isfinite(stress)) or
                    not np.all(np.isfinite(generalized)) or np.linalg.det(cell) <= 0):
                raise RuntimeError("Nonfinite forces/stress/energy or invalid cell")
            if not np.all(atoms.pbc) or atoms.arrays["parent_site_id"].tolist() != checked["geometry_case"]["remaining_parent_ids"]:
                raise RuntimeError("Atom identity or periodic boundary changed")
            return {"energy_eV": energy, "fmax_eV_A": float(np.linalg.norm(forces, axis=1).max()),
                    "filter_fmax_eV_A": float(np.linalg.norm(generalized, axis=1).max()),
                    "stress_max_abs_GPa": float(np.abs(stress).max() * EV_A3_TO_GPA),
                    "stress_eV_A3": stress.tolist(), "stress_GPa": (stress * EV_A3_TO_GPA).tolist(),
                    "cell_A": cell.tolist(), "volume_A3": volume,
                    "cell_lengths_A": atoms.cell.lengths().tolist(), "cell_angles_deg": atoms.cell.angles().tolist(),
                    "grid_spacing_along_cell_vectors_A": (np.linalg.norm(cell, axis=1) / manifest["grid"]).tolist()}

        def checkpoint():
            check_deadline()
            current = metrics()
            write(attempt / "checkpoint.tmp.traj", atoms, format="traj")
            (attempt / "checkpoint.tmp.traj").replace(attempt / "checkpoint.traj")
            save_density(attempt / "checkpoint_density.npz", calc.dftpy_results["density"])
            current.update(utc=utc(), ionic_step=int(dyn.nsteps),
                           checkpoint_sha256=sha256(attempt / "checkpoint.traj"),
                           checkpoint_density_sha256=sha256(attempt / "checkpoint_density.npz"))
            # A single atomic bundle is the authoritative completed checkpoint.
            # A signal between replacing separate convenience files cannot
            # destroy the preceding coherent geometry+density state.
            rho = calc.dftpy_results["density"]
            np.savez_compressed(attempt / "checkpoint_bundle.tmp.npz",
                                metadata_json=json.dumps(current, allow_nan=False),
                                positions_A=atoms.positions, cell_A=atoms.cell.array,
                                numbers=atoms.numbers, pbc=atoms.pbc,
                                parent_site_id=atoms.arrays["parent_site_id"],
                                forces_eV_A=atoms.get_forces(),
                                density_electrons_bohr3=np.asarray(rho),
                                lattice_bohr=np.asarray(rho.grid.lattice),
                                grid=np.asarray(rho.grid.nr), origin_bohr=np.asarray(rho.grid.origin))
            (attempt / "checkpoint_bundle.tmp.npz").replace(attempt / "checkpoint_bundle.npz")
            current["checkpoint_bundle_sha256"] = sha256(attempt / "checkpoint_bundle.npz")
            dump(attempt / "checkpoint.json", current)
            trajectory.write(atoms)
            with (attempt / "ionic_evaluations.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(current, allow_nan=False) + "\n")

        result = {"case": args.case, "attempt": args.attempt, "status": "running", "accepted": False,
                  "n_atoms": len(atoms), "vacancy_count": checked["vacancy_count"], "started_utc": utc(),
                  "initial_cell_A": atoms.cell.array.tolist(), "target_pressure_GPa": 0.0,
                  "relaxation_mode": "fixed_cell_atoms_after_axial_relaxation; original axial filter/stress acceptance retained",
                  "effective_runtime_budget_s": budget, "scientific_scope": checked["scientific_scope"]}
        dump(attempt / "result.json", result)
        Optimization.optimize_rho = monitored
        try:
            with budget_guard(budget), capture(attempt / "dftpy_full_stdout.log"):
                trajectory = Trajectory(attempt / "relax.traj", "w", atoms)
                dyn = make_optimizer(cell_filter, logfile=str(attempt / "relax.log"),
                                     ionic=manifest["ionic"])
                dyn.attach(checkpoint, interval=1)
                converged = bool(dyn.run(fmax=manifest["ionic"]["filter_fmax_eV_A"],
                                         steps=manifest["ionic"]["max_steps"]))
                final = metrics()
                terms = energy_components(calc)
                if not math.isclose(final["energy_eV"], terms["TOTAL"], abs_tol=1e-8, rel_tol=0):
                    raise RuntimeError("Total energy and component units disagree")
                accepted = acceptance(converged, final, events, manifest)
                write(attempt / "final.extxyz", atoms, format="extxyz")
                write(attempt / "final.traj", atoms, format="traj")
                write(attempt / "final.vasp", atoms, format="vasp", direct=True, vasp5=True)
                rho = calc.dftpy_results["density"]
                save_density(attempt / "final_density.npz", rho)
                np.save(attempt / "final_forces_eV_A.npy", atoms.get_forces())
                result.update(**final, status="converged" if accepted else "relaxation_unconverged",
                              accepted=accepted, ionic_converged=converged, ionic_steps=int(dyn.nsteps),
                              final_fmax_eV_A=final["fmax_eV_A"],
                              final_filter_fmax_eV_A=final["filter_fmax_eV_A"],
                              energy_components_eV=terms, kinetic_energy_eV=terms.get("KEDF"),
                              tf_energy_eV=terms.get("KEDF-TF"), vw_energy_eV=terms.get("KEDF-VW"),
                              energy_Ha=final["energy_eV"] / Hartree, energy_conversion_eV_per_Ha=Hartree,
                              all_electronic_calls_converged=bool(events) and all(e.get("accepted") is True for e in events),
                              density_integral_electrons=float(np.sum(rho.integral())),
                              stress_convention="ASE tensile-positive; axial zz stress gated; transverse stress recorded but cell frozen; stress uses vacuum-containing box volume")
        except BaseException as exc:
            status = ("time_budget_exhausted" if isinstance(exc, BudgetExpired) else
                      "interrupted" if isinstance(exc, (KeyboardInterrupt, Terminated)) else "failed")
            result.update(status=status, accepted=False, error=repr(exc), traceback=traceback.format_exc(),
                          ionic_steps=int(dyn.nsteps) if dyn is not None else None,
                          last_completed_checkpoint="checkpoint_bundle.npz" if (attempt / "checkpoint_bundle.npz").exists() else None)
        finally:
            Optimization.optimize_rho = original_optimize
            if trajectory is not None:
                trajectory.close()
            result.update(finished_utc=utc(), elapsed_s=time.monotonic() - started,
                          electronic_calls=len(events), peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                          artifact_sha256={str(p.relative_to(attempt)): sha256(p)
                                           for p in sorted(attempt.rglob("*")) if p.is_file()
                                           and p.name not in ("result.json", "result.json.tmp")})
            dump(attempt / "result.json", result)
        print(json.dumps({"case": args.case, "status": result["status"],
                          "attempt_dir": str(attempt), "elapsed_s": result["elapsed_s"]}, indent=2))
        return 0 if result["accepted"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN)
    parser.add_argument("--case", choices=CASE_IDS, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--attempt", default="attempt_001")
    parser.add_argument("--threads", type=int, choices=(1,), default=1)
    parser.add_argument("--max-runtime-s", type=float)
    parser.add_argument("--full-report", action="store_true")
    args = parser.parse_args(argv)
    if args.execute and (args.output_root is None or sys.platform != "linux"):
        parser.error("Execute requires Linux and an explicit outside-Git --output-root")
    if args.max_runtime_s is not None:
        positive(args.max_runtime_s, "runtime budget")
    for key in THREAD_KEYS:
        os.environ[key] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    checked, atoms, snapshots = preflight(args.campaign, args.case)
    pp = contained_path(args.campaign, checked["protocol"]["pseudo"]["file"])
    config = make_config(atoms, pp, checked["protocol"])
    checked["resolved_config"] = config
    checked["packages"] = {name: version(name) for name in ("dftpy", "ase", "numpy", "scipy")}
    if args.output_root:
        checked["output_root"] = str(validate_output_root(args.output_root))
    if not args.execute:
        display = checked if args.full_report else {
            key: value for key, value in checked.items() if key not in ("geometry_case", "resolved_config")}
        print(json.dumps(display, indent=2, allow_nan=False, default=str))
        return 0
    return run(args, checked, atoms, snapshots, config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, FileExistsError, FileNotFoundError, BlockingIOError) as exc:
        print("[REJECTED] " + str(exc), file=sys.stderr)
        raise SystemExit(2)
