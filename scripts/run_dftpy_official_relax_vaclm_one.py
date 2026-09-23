from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import BFGS, BFGSLineSearch, LBFGS, MDMin
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG

try:
    from ase.filters import UnitCellFilter
except Exception:  # pragma: no cover - compatibility fallback
    from ase.constraints import UnitCellFilter

from dftpy.api.api4ase import DFTpyCalculator
from dftpy.config import DefaultOption, OptionFormat


HARTREE_TO_EV = 27.211386245988


def patch_dftpy_linesearch_compat() -> None:
    """Keep the official ASE workflow, but patch DFTpy/SciPy API drift if needed."""
    try:
        from scipy.optimize._dcsrch import DCSRCH
        import dftpy.math_utils as dftpy_math_utils
        import dftpy.optimization.optimization as dftpy_optimization
    except Exception:
        return

    if getattr(dftpy_math_utils, "_codex_dcsrch_compat", False):
        return

    def _coerce_start_step(alpha0: float, amin: float, amax: float) -> float:
        alpha = float(alpha0)
        lower = max(float(amin), 0.0)
        upper = float(amax)
        if upper <= lower:
            return max(lower, 1e-3)
        if alpha <= lower:
            alpha = min(upper, max(lower + 1e-3, 1e-3))
        return alpha

    def _line_search_scalar(func, alpha0=None, func0=None, c1=1e-4, c2=0.9, amax=1.0, amin=0.0, xtol=1e-14, maxiter=100):
        cache: dict[float, object] = {}
        eval_count = 0

        def evaluate(alpha: float):
            nonlocal eval_count
            key = float(alpha)
            if key not in cache:
                cache[key] = func(key)
                eval_count += 1
            return cache[key]

        if func0 is None:
            func0 = evaluate(0.0)
        else:
            cache[0.0] = func0

        alpha1 = _coerce_start_step(0.0 if alpha0 is None else alpha0, amin=amin, amax=amax)
        search = DCSRCH(
            lambda a: float(evaluate(a)[0]),
            lambda a: float(evaluate(a)[1]),
            c1,
            c2,
            xtol,
            amin,
            amax,
        )
        alpha_out, _, _, task = search(alpha1, phi0=float(func0[0]), derphi0=float(func0[1]), maxiter=int(maxiter))
        if alpha_out is None:
            return None, float(func0[0]), float(func0[1]), task, eval_count, func0

        func1 = evaluate(alpha_out)
        if task[:5] == b"ERROR" or task[:4] == b"WARN":
            return None, float(func1[0]), float(func1[1]), task, eval_count, func1
        return alpha_out, float(func1[0]), float(func1[1]), task, eval_count, func1

    dftpy_math_utils.LineSearchDcsrch = _line_search_scalar
    dftpy_math_utils.LineSearchDcsrch2 = _line_search_scalar
    dftpy_math_utils._codex_dcsrch_compat = True
    dftpy_optimization.LineSearchDcsrch2 = _line_search_scalar


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run one official-style DFTpy/ASE UnitCellFilter relaxation for a VACLM point."
    )
    ap.add_argument("--source-case", required=True, help="Existing case folder containing raw VASP files and manifest.")
    ap.add_argument("--out-case", required=True, help="Output case folder for the official-style rerun.")
    ap.add_argument("--optimizer", default="BFGS", choices=["BFGS", "LBFGS", "BFGSLineSearch", "SciPyFminBFGS", "SciPyFminCG", "MDMin"])
    ap.add_argument("--fmax", type=float, default=None, help="ASE optimizer fmax. Defaults to point_manifest value.")
    ap.add_argument("--steps", type=int, default=None, help="ASE optimizer max steps. Defaults to point_manifest value.")
    ap.add_argument("--abort-filter-fmax", type=float, default=25.0, help="Abort a pathological cell relaxation if ASE filter fmax exceeds this value after --abort-after-steps.")
    ap.add_argument("--abort-after-steps", type=int, default=50, help="Start applying --abort-filter-fmax after this many optimizer steps.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing result.json.")
    return ap.parse_args()


def select_optimizer(name: str):
    table = {
        "BFGS": BFGS,
        "LBFGS": LBFGS,
        "BFGSLineSearch": BFGSLineSearch,
        "SciPyFminBFGS": SciPyFminBFGS,
        "SciPyFminCG": SciPyFminCG,
        "MDMin": MDMin,
    }
    return table[name]


def write_config_snapshot(path: Path, *, role: str, pp_name: str, spacing: float, xc: str, kedf: str, lam: float, mu: float) -> None:
    path.write_text(
        "\n".join(
            [
                "# Official-style DFTpyCalculator config snapshot.",
                "# This file documents the Python/ASE relaxation input.",
                "# The structural relaxation is driven by ASE UnitCellFilter, not by python -m dftpy.",
                "",
                "[JOB]",
                "calctype = Energy Force Stress",
                "",
                "[PATH]",
                "pppath = ./",
                "",
                "[PP]",
                f"Al = {pp_name}",
                "",
                "[GRID]",
                f"spacing = {spacing:.12f}",
                "",
                "[EXC]",
                f"xc = {xc}",
                "",
                "[KEDF]",
                f"kedf = {kedf}",
                f"x = {lam:.12f}",
                f"y = {mu:.12f}",
                "",
                "[OPT]",
                "method = TN",
                "maxiter = 300",
                "econv = 1e-6",
                "",
                f"# role = {role}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_dftpy_config(*, out_case: Path, pp_name: str, spacing: float, xc: str, kedf: str, lam: float, mu: float):
    conf = DefaultOption()
    conf["PATH"]["pppath"] = str(out_case)
    conf["PP"]["Al"] = pp_name
    conf["JOB"]["calctype"] = "Energy Force Stress"
    conf["EXC"]["xc"] = str(xc).upper()
    conf["KEDF"]["kedf"] = str(kedf).upper()
    conf["KEDF"]["x"] = float(lam)
    conf["KEDF"]["y"] = float(mu)
    conf["GRID"]["spacing"] = float(spacing)
    conf["OPT"]["method"] = "TN"
    conf["OPT"]["maxiter"] = 300
    conf["OPT"]["econv"] = 1e-6
    return OptionFormat(conf)


def extract_energy_terms_ev(atoms) -> tuple[dict[str, float], str]:
    calc = getattr(atoms, "calc", None)
    if calc is None:
        return {}, "missing_calculator"

    dftpy_results = getattr(calc, "dftpy_results", None)
    if not isinstance(dftpy_results, dict):
        return {}, "missing_calc_dftpy_results"

    energypotential = dftpy_results.get("energypotential")
    if energypotential is None:
        return {}, "missing_energypotential"

    terms: dict[str, float] = {}
    for name, output in energypotential.items():
        energy = getattr(output, "energy", None)
        if energy is None:
            continue
        try:
            terms[str(name)] = float(energy) * HARTREE_TO_EV
        except Exception:
            continue

    if not terms:
        return {}, "no_energy_terms_with_energy_attribute"

    terms["KEDF"] = float(sum(value for key, value in terms.items() if key.startswith("KEDF-")))
    return terms, "ok"


def write_dftpy_out(path: Path, *, energy_ev: float, stress_gpa: np.ndarray, terms: dict[str, float], term_status: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("DFTpy official-style ASE relaxation summary\n")
        f.write(f"total energy (eV) : {energy_ev:.12f}\n")
        f.write(f"energy_terms_status : {term_status}\n")
        if terms:
            f.write("ENERGY COMPONENTS (eV):\n")
            for key in sorted(terms):
                f.write(f"{key:30s} : {terms[key]: .12f}\n")
        f.write("TOTAL stress (GPa):\n")
        f.write(f"{stress_gpa[0,0]:14.6f} {stress_gpa[0,1]:14.6f} {stress_gpa[0,2]:14.6f}\n")
        f.write(f"{stress_gpa[1,0]:14.6f} {stress_gpa[1,1]:14.6f} {stress_gpa[1,2]:14.6f}\n")
        f.write(f"{stress_gpa[2,0]:14.6f} {stress_gpa[2,1]:14.6f} {stress_gpa[2,2]:14.6f}\n")


def run_role(*, role: str, atoms_path: Path, out_case: Path, pp_name: str, spacing: float, xc: str, kedf: str, lam: float, mu: float, optimizer_name: str, fmax: float, steps: int, abort_filter_fmax: float, abort_after_steps: int):
    atoms = read(str(atoms_path))
    conf = build_dftpy_config(out_case=out_case, pp_name=pp_name, spacing=spacing, xc=xc, kedf=kedf, lam=lam, mu=mu)
    calc = DFTpyCalculator(config=conf)
    atoms.calc = calc

    for old_name in [
        f"{role}_official_relax.log",
        f"{role}_official_relax.traj",
        f"{role}_official_dftpy.out",
        f"{role}_official_relaxed.vasp",
        f"{role}_official_relaxed.xyz",
    ]:
        old_path = out_case / old_name
        if old_path.exists():
            old_path.unlink()

    write_config_snapshot(
        out_case / f"official_dftpy_{role}_config.ini",
        role=role,
        pp_name=pp_name,
        spacing=spacing,
        xc=xc,
        kedf=kedf,
        lam=lam,
        mu=mu,
    )

    af = UnitCellFilter(atoms)
    optimizer_cls = select_optimizer(optimizer_name)
    dyn = optimizer_cls(
        af,
        logfile=str(out_case / f"{role}_official_relax.log"),
        trajectory=str(out_case / f"{role}_official_relax.traj"),
    )

    counter = {"n": 0}

    def abort_pathological_relaxation() -> None:
        counter["n"] += 1
        if counter["n"] < int(abort_after_steps):
            return
        filter_forces = np.asarray(af.get_forces(), dtype=float)
        if filter_forces.size == 0:
            return
        filter_fmax = float(np.linalg.norm(filter_forces, axis=1).max())
        if not np.isfinite(filter_fmax):
            raise RuntimeError(f"{role}: non-finite ASE filter fmax at step {counter['n']}")
        if filter_fmax > float(abort_filter_fmax):
            raise RuntimeError(
                f"{role}: aborting pathological relaxation at step {counter['n']}; "
                f"ASE filter fmax={filter_fmax:.6g} > {float(abort_filter_fmax):.6g}"
            )

    dyn.attach(abort_pathological_relaxation, interval=1)
    converged = bool(dyn.run(fmax=float(fmax), steps=int(steps)))

    energy_ev = float(atoms.get_potential_energy())
    stress_gpa = np.asarray(atoms.get_stress(voigt=False), dtype=float) * 160.21766208
    forces = np.asarray(atoms.get_forces(), dtype=float)
    final_fmax = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else float("nan")
    terms, term_status = extract_energy_terms_ev(atoms)

    write_dftpy_out(
        out_case / f"{role}_official_dftpy.out",
        energy_ev=energy_ev,
        stress_gpa=stress_gpa,
        terms=terms,
        term_status=term_status,
    )
    write(str(out_case / f"{role}_official_relaxed.vasp"), atoms, direct=True, vasp5=True)
    write(str(out_case / f"{role}_official_relaxed.xyz"), atoms)

    return {
        "converged": converged,
        "energy_eV": energy_ev,
        "stress_GPa": stress_gpa.tolist(),
        "final_fmax_eV_A": final_fmax,
        "energy_terms_eV": terms,
        "energy_terms_status": term_status,
        "cell_lengths_A": [float(x) for x in atoms.cell.lengths()],
        "cell_angles_deg": [float(x) for x in atoms.cell.angles()],
        "volume_A3": float(atoms.get_volume()),
        "trajectory": str(out_case / f"{role}_official_relax.traj"),
        "relax_log": str(out_case / f"{role}_official_relax.log"),
        "dftpy_out": str(out_case / f"{role}_official_dftpy.out"),
        "relaxed_vasp": str(out_case / f"{role}_official_relaxed.vasp"),
        "optimizer_steps_recorded_by_guard": int(counter["n"]),
    }


def copy_inputs(source_case: Path, out_case: Path) -> None:
    out_case.mkdir(parents=True, exist_ok=True)
    for name in [
        "point_manifest.json",
        "al.lda.recpot",
        "pristine_raw.vasp",
        "vacancy_start.vasp",
        "dftpy_pristine_input.ini",
        "dftpy_vacancy_input.ini",
    ]:
        src = source_case / name
        if src.exists():
            shutil.copy2(src, out_case / name)


def main() -> int:
    args = parse_args()
    patch_dftpy_linesearch_compat()

    source_case = Path(args.source_case).resolve()
    out_case = Path(args.out_case).resolve()
    result_path = out_case / "result.json"
    if result_path.exists() and not args.force:
        print(f"[SKIP] result exists: {result_path}")
        return 0

    copy_inputs(source_case, out_case)

    manifest_path = out_case / "point_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pp_name = "al.lda.recpot"
    spacing = float(manifest.get("spacing_A_derived_from_ecut", 0.2503431520318134))
    xc = str(manifest.get("xc", "LDA"))
    kedf = str(manifest.get("kedf", "TFVW"))
    lam = float(manifest.get("lambda", manifest.get("kedf_x")))
    mu = float(manifest.get("mu", manifest.get("kedf_y")))
    fmax = float(args.fmax if args.fmax is not None else manifest.get("fmax_eV_per_A", 0.01))
    steps = int(args.steps if args.steps is not None else manifest.get("relax_steps", 5000))
    optimizer = str(args.optimizer)

    metadata = {
        "workflow": "official_dftpy_relax_tutorial_style",
        "source_case": str(source_case),
        "out_case": str(out_case),
        "official_reference": "https://dftpy.rutgers.edu/tutorials/ofdft/relax.html",
        "relaxation_driver": "ASE UnitCellFilter + DFTpyCalculator",
        "trajectory_policy": "ASE trajectory written for pristine and vacancy",
        "optimizer": optimizer,
        "fmax_eV_A": fmax,
        "steps": steps,
        "abort_filter_fmax": float(args.abort_filter_fmax),
        "abort_after_steps": int(args.abort_after_steps),
        "ecut_eV": float(manifest.get("ecut_eV", 600.0)),
        "spacing_A": spacing,
        "xc": xc,
        "kedf": kedf,
        "lambda": lam,
        "mu": mu,
    }
    (out_case / "official_relax_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    pristine = run_role(
        role="pristine",
        atoms_path=out_case / "pristine_raw.vasp",
        out_case=out_case,
        pp_name=pp_name,
        spacing=spacing,
        xc=xc,
        kedf=kedf,
        lam=lam,
        mu=mu,
        optimizer_name=optimizer,
        fmax=fmax,
        steps=steps,
        abort_filter_fmax=float(args.abort_filter_fmax),
        abort_after_steps=int(args.abort_after_steps),
    )
    vacancy = run_role(
        role="vacancy",
        atoms_path=out_case / "vacancy_start.vasp",
        out_case=out_case,
        pp_name=pp_name,
        spacing=spacing,
        xc=xc,
        kedf=kedf,
        lam=lam,
        mu=mu,
        optimizer_name=optimizer,
        fmax=fmax,
        steps=steps,
        abort_filter_fmax=float(args.abort_filter_fmax),
        abort_after_steps=int(args.abort_after_steps),
    )

    n_pristine = int(manifest.get("pristine_n_atoms", 108))
    n_vacancy = int(manifest.get("vacancy_n_atoms", 107))
    scale = float(n_vacancy) / float(n_pristine)
    vacancy_formation_energy = float(vacancy["energy_eV"] - scale * pristine["energy_eV"])

    pristine_kedf = pristine["energy_terms_eV"].get("KEDF")
    vacancy_kedf = vacancy["energy_terms_eV"].get("KEDF")
    if pristine_kedf is None or vacancy_kedf is None:
        vacancy_formation_kedf = None
        kedf_status = "missing_pristine_or_vacancy_kedf"
    else:
        vacancy_formation_kedf = float(vacancy_kedf - scale * pristine_kedf)
        kedf_status = "ok"

    lattice_constant_mean_A = float(sum(pristine["cell_lengths_A"]) / 9.0)
    lattice_constant_from_volume_A = float((pristine["volume_A3"] / 27.0) ** (1.0 / 3.0))

    result = {
        **metadata,
        "setting": str(manifest.get("setting", source_case.name)),
        "status": "ok",
        "pristine_n_atoms": n_pristine,
        "vacancy_n_atoms": n_vacancy,
        "pristine": pristine,
        "vacancy": vacancy,
        "pristine_energy_eV": pristine["energy_eV"],
        "vacancy_energy_eV": vacancy["energy_eV"],
        "vacancy_formation_energy_eV": vacancy_formation_energy,
        "pristine_kedf_energy_eV": pristine_kedf,
        "vacancy_kedf_energy_eV": vacancy_kedf,
        "vacancy_formation_kedf_energy_eV": vacancy_formation_kedf,
        "kedf_formation_status": kedf_status,
        "lattice_constant_A": lattice_constant_mean_A,
        "lattice_constant_from_pristine_volume_A": lattice_constant_from_volume_A,
        "formation_energy_formula_total": "E_f^vac = E_vac^(N-1) - ((N-1)/N) E_pristine^N",
        "formation_energy_formula_kedf": "KEDF_f^vac = KEDF_vac^(N-1) - ((N-1)/N) KEDF_pristine^N",
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "setting": result["setting"], "result": str(result_path)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("[ERROR]", repr(exc), file=sys.stderr)
        traceback.print_exc()
        raise
