from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.constraints import FixAtoms
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS, FIRE
from dftpy.api.api4ase import DFTpyCalculator
from dftpy.config import DefaultOption, OptionFormat


def _install_dftpy_linesearch_compat() -> bool:
    try:
        from scipy.optimize import minpack2
    except Exception:
        return False

    if hasattr(minpack2, "dcsrch"):
        return False

    try:
        from scipy.optimize._dcsrch import DCSRCH
        import dftpy.math_utils as math_utils
        import dftpy.optimization.optimization as optimization_mod
    except Exception:
        return False

    if getattr(math_utils.LineSearchDcsrch2, "__name__", "") == "_compat_linesearch_dcsrch2":
        return False

    def _compat_linesearch_dcsrch2(
        func,
        alpha0=None,
        func0=None,
        c1=1e-4,
        c2=0.9,
        amax=1.0,
        amin=0.0,
        xtol=1e-14,
        maxiter=100,
    ):
        if alpha0 is None:
            alpha0 = 0.0
        if func0 is None:
            func0 = func(0.0)

        cache = {0.0: func0}

        def _scalar(alpha) -> float:
            return float(np.asarray(alpha, dtype=float))

        def _evaluate(alpha):
            key = _scalar(alpha)
            if key not in cache:
                cache[key] = func(alpha)
            return cache[key]

        phi0 = float(func0[0])
        derphi0 = float(func0[1])
        start_alpha = _scalar(alpha0)
        if not np.isfinite(start_alpha) or start_alpha <= 0.0:
            start_alpha = max(float(amin) + 1e-8, 1e-3)
        start_alpha = min(start_alpha, float(amax))

        search = DCSRCH(
            lambda alpha: float(_evaluate(alpha)[0]),
            lambda alpha: float(_evaluate(alpha)[1]),
            float(c1),
            float(c2),
            float(xtol),
            float(amin),
            float(amax),
        )
        alpha1, _, _, task = search(
            start_alpha,
            phi0=phi0,
            derphi0=derphi0,
            maxiter=int(maxiter),
        )
        if alpha1 is None:
            return None, phi0, derphi0, task, max(0, len(cache) - 1), func0

        func1 = _evaluate(alpha1)
        return (
            float(alpha1),
            float(func1[0]),
            float(func1[1]),
            task,
            max(0, len(cache) - 1),
            func1,
        )

    math_utils.LineSearchDcsrch2 = _compat_linesearch_dcsrch2
    optimization_mod.LineSearchDcsrch2 = _compat_linesearch_dcsrch2
    return True


def _build_dftpy_config(pp_file: str | Path, spacing: float, atoms) -> OptionFormat:
    pp_path = Path(pp_file).expanduser().resolve()
    if not pp_path.exists():
        raise FileNotFoundError(f"pp_file not found: {pp_path}")

    conf = DefaultOption()
    conf["PATH"]["pppath"] = str(pp_path.parent)

    syms = sorted(set(atoms.get_chemical_symbols()))
    for sym in syms:
        conf["PP"][sym] = pp_path.name

    conf["JOB"]["calctype"] = "Energy Force Stress"
    conf["KEDF"]["kedf"] = "SM"
    conf["OPT"]["method"] = "LBFGS"
    conf["GRID"]["spacing"] = float(spacing)

    return OptionFormat(conf)


def relax_atoms(
    atoms,
    pp_file: str | Path,
    spacing: float,
    fixed_idx,
    fmax: float = 0.02,
    steps: int = 200,
    bfgs_maxstep: float = 0.04,
    fire_maxstep: float = 0.04,
    logfile: str | None = None,
    trajfile: str | None = None,
    dftpy_outfile: str | None = None,
    debug_fixed: bool = True,
):
    fixed_idx = np.asarray(fixed_idx, dtype=int).ravel()
    atoms.set_constraint(FixAtoms(indices=fixed_idx))

    conf = _build_dftpy_config(pp_file=pp_file, spacing=spacing, atoms=atoms)
    if _install_dftpy_linesearch_compat():
        print("[relax_atoms] linesearch compatibility installed")
    calc = DFTpyCalculator(config=conf)
    atoms.calc = calc

    if logfile:
        Path(logfile).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    if trajfile:
        Path(trajfile).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    if dftpy_outfile:
        Path(dftpy_outfile).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    if debug_fixed and fixed_idx.size > 0:
        z = atoms.get_positions()[:, 2]
        print(
            f"[relax_atoms] before run fixed_z(min/max)="
            f"{z[fixed_idx].min():.6f}/{z[fixed_idx].max():.6f}"
        )

    switch_fmax = max(float(fmax), 0.05)
    log_fh = open(logfile, "a", buffering=1) if logfile else None
    traj = Trajectory(trajfile, "a", atoms) if trajfile else None

    try:
        dyn_bfgs = BFGS(
            atoms,
            logfile=log_fh,
            maxstep=float(bfgs_maxstep),
        )
        if traj is not None:
            dyn_bfgs.attach(traj.write, interval=1)
        dyn_bfgs.run(fmax=switch_fmax, steps=int(steps))

        steps_taken = dyn_bfgs.get_number_of_steps()
        steps_left = int(steps) - steps_taken

        if float(fmax) < switch_fmax and steps_left > 0:
            print(
                f"\n[relax_atoms] switching to FIRE for final convergence "
                f"(target fmax={float(fmax):.4f}, steps_left={steps_left})"
            )
            dyn_fire = FIRE(
                atoms,
                logfile=log_fh,
                maxstep=float(fire_maxstep),
            )
            if traj is not None:
                dyn_fire.attach(traj.write, interval=1)
            dyn_fire.run(fmax=float(fmax), steps=steps_left)
    finally:
        if traj is not None:
            traj.close()
        if log_fh is not None:
            log_fh.close()

    if debug_fixed and fixed_idx.size > 0:
        z2 = atoms.get_positions()[:, 2]
        print(
            f"[relax_atoms] after  run fixed_z(min/max)="
            f"{z2[fixed_idx].min():.6f}/{z2[fixed_idx].max():.6f}"
        )

    energy = float(atoms.get_potential_energy())
    stress = atoms.get_stress(voigt=False) * 160.21766208

    if dftpy_outfile:
        with open(dftpy_outfile, "w", encoding="utf-8") as fh:
            fh.write(f"total energy (eV) : {energy:.12f}\n")
            fh.write("TOTAL stress (GPa):\n")
            fh.write(f"{stress[0,0]:14.6f} {stress[0,1]:14.6f} {stress[0,2]:14.6f}\n")
            fh.write(f"{stress[1,0]:14.6f} {stress[1,1]:14.6f} {stress[1,2]:14.6f}\n")
            fh.write(f"{stress[2,0]:14.6f} {stress[2,1]:14.6f} {stress[2,2]:14.6f}\n")

    return atoms, energy, stress
