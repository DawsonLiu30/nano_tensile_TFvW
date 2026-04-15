from __future__ import annotations

from pathlib import Path
import numpy as np

from ase.constraints import FixAtoms
from ase.optimize import BFGS

from dftpy.config import DefaultOption, OptionFormat
from dftpy.api.api4ase import DFTpyCalculator


def _patch_dftpy_linesearch_compat() -> None:
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

        if alpha0 is None:
            alpha0 = 0.0
        alpha1 = _coerce_start_step(alpha0, amin=amin, amax=amax)

        search = DCSRCH(
            lambda a: float(evaluate(a)[0]),
            lambda a: float(evaluate(a)[1]),
            c1,
            c2,
            xtol,
            amin,
            amax,
        )
        alpha_out, _, _, task = search(
            alpha1,
            phi0=float(func0[0]),
            derphi0=float(func0[1]),
            maxiter=int(maxiter),
        )
        if alpha_out is None:
            return None, float(func0[0]), float(func0[1]), task, eval_count, func0

        func1 = evaluate(alpha_out)
        if task[:5] == b"ERROR" or task[:4] == b"WARN":
            return None, float(func1[0]), float(func1[1]), task, eval_count, func1
        return alpha_out, float(func1[0]), float(func1[1]), task, eval_count, func1

    def _line_search_vector(func, alpha0=None, func0=None, c1=1e-4, c2=0.9, amax=1.0, amin=0.0, xtol=1e-14, maxiter=100):
        econv = 1e-5
        total_evals = 0

        if alpha0 is None or func0 is None:
            raise ValueError("alpha0 and func0 are required for vector line search.")

        alpha1 = alpha0.copy()
        x1 = float(func0[0])
        g1 = np.asarray(func0[1], dtype=float)
        func1 = func0

        resA = []
        dirA = []
        task = b"START"

        for _ in range(1, 5):
            alpha_base = alpha1.copy()
            resA.append(g1)
            direction = dftpy_math_utils.get_direction_CG(resA, dirA=dirA, method="CG-PR")
            dirA.append(direction)
            grad0 = float((g1 * direction).sum())
            if grad0 > 0.0:
                direction = -g1
                grad0 = float((g1 * direction).sum())

            factor = float(np.abs(direction).max())
            if factor <= 0.0:
                break

            beta0 = min(0.1, 0.1 * np.pi / factor)
            beta0 = _coerce_start_step(beta0, amin=amin / factor, amax=amax / factor)
            cache: dict[float, object] = {0.0: func0}
            eval_count = 0

            def evaluate(beta: float):
                nonlocal eval_count
                key = float(beta)
                if key not in cache:
                    cache[key] = func(alpha_base + key * direction)
                    eval_count += 1
                return cache[key]

            search = DCSRCH(
                lambda b: float(evaluate(b)[0]),
                lambda b: float((np.asarray(evaluate(b)[1], dtype=float) * direction).sum()),
                c1,
                c2,
                xtol,
                amin / factor,
                amax / factor,
            )
            beta, _, _, task = search(
                beta0,
                phi0=x1,
                derphi0=grad0,
                maxiter=int(maxiter),
            )
            total_evals += eval_count

            if beta is None:
                alpha1 = None
                break

            func1 = evaluate(beta)
            x1 = float(func1[0])
            g1 = np.asarray(func1[1], dtype=float)
            if task[:5] == b"ERROR" or task[:4] == b"WARN":
                alpha1 = None
                break

            alpha1 = alpha_base + float(beta) * direction
            if float((g1 * g1).sum()) < econv:
                break

        return alpha1, x1, g1, task, total_evals, func1

    dftpy_math_utils.LineSearchDcsrch = _line_search_scalar
    dftpy_math_utils.LineSearchDcsrch2 = _line_search_scalar
    dftpy_math_utils.LineSearchDcsrchVector = _line_search_vector
    dftpy_math_utils._codex_dcsrch_compat = True

    dftpy_optimization.LineSearchDcsrch2 = _line_search_scalar
    dftpy_optimization.LineSearchDcsrchVector = _line_search_vector


_patch_dftpy_linesearch_compat()


def normalize_kedf_name(kedf: str | None) -> str:
    if kedf is None or not str(kedf).strip():
        return "TFVW"

    name = str(kedf).strip().upper()
    compact = "".join(ch for ch in name if ch.isalnum())
    aliases = {
        "TFVW": "TFVW",
        "XTFYVW": "TFVW",
        "SM": "SM",
        "WT": "WT",
        "FP": "FP",
        "VW": "VW",
        "TF": "TF",
        "WTE": "WTE",
    }
    return aliases.get(compact, name)


def _build_dftpy_config(pp_file: str | Path, spacing: float, atoms, kedf: str = "TFVW") -> OptionFormat:
    pp_path = Path(pp_file).expanduser().resolve()
    if not pp_path.exists():
        raise FileNotFoundError(f"pp_file not found: {pp_path}")

    conf = DefaultOption()
    conf["PATH"]["pppath"] = str(pp_path.parent)

    syms = sorted(set(atoms.get_chemical_symbols()))
    for sym in syms:
        conf["PP"][sym] = pp_path.name

    conf["JOB"]["calctype"] = "Energy Force Stress"
    conf["KEDF"]["kedf"] = normalize_kedf_name(kedf)
    conf["OPT"]["method"] = "LBFGS"
    conf["GRID"]["spacing"] = float(spacing)

    return OptionFormat(conf)


def _write_dftpy_out(dftpy_outfile: str | None, energy_ev: float, stress_gpa: np.ndarray) -> None:
    if not dftpy_outfile:
        return

    with open(dftpy_outfile, "w", encoding="utf-8") as f:
        f.write(f"total energy (eV) : {energy_ev:.12f}\n")
        f.write("TOTAL stress (GPa):\n")
        f.write(f"{stress_gpa[0,0]:14.6f} {stress_gpa[0,1]:14.6f} {stress_gpa[0,2]:14.6f}\n")
        f.write(f"{stress_gpa[1,0]:14.6f} {stress_gpa[1,1]:14.6f} {stress_gpa[1,2]:14.6f}\n")
        f.write(f"{stress_gpa[2,0]:14.6f} {stress_gpa[2,1]:14.6f} {stress_gpa[2,2]:14.6f}\n")


def evaluate_atoms(
    atoms,
    pp_file: str | Path,
    spacing: float,
    kedf: str = "TFVW",
    dftpy_outfile: str | None = None,
):
    if dftpy_outfile:
        Path(dftpy_outfile).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    conf = _build_dftpy_config(pp_file=pp_file, spacing=spacing, atoms=atoms, kedf=kedf)
    calc = DFTpyCalculator(config=conf)
    atoms.calc = calc

    energy_ev = float(atoms.get_potential_energy())
    stress_gpa = atoms.get_stress(voigt=False) * 160.21766208
    _write_dftpy_out(dftpy_outfile, energy_ev=energy_ev, stress_gpa=stress_gpa)
    return atoms, energy_ev, stress_gpa


def relax_atoms(
    atoms,
    pp_file: str | Path,
    spacing: float,
    fixed_idx,
    kedf: str = "TFVW",
    fmax: float = 0.05,
    steps: int = 200,
    logfile: str | None = None,
    trajfile: str | None = None,
    dftpy_outfile: str | None = None,
    debug_fixed: bool = True,
):
    fixed_idx = np.asarray(fixed_idx, dtype=int).ravel()
    atoms.set_constraint(FixAtoms(indices=fixed_idx))

    conf = _build_dftpy_config(pp_file=pp_file, spacing=spacing, atoms=atoms, kedf=kedf)
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
        print(f"[relax_atoms] before run fixed_z(min/max)={z[fixed_idx].min():.6f}/{z[fixed_idx].max():.6f}")

    dyn = BFGS(atoms, trajectory=trajfile, logfile=logfile)
    dyn.run(fmax=float(fmax), steps=int(steps))

    if debug_fixed and fixed_idx.size > 0:
        z2 = atoms.get_positions()[:, 2]
        print(f"[relax_atoms] after  run fixed_z(min/max)={z2[fixed_idx].min():.6f}/{z2[fixed_idx].max():.6f}")

    E = float(atoms.get_potential_energy())
    S = atoms.get_stress(voigt=False) * 160.21766208

    _write_dftpy_out(dftpy_outfile, energy_ev=E, stress_gpa=S)

    return atoms, E, S
