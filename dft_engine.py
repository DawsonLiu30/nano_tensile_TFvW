from __future__ import annotations

from pathlib import Path
import os
import numpy as np

from ase.constraints import FixAtoms
from ase.optimize import BFGS

from dftpy.config import DefaultOption, OptionFormat
from dftpy.api.api4ase import DFTpyCalculator


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
    conf["KEDF"]["kedf"] = "WT"
    conf["OPT"]["method"] = "TN"
    conf["GRID"]["spacing"] = float(spacing)

    return OptionFormat(conf)


def relax_atoms(
    atoms,
    pp_file: str | Path,
    spacing: float,
    fixed_idx,
    fmax: float = 0.05,
    steps: int = 200,
    logfile: str | None = None,
    trajfile: str | None = None,
    dftpy_outfile: str | None = None,
    debug_fixed: bool = True,
):
    fixed_idx = np.asarray(fixed_idx, dtype=int).ravel()
    atoms.set_constraint(FixAtoms(indices=fixed_idx))

    conf = _build_dftpy_config(pp_file=pp_file, spacing=spacing, atoms=atoms)
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

    if dftpy_outfile:
        with open(dftpy_outfile, "w") as f:
            f.write(f"total energy (eV) : {E:.12f}\n")
            f.write("TOTAL stress (GPa):\n")
            f.write(f"{S[0,0]:14.6f} {S[0,1]:14.6f} {S[0,2]:14.6f}\n")
            f.write(f"{S[1,0]:14.6f} {S[1,1]:14.6f} {S[1,2]:14.6f}\n")
            f.write(f"{S[2,0]:14.6f} {S[2,1]:14.6f} {S[2,2]:14.6f}\n")

    return atoms, E, S

