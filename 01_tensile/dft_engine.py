from __future__ import annotations

from pathlib import Path
import numpy as np

from ase.constraints import FixAtoms
from ase.optimize import BFGS

from dftpy.config import DefaultOption, OptionFormat
from dftpy.api.api4ase import DFTpyCalculator


def relax_atoms(atoms, pp_file, spacing, fixed_idx, fmax=0.05, steps=200, logfile=None, trajfile=None, dftpy_outfile=None):
    fixed_idx = np.asarray(fixed_idx, dtype=int).ravel()
    atoms.set_constraint(FixAtoms(indices=fixed_idx))

    pp = Path(pp_file).expanduser().resolve()
    conf = DefaultOption()
    conf["PATH"]["pppath"] = str(pp.parent)
    conf["PP"]["Al"] = pp.name
    conf["JOB"]["calctype"] = "Energy Force Stress"
    conf["KEDF"]["kedf"] = "TFvW"
    conf["GRID"]["spacing"] = float(spacing)
    conf = OptionFormat(conf)

    calc = DFTpyCalculator(config=conf)
    atoms.set_calculator(calc)

    dyn = BFGS(atoms, trajectory=trajfile, logfile=logfile)
    dyn.run(fmax=float(fmax), steps=int(steps))

    E = float(atoms.get_potential_energy())
    S = atoms.get_stress(voigt=False) * 160.21766208

    if dftpy_outfile:
        with open(dftpy_outfile, "w") as f:
            f.write(f"total energy (eV) : {E:.12f}\n")
            f.write("TOTAL stress (GPa):\n")
            f.write(f"{S[0,0]:14.6f} {S[0,1]:14.6f} {S[0,2]:14.6f}\n")
            f.write(f"{S[1,0]:14.6f} {S[1,1]:14.6f} {S[1,2]:14.6f}\n")
            f.write(f"{S[2,0]:14.6f} {S[2,1]:14.6f} {S[2,2]:14.6f}\n")

    return atoms
