#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from ase.constraints import FixAtoms
from ase.optimize import BFGS, FIRE

from dftpy.config import DefaultOption, OptionFormat
from dftpy.api.api4ase import DFTpyCalculator


def run_dft_relax(atoms, fixed_mask, config_args):
    """
    用 DFTpy 進行幾何優化（relax），並回傳 (relaxed_atoms, energy)

    注意：
    - 只固定 fixed_mask 那些原子（bottom/top grips）
    - 不設定 PATH.workdir（你的 DFTpy 版本不支援，會 KeyError）
    """
    calc_atoms = atoms.copy()
    calc_atoms.set_constraint(FixAtoms(mask=fixed_mask))

    conf = DefaultOption()
    conf["PATH"]["pppath"] = str(Path(config_args.pppath))
    conf["PP"][config_args.element] = str(config_args.ppfile)
    conf["JOB"]["calctype"] = "Energy Force"

    conf["KEDF"]["kedf"] = config_args.kedf
    conf["GRID"]["spacing"] = config_args.spacing

    calc_atoms.calc = DFTpyCalculator(config=OptionFormat(conf))

    opt_type = FIRE if str(config_args.optimizer).upper() == "FIRE" else BFGS
    opt = opt_type(calc_atoms, logfile=None)
    opt.run(fmax=float(config_args.fmax), steps=int(config_args.relax_steps))

    energy = calc_atoms.get_potential_energy()
    return calc_atoms, energy

