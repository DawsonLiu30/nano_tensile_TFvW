#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from ase.constraints import FixAtoms
from ase.optimize import BFGS, FIRE

from dftpy.config import DefaultOption, OptionFormat
from dftpy.api.api4ase import DFTpyCalculator


def run_dft_relax(atoms, fixed_mask, config_args, *, workdir: str | Path | None = None):
    """
    呼叫 DFTpy 進行幾何優化（ASE optimizer 驅動）
    回傳：relaxed_atoms, energy
    """
    calc_atoms = atoms.copy()
    calc_atoms.set_constraint(FixAtoms(mask=fixed_mask))

    # 建立 DFTpy 配置
    conf = DefaultOption()

    pppath = Path(config_args.pppath).expanduser().resolve()
    ppfile = str(config_args.ppfile)

    conf["PATH"]["pppath"] = str(pppath)
    conf["PP"][config_args.element] = ppfile
    conf["JOB"]["calctype"] = "Energy Force"
    conf["KEDF"]["kedf"] = config_args.kedf
    conf["GRID"]["spacing"] = config_args.spacing

    if workdir is not None:
        conf["PATH"]["workdir"] = str(Path(workdir).resolve())

    calc_atoms.calc = DFTpyCalculator(config=OptionFormat(conf))

    # 優化器
    opt_name = config_args.optimizer.upper()
    opt_cls = FIRE if opt_name == "FIRE" else BFGS

    # 留 log 會救命（老賊罵你時你可以甩 log 給他）
    logfile = getattr(config_args, "opt_log", None)
    opt = opt_cls(calc_atoms, logfile=logfile)

    opt.run(fmax=config_args.fmax, steps=config_args.relax_steps)

    energy = float(calc_atoms.get_potential_energy())
    return calc_atoms, energy
