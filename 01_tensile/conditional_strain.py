#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple, Dict

import numpy as np

import ase.io
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io.trajectory import Trajectory

from dftpy.config import DefaultOption, OptionFormat
from dftpy.api.api4ase import DFTpyCalculator


def stretch_free_region_z(atoms: Atoms, fixed_mask: np.ndarray, z_mid: float, stretch: float,tmp_filename="") -> Atoms:
    """
    Scale ONLY free region along z about z_mid; grips remain unchanged.

    atoms       : is the atoms object containing information (postition, lattice, etc.) of the material
    fixed_mask  :
    NOT NEEDED: z_mid       : center part
    stretch     :
    tmp_filename: give name different from "" to write the intermediate strain structure
    """

    # ToDO : redo according to discussion on 7th jan 2026
    pos = atoms.get_positions().copy()
    free = ~fixed_mask
    pos[free, 2] = z_mid + stretch * (pos[free, 2] - z_mid)
    atoms.set_positions(pos)

    if not tmp_filename=="":
        from ase.io import write
        write(atoms,tmp_filename,format='xyz')

    return atoms