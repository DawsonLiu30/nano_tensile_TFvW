#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make Al FCC supercell (~300 atoms), add 5 Å vacuum, then create a single vacancy.

Default:
  FCC conventional cell (cubic=True) has 4 atoms
  reps = (5, 5, 3) -> 4*5*5*3 = 300 atoms

Vacuum:
  Add 5 Å padding around the cluster inside a big box (PBC True, but large vacuum).

Vacancy:
  Remove the atom closest to the geometric center (simple + deterministic).
"""

import argparse
import numpy as np
from ase.build import bulk
from ase.io import write


def build_supercell(element="Al", a=4.05, reps=(5, 5, 3)):
    uc = bulk(element, "fcc", a=a, cubic=True)  # conventional FCC cell, 4 atoms
    atoms = uc.repeat(reps)
    return atoms


def embed_in_vacuum(atoms, vacuum=5.0):
    pos = atoms.get_positions()
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)

    lengths = (maxs - mins) + 2.0 * vacuum
    cell = np.diag(lengths)

    pos_shifted = pos - mins + vacuum  # shift into box with padding
    atoms.set_cell(cell)
    atoms.set_positions(pos_shifted)
    atoms.set_pbc(True)  # DFTpy expects PBC; vacuum makes it effectively isolated
    return atoms


def remove_center_atom(atoms):
    pos = atoms.get_positions()
    center = pos.mean(axis=0)
    d2 = ((pos - center) ** 2).sum(axis=1)
    idx = int(np.argmin(d2))
    removed_pos = pos[idx].copy()
    del atoms[idx]
    return idx, removed_pos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=float, default=4.05, help="Al lattice constant (Å)")
    p.add_argument("--reps", type=int, nargs=3, default=[5, 5, 3], help="Repeat nx ny nz (default 5 5 3 -> 300 atoms)")
    p.add_argument("--vacuum", type=float, default=5.0, help="Vacuum padding (Å), default 5 Å")
    p.add_argument("--out", default="Al_supercell300_vac5_vacancy.xyz", help="Output xyz filename")
    args = p.parse_args()

    atoms = build_supercell(a=args.a, reps=tuple(args.reps))
    nat0 = len(atoms)

    atoms = embed_in_vacuum(atoms, vacuum=args.vacuum)

    idx, rpos = remove_center_atom(atoms)
    nat1 = len(atoms)

    write(args.out, atoms)

    print("[OK] Structure generated:")
    print(f"     reps = {tuple(args.reps)}  -> atoms before vacancy = {nat0}")
    print(f"     atoms after vacancy = {nat1}")
    print(f"     removed atom index (before deletion) = {idx}")
    print(f"     removed atom position (Å) = {rpos}")
    print(f"     cell (Å) =\n{atoms.get_cell()}")
    print(f"     wrote: {args.out}")


if __name__ == "__main__":
    main()

