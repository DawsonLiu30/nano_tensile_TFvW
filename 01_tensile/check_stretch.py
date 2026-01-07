#!/usr/bin/env python3
import glob
import numpy as np
from ase.io import read

OUTDIR = "/work/dawson666/dftpy_project/nano_tensile_Al300Vac/02_results/run_Al300Vac"
GRIP_THICKNESS = 3.0

def grip_masks(atoms, t):
    z = atoms.positions[:, 2]
    zmin, zmax = z.min(), z.max()
    bottom = z <= zmin + t
    top = z >= zmax - t
    middle = ~(bottom | top)
    return bottom, top, middle

def summary(xyz):
    atoms = read(xyz)
    bottom, top, middle = grip_masks(atoms, GRIP_THICKNESS)
    z = atoms.positions[:, 2]
    L_grip = z[top].mean() - z[bottom].mean()
    L_mid = z[middle].max() - z[middle].min()
    return L_grip, L_mid, int(bottom.sum()), int(top.sum()), int(middle.sum())

files = sorted(glob.glob(f"{OUTDIR}/cycle_*_stretched.xyz") + glob.glob(f"{OUTDIR}/cycle_*_relaxed.xyz"))
if not files:
    print("No cycle_*.xyz found in OUTDIR:", OUTDIR)
    raise SystemExit(1)

print("file,L_grip_A,L_mid_A,nbottom,ntop,nmid")
for f in files:
    Lg, Lm, nb, nt, nm = summary(f)
    print(f"{f.split('/')[-1]},{Lg:.6f},{Lm:.6f},{nb},{nt},{nm}")

