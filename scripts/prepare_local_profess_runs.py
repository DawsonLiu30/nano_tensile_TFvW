#!/usr/bin/env python3
"""Prepare local PROFESS single-point runs from VASP structures."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ase.io import read


DEFAULT_INPT = """ecut 1600
meth NTN
KINE WT
exch lda
geometryfile {stem}.ion

print minimizer density 2
calculate stresses
"""


def vasp_to_profess_ion(vasp_path: Path, ion_path: Path, pseudo_name: str) -> None:
    atoms = read(str(vasp_path), format="vasp")
    cell = atoms.cell.array
    scaled = atoms.get_scaled_positions(wrap=False)
    symbols = atoms.get_chemical_symbols()

    with ion_path.open("w", encoding="utf-8") as fh:
        fh.write("%BLOCK LATTICE_CART\n")
        for vec in cell:
            fh.write(f"{vec[0]:18.10f} {vec[1]:18.10f} {vec[2]:18.10f}\n")
        fh.write("%END BLOCK LATTICE_CART\n")
        fh.write("%BLOCK POSITIONS_FRAC\n")
        for sym, pos in zip(symbols, scaled):
            fh.write(f"{sym:2s} {pos[0]:18.10f} {pos[1]:18.10f} {pos[2]:18.10f}\n")
        fh.write("%END BLOCK POSITIONS_FRAC\n")
        fh.write("%BLOCK SPECIES_POT\n")
        fh.write(f"    Al {pseudo_name}\n")
        fh.write("%END BLOCK SPECIES_POT\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--profess", required=True, type=Path)
    parser.add_argument("--pseudo", required=True, type=Path)
    parser.add_argument("--inpt-template", type=Path)
    parser.add_argument("--vasp", required=True, action="append", type=Path)
    args = parser.parse_args()

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    profess_dst = outdir / "PROFESS"
    pseudo_dst = outdir / args.pseudo.name
    shutil.copy2(args.profess, profess_dst)
    shutil.copy2(args.pseudo, pseudo_dst)

    template_text = DEFAULT_INPT
    if args.inpt_template:
        template_text = args.inpt_template.read_text(encoding="utf-8", errors="ignore")

    run_lines = ["#!/bin/bash", "set -e", "chmod +x ./PROFESS"]
    manifest_lines = ["case,vasp,ion,inpt"]

    for vasp in args.vasp:
        vasp = vasp.resolve()
        stem = vasp.stem
        case_dir = outdir / stem
        case_dir.mkdir(parents=True, exist_ok=True)
        vasp_dst = case_dir / vasp.name
        ion_path = case_dir / f"{stem}.ion"
        inpt_path = case_dir / f"{stem}.inpt"
        shutil.copy2(vasp, vasp_dst)
        shutil.copy2(args.pseudo, case_dir / args.pseudo.name)
        vasp_to_profess_ion(vasp, ion_path, pseudo_dst.name)
        inpt_path.write_text(template_text.format(stem=stem), encoding="utf-8")
        run_lines.append(f"(cd {case_dir.name} && ../PROFESS {stem} > {stem}.stdout 2> {stem}.stderr)")
        manifest_lines.append(f"{stem},{vasp_dst.name},{ion_path.name},{inpt_path.name}")

    run_path = outdir / "run_all_profess.sh"
    run_path.write_text("\n".join(run_lines) + "\n", encoding="utf-8")
    (outdir / "manifest.csv").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"Wrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
