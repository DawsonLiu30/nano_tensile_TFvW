#!/usr/bin/env python3
"""Run local PROFESS KEDF sweep for vacancy nanostructure radius cases.

This script is intended for local Windows/WSL execution because the provided
PROFESS binary is a Linux executable.  It prepares PROFESS input files from
VASP structures, runs SCF-only density minimization, parses total energies,
and plots direction-wise relative energies versus radius.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ase.io import read


CASE_RE = re.compile(
    r"Al_vac_(?P<direction>\d+)_r(?P<radius>[\d.]+)_zr(?P<zr>[\d.]+)_n(?P<natoms>\d+)"
)


KEDF_TEMPLATES: dict[str, list[str]] = {
    "WT": ["KINE WT"],
    "WGC": [
        "KINE WGC",
        "WGCT 2",
        "PARA LAMB 1",
        "PARA MU 1",
        "PARA ALPHA 1.206011329583298",
        "PARA BETA 0.460655337083368",
        "PARA GAMMA 2.7",
    ],
    "CAT": ["KINE CAT"],
    "HC": ["KINE HC", "PARA BETA 0.460655337083368"],
    "TFPLUS_DEFAULT": ["KINE TF+"],
    "TFVW_L1_M1": ["KINE TF+", "PARA LAMB 1", "PARA MU 1"],
}


@dataclass(frozen=True)
class CaseInfo:
    case: str
    direction: str
    radius_A: float
    zr: float
    natoms_from_name: int
    vasp_path: Path


def win_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = str(resolved)[3:].replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def discover_cases(source_root: Path, *, sort_by: str, max_atoms: int | None) -> list[CaseInfo]:
    cases: list[CaseInfo] = []
    for vasp_path in sorted(source_root.glob("Al_vac_*/Al_vac_*.vasp")):
        match = CASE_RE.match(vasp_path.stem)
        if not match:
            continue
        natoms = int(match.group("natoms"))
        if max_atoms is not None and natoms > max_atoms:
            continue
        cases.append(
            CaseInfo(
                case=vasp_path.stem,
                direction=match.group("direction"),
                radius_A=float(match.group("radius")),
                zr=float(match.group("zr")),
                natoms_from_name=natoms,
                vasp_path=vasp_path,
            )
        )
    if sort_by == "natoms":
        cases.sort(key=lambda c: (c.natoms_from_name, c.direction, c.radius_A, c.case))
    else:
        cases.sort(key=lambda c: c.case)
    return cases


def vasp_to_profess_ion(vasp_path: Path, ion_path: Path, pseudo_name: str) -> dict[str, float]:
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

    lengths = atoms.cell.lengths()
    angles = atoms.cell.angles()
    return {
        "natoms": len(atoms),
        "volume_A3": float(atoms.get_volume()),
        "a_A": float(lengths[0]),
        "b_A": float(lengths[1]),
        "c_A": float(lengths[2]),
        "alpha_deg": float(angles[0]),
        "beta_deg": float(angles[1]),
        "gamma_deg": float(angles[2]),
    }


def write_inpt(path: Path, *, stem: str, ecut: float, kedf: str) -> None:
    lines = [
        f"ecut {ecut:g}",
        "meth NTN",
        *KEDF_TEMPLATES[kedf],
        "exch lda",
        f"geometryfile {stem}.ion",
        "",
        "print minimizer density 2",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_total_energy(out_path: Path) -> float:
    energy = math.nan
    if not out_path.exists():
        return energy
    for line in out_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "TOTAL ENERGY" in line:
            match = re.search(r"=\s*([-+0-9.Ee]+)\s*eV", line)
            if match:
                energy = float(match.group(1))
    return energy


def run_profess(case_dir: Path, stem: str, timeout_s: int) -> tuple[int, str]:
    # Case directories are nested as outdir/KEDF/case, while PROFESS is copied
    # to outdir/PROFESS.
    cmd = f"cd {win_to_wsl(case_dir)} && ../../PROFESS {stem} > {stem}.out 2> {stem}.err"
    proc = subprocess.run(
        ["wsl", "bash", "-lc", cmd],
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def plot_relative(df: pd.DataFrame, outdir: Path) -> None:
    done = df[df["status"].eq("OK") & df["energy_eV_per_atom"].notna()].copy()
    if done.empty:
        return

    done["relative_eV_atom"] = done.groupby(["kedf", "direction"])["energy_eV_per_atom"].transform(
        lambda s: s - s.min()
    )
    done["relative_meV_atom"] = done["relative_eV_atom"] * 1000.0
    done.to_csv(outdir / "profess_vacancy_radius_relative_energy.csv", index=False)

    colors = {
        "WT": "#305f72",
        "WGC": "#00876c",
        "CAT": "#d98324",
        "HC": "#7a5195",
        "TFPLUS_DEFAULT": "#5b8e7d",
        "TFVW_L1_M1": "#b0413e",
    }
    markers = {
        "WT": "o",
        "WGC": "s",
        "CAT": "^",
        "HC": "D",
        "TFPLUS_DEFAULT": "P",
        "TFVW_L1_M1": "X",
    }

    directions = sorted(done["direction"].unique())
    fig, axes = plt.subplots(1, len(directions), figsize=(5.2 * len(directions), 4.2), sharey=True)
    if len(directions) == 1:
        axes = [axes]

    for ax, direction in zip(axes, directions):
        subdir = done[done["direction"].eq(direction)]
        for kedf in sorted(subdir["kedf"].unique()):
            sub = subdir[subdir["kedf"].eq(kedf)].sort_values("radius_A")
            ax.plot(
                sub["radius_A"],
                sub["relative_meV_atom"],
                marker=markers.get(kedf, "o"),
                linewidth=1.8,
                label=kedf,
                color=colors.get(kedf),
            )
        ax.set_title(f"[{direction}] vacancy structures")
        ax.set_xlabel("Radius (A)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Relative energy (meV/atom)")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Local PROFESS vacancy radius sweep: KEDF dependence", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "profess_vacancy_radius_relative_energy.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    summary = (
        done.groupby("kedf")["energy_eV_per_atom"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .sort_values("mean")
    )
    ax.bar(summary["kedf"], summary["std"].fillna(0) * 1000.0, color="#6f8faf")
    ax.set_ylabel("Std. dev. across vacancy cases (meV/atom)")
    ax.set_title("Energy-per-atom spread across vacancy radius cases")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "profess_vacancy_kedf_spread.png", dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--profess", default=Path(r"C:\Users\dawso\Downloads\PROFESS"), type=Path)
    parser.add_argument(
        "--pseudo",
        default=Path(r"C:\Users\dawso\Downloads\myTest_profess3\myTest\al_HC.lda.recpot"),
        type=Path,
    )
    parser.add_argument("--kedf", action="append", choices=sorted(KEDF_TEMPLATES), help="Repeatable.")
    parser.add_argument("--ecut", type=float, default=1600.0)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--only-prepare", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip case runs that already have a parsed total energy.")
    parser.add_argument(
        "--sort-by",
        choices=("natoms", "name"),
        default="natoms",
        help="Run small cases first by default so slow KEDFs do not block all output.",
    )
    parser.add_argument("--max-atoms", type=int, help="Optional cap for quick diagnostic sweeps.")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    kedfs = args.kedf or ["WT", "WGC", "CAT", "HC", "TFPLUS_DEFAULT", "TFVW_L1_M1"]

    cases = discover_cases(source_root, sort_by=args.sort_by, max_atoms=args.max_atoms)
    if not cases:
        raise SystemExit(f"No Al_vac_*/Al_vac_*.vasp files found under {source_root}")

    shutil.copy2(args.profess, outdir / "PROFESS")
    shutil.copy2(args.pseudo, outdir / args.pseudo.name)

    rows: list[dict[str, object]] = []
    for kedf in kedfs:
        for case in cases:
            case_dir = outdir / kedf / case.case
            case_dir.mkdir(parents=True, exist_ok=True)
            stem = case.case
            vasp_dst = case_dir / f"{stem}.vasp"
            ion_path = case_dir / f"{stem}.ion"
            inpt_path = case_dir / f"{stem}.inpt"
            shutil.copy2(case.vasp_path, vasp_dst)
            shutil.copy2(args.pseudo, case_dir / args.pseudo.name)
            meta = vasp_to_profess_ion(case.vasp_path, ion_path, args.pseudo.name)
            write_inpt(inpt_path, stem=stem, ecut=args.ecut, kedf=kedf)

            rc = None
            status = "PREPARED"
            note = ""
            existing_energy = parse_total_energy(case_dir / f"{stem}.out")
            if args.resume and not math.isnan(existing_energy):
                rc = 0
                status = "OK"
                note = "Reused existing PROFESS output."
            elif not args.only_prepare:
                try:
                    rc, note = run_profess(case_dir, stem, args.timeout_s)
                    status = "OK" if rc == 0 and not math.isnan(parse_total_energy(case_dir / f"{stem}.out")) else "FAILED"
                except subprocess.TimeoutExpired:
                    rc = -1
                    status = "TIMEOUT"
                    note = f"Timed out after {args.timeout_s} s"

            energy = parse_total_energy(case_dir / f"{stem}.out")
            rows.append(
                {
                    "kedf": kedf,
                    "case": case.case,
                    "direction": case.direction,
                    "radius_A": case.radius_A,
                    "zr": case.zr,
                    "natoms_from_name": case.natoms_from_name,
                    **meta,
                    "energy_eV": energy,
                    "energy_eV_per_atom": energy / meta["natoms"] if not math.isnan(energy) else math.nan,
                    "status": status,
                    "returncode": rc,
                    "case_dir": str(case_dir),
                    "source_vasp": str(case.vasp_path),
                    "note": note.strip(),
                }
            )
            print(f"{kedf:16s} {case.case:32s} {status:8s} E={energy:.6f}")
            pd.DataFrame(rows).to_csv(outdir / "profess_vacancy_radius_sweep_results.in_progress.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "profess_vacancy_radius_sweep_results.csv", index=False)
    plot_relative(df, outdir)

    manifest = {
        "source_root": str(source_root),
        "outdir": str(outdir),
        "profess": str(args.profess),
        "pseudo": str(args.pseudo),
        "ecut": args.ecut,
        "kedfs": kedfs,
        "case_count": len(cases),
        "scf_only": True,
        "relaxation": "none; fixed exported VASP structures",
    }
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {outdir / 'profess_vacancy_radius_sweep_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
