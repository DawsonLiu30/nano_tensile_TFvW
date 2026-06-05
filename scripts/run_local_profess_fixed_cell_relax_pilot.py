#!/usr/bin/env python3
"""Run local PROFESS fixed-cell ionic-relax pilot cases.

This is a conservative follow-up to the SCF-only PROFESS radius sweep.  The
vacancy nanostructures contain large x/y vacuum regions, so this script does
not use full cell optimization.  It keeps the simulation cell fixed and relaxes
only ion positions with PROFESS (`MINI ion`, `method ion bfgs`).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from run_local_profess_vacancy_radius_sweep import (
    KEDF_TEMPLATES,
    CaseInfo,
    discover_cases,
    vasp_to_profess_ion,
    win_to_wsl,
)


HA_PER_BOHR_TO_EV_PER_A = 51.4220674763


def write_scf_inpt(path: Path, *, stem: str, ecut: float, kedf: str) -> None:
    lines = [
        f"ecut {ecut:g}",
        "method ntn",
        *KEDF_TEMPLATES[kedf],
        "exch lda",
        f"geometryfile {stem}.ion",
        "",
        "print minimizer density 2",
        "calculate stresses",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ion_relax_inpt(
    path: Path,
    *,
    stem: str,
    ecut: float,
    kedf: str,
    ion_method: str,
    ion_tolf_ha_bohr: float | None,
) -> None:
    lines = [
        f"ecut {ecut:g}",
        "MINI ion",
        "method ntn",
        f"method ion {ion_method}",
        *KEDF_TEMPLATES[kedf],
        "exch lda",
        f"geometryfile {stem}.ion",
        "",
        "print minimizer density 2",
        "print minimizer geom 2",
        "calculate stresses",
        "",
    ]
    if ion_tolf_ha_bohr is not None:
        # PROFESS parses TOLF as the ionic force cutoff.  The output reports
        # maxForce in Ha/bohr, so convert user-facing eV/A before writing.
        lines.insert(4, f"TOLF {ion_tolf_ha_bohr:.12g}")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_total_energies(out_path: Path) -> list[float]:
    energies: list[float] = []
    if not out_path.exists():
        return energies
    for line in out_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "TOTAL ENERGY" not in line:
            continue
        match = re.search(r"=\s*([-+0-9.Ee]+)\s*eV", line)
        if match:
            energies.append(float(match.group(1)))
    return energies


def parse_ion_relax_lines(out_path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    if not out_path.exists():
        return rows
    ion_relax_pattern = re.compile(
        r"\(Ion-Relax\).*?Iter=\s*(?P<step>\d+),\s*"
        r"totEnergy=\s*(?P<energy>[-+0-9.Ee]+)\s*\(Ha\),\s*"
        r"maxForce=\s*(?P<force>[-+0-9.Ee]+)"
    )
    optimizer_pattern = re.compile(
        r"\((?P<method>[^)]+)\):.*?(?:iter|Iter):?\s*(?P<step>\d+).*?"
        r"totEnergy=\s*(?P<energy>[-+0-9.Ee]+)\s*\(Ha\).*?"
        r"maxForce=\s*(?P<force>[-+0-9.Ee]+)\s*Ha/bohr",
        re.IGNORECASE,
    )
    for line in out_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = ion_relax_pattern.search(line) or optimizer_pattern.search(line)
        if not match:
            continue
        force_ha_bohr = float(match.group("force"))
        rows.append(
            {
                "ion_step": int(match.group("step")),
                "ion_energy_Ha": float(match.group("energy")),
                "ion_max_force_Ha_bohr": force_ha_bohr,
                "ion_max_force_eV_A": force_ha_bohr * HA_PER_BOHR_TO_EV_PER_A,
            }
        )
    return rows


def parse_requested_tolf_from_input(out_path: Path) -> dict[str, float]:
    """Parse the requested ionic TOLF from the sibling PROFESS input file."""
    result = {
        "input_tolf_Ha_bohr": math.nan,
        "input_tolf_eV_A": math.nan,
    }
    inpt_path = out_path.with_suffix(".inpt")
    if not inpt_path.exists():
        return result
    pattern = re.compile(r"^\s*TOLF\s+(?P<tolf>[-+0-9.Ee]+)\s*$", re.IGNORECASE)
    for line in inpt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        tolf = float(match.group("tolf"))
        result["input_tolf_Ha_bohr"] = tolf
        result["input_tolf_eV_A"] = tolf * HA_PER_BOHR_TO_EV_PER_A
    return result


def parse_profess_success_force(out_path: Path) -> dict[str, float | bool]:
    """Parse PROFESS' official ion-relax convergence line, if present."""
    result: dict[str, float | bool] = {
        "profess_ion_relax_success": False,
        "profess_success_force_Ha_bohr": math.nan,
        "profess_success_force_eV_A": math.nan,
        "profess_tolf_Ha_bohr": math.nan,
        "profess_tolf_eV_A": math.nan,
    }
    if not out_path.exists():
        return result
    pattern = re.compile(
        r"\(Ion-Relax\):\s*Max Force=\s*(?P<force>[-+0-9.Ee]+)\s*<\s*"
        r"(?P<tolf>[-+0-9.Ee]+).*?ion-relax is successful",
        re.IGNORECASE,
    )
    for line in out_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        force = float(match.group("force"))
        tolf = float(match.group("tolf"))
        result.update(
            {
                "profess_ion_relax_success": True,
                "profess_success_force_Ha_bohr": force,
                "profess_success_force_eV_A": force * HA_PER_BOHR_TO_EV_PER_A,
                "profess_tolf_Ha_bohr": tolf,
                "profess_tolf_eV_A": tolf * HA_PER_BOHR_TO_EV_PER_A,
            }
        )
    return result


def has_profess_end(out_path: Path) -> bool:
    if not out_path.exists():
        return False
    text = out_path.read_text(encoding="utf-8", errors="ignore")
    return "END OF PROFESS" in text or "Total Run Time" in text


def run_profess(case_dir: Path, stem: str, timeout_s: int) -> tuple[int, str]:
    cmd = f"cd {win_to_wsl(case_dir)} && ../../PROFESS {stem} > {stem}.out 2> {stem}.err"
    proc = subprocess.Popen(
        ["wsl", "bash", "-lc", cmd],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        return int(proc.returncode), (stdout or "") + (stderr or "")
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # Killing the WSL shell is not always enough; stop the nested PROFESS.
        kill_pattern = f"PROFESS {stem}"
        subprocess.run(
            ["wsl", "bash", "-lc", f"pkill -f {kill_pattern!r} || true"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return 124, "TIMEOUT"


def choose_pilot_cases(
    cases: list[CaseInfo],
    *,
    explicit_cases: set[str],
    limit_per_direction: int,
) -> list[CaseInfo]:
    if explicit_cases:
        return [case for case in cases if case.case in explicit_cases]

    selected: list[CaseInfo] = []
    for direction in sorted({case.direction for case in cases}):
        sub = [case for case in cases if case.direction == direction]
        sub.sort(key=lambda case: (case.natoms_from_name, case.radius_A, case.case))
        selected.extend(sub[:limit_per_direction])
    selected.sort(key=lambda case: (case.natoms_from_name, case.direction, case.radius_A, case.case))
    return selected


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_row(
    *,
    kedf: str,
    case: CaseInfo,
    meta: dict[str, float],
    start_rc: int,
    relax_rc: int,
    start_out: Path,
    relax_out: Path,
    note: str,
) -> dict[str, object]:
    start_energies = parse_total_energies(start_out)
    relax_energies = parse_total_energies(relax_out)
    ion_rows = parse_ion_relax_lines(relax_out)
    profess_success = parse_profess_success_force(relax_out)
    input_tolf = parse_requested_tolf_from_input(relax_out)
    final_ion = ion_rows[-1] if ion_rows else {}

    start_energy = start_energies[-1] if start_energies else math.nan
    relax_initial_energy = relax_energies[0] if relax_energies else math.nan
    relax_final_energy = relax_energies[-1] if relax_energies else math.nan
    delta_vs_start = (
        relax_final_energy - start_energy
        if not math.isnan(start_energy) and not math.isnan(relax_final_energy)
        else math.nan
    )

    if relax_rc == 124:
        status = "TIMEOUT"
    elif start_rc != 0 or relax_rc != 0:
        status = "FAILED"
    elif math.isnan(start_energy) or math.isnan(relax_final_energy):
        status = "NO_TOTAL_ENERGY"
    elif not has_profess_end(relax_out):
        status = "PARTIAL_NO_END_MARKER"
    else:
        status = "OK"

    final_force_ha_bohr = final_ion.get("ion_max_force_Ha_bohr", math.nan)
    final_force_eV_A = final_ion.get("ion_max_force_eV_A", math.nan)
    input_tolf_ha_bohr = input_tolf["input_tolf_Ha_bohr"]
    input_tolf_eV_A = input_tolf["input_tolf_eV_A"]
    force_meets_tolf = (
        bool(final_force_ha_bohr <= input_tolf_ha_bohr)
        if not math.isnan(final_force_ha_bohr) and not math.isnan(input_tolf_ha_bohr)
        else False
    )
    ion_relax_valid_by_force = bool(profess_success["profess_ion_relax_success"]) or force_meets_tolf

    return {
        "kedf": kedf,
        "case": case.case,
        "direction": case.direction,
        "radius_A": case.radius_A,
        "zr": case.zr,
        "natoms_from_name": case.natoms_from_name,
        **meta,
        "start_scf_returncode": start_rc,
        "relax_returncode": relax_rc,
        "start_scf_energy_eV": start_energy,
        "relax_initial_energy_eV": relax_initial_energy,
        "relax_final_energy_eV": relax_final_energy,
        "relax_delta_vs_start_scf_eV": delta_vs_start,
        "relax_lower_than_start_scf": bool(delta_vs_start <= 0.0) if not math.isnan(delta_vs_start) else False,
        "ion_steps_observed": len(ion_rows),
        "final_ion_step": final_ion.get("ion_step", math.nan),
        "final_ion_max_force_Ha_bohr": final_force_ha_bohr,
        "final_ion_max_force_eV_A": final_force_eV_A,
        **input_tolf,
        "ion_force_meets_tolf_numeric": force_meets_tolf,
        "ion_relax_valid_by_force": ion_relax_valid_by_force,
        **profess_success,
        "has_relax_end_marker": has_profess_end(relax_out),
        "status": status,
        "note": note,
        "case_dir": str(relax_out.parent),
        "source_vasp": str(case.vasp_path),
        "start_scf_out": str(start_out),
        "relax_out": str(relax_out),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default=Path(
            r"C:\Users\dawso\Desktop\NUS_upload\2026-04-21_full_sync\qe_runs\vacancy_nanocrystal_relax"
        ),
        type=Path,
    )
    parser.add_argument(
        "--outdir",
        default=Path(r"C:\Users\dawso\Desktop\LOCAL_PROFESS_FIXED_CELL_RELAX_PILOT_20260604"),
        type=Path,
    )
    parser.add_argument("--profess", default=Path(r"C:\Users\dawso\Downloads\PROFESS"), type=Path)
    parser.add_argument(
        "--pseudo",
        default=Path(r"C:\Users\dawso\Downloads\myTest_profess3\myTest\al_HC.lda.recpot"),
        type=Path,
    )
    parser.add_argument("--kedf", action="append", choices=sorted(KEDF_TEMPLATES))
    parser.add_argument("--ecut", type=float, default=1600.0)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument(
        "--ion-method",
        default="bfgs",
        help="PROFESS ion relaxation method, e.g. bfgs, cg, cg2, qui.",
    )
    parser.add_argument(
        "--ion-tolf-ev-a",
        type=float,
        default=0.002,
        help="Ionic force cutoff requested in eV/A. Written to PROFESS TOLF after conversion to Ha/bohr.",
    )
    parser.add_argument("--max-atoms", type=int, default=70)
    parser.add_argument("--limit-per-direction", type=int, default=2)
    parser.add_argument("--case", action="append", help="Exact case name. Repeatable.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-prepare", action="store_true")
    args = parser.parse_args()

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.profess, outdir / "PROFESS")
    shutil.copy2(args.pseudo, outdir / args.pseudo.name)

    kedfs = args.kedf or ["TFPLUS_DEFAULT"]
    ion_tolf_ha_bohr = (
        float(args.ion_tolf_ev_a) / HA_PER_BOHR_TO_EV_PER_A
        if args.ion_tolf_ev_a and args.ion_tolf_ev_a > 0
        else None
    )
    all_cases = discover_cases(args.source_root, sort_by="natoms", max_atoms=args.max_atoms)
    cases = choose_pilot_cases(
        all_cases,
        explicit_cases=set(args.case or []),
        limit_per_direction=max(1, int(args.limit_per_direction)),
    )
    if not cases:
        raise SystemExit("No cases selected. Check --source-root, --max-atoms, or --case.")

    rows: list[dict[str, object]] = []
    summary_path = outdir / "profess_fixed_cell_relax_pilot_summary.csv"
    for kedf in kedfs:
        for case in cases:
            case_dir = outdir / kedf / case.case
            case_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(case.vasp_path, case_dir / f"{case.case}.vasp")
            shutil.copy2(args.pseudo, case_dir / args.pseudo.name)
            meta = vasp_to_profess_ion(case.vasp_path, case_dir / f"{case.case}.ion", args.pseudo.name)
            write_scf_inpt(case_dir / f"{case.case}_start_scf.inpt", stem=case.case, ecut=args.ecut, kedf=kedf)
            write_ion_relax_inpt(
                case_dir / f"{case.case}_ion_relax.inpt",
                stem=case.case,
                ecut=args.ecut,
                kedf=kedf,
                ion_method=str(args.ion_method),
                ion_tolf_ha_bohr=ion_tolf_ha_bohr,
            )

            start_out = case_dir / f"{case.case}_start_scf.out"
            relax_out = case_dir / f"{case.case}_ion_relax.out"
            if args.only_prepare:
                row = {
                    "kedf": kedf,
                    "case": case.case,
                    **asdict(case),
                    **meta,
                    "status": "PREPARED_ONLY",
                    "case_dir": str(case_dir),
                }
                rows.append(row)
                write_rows(summary_path, rows)
                continue

            start_rc = 0
            relax_rc = 0
            note = ""
            if not (args.resume and start_out.exists() and parse_total_energies(start_out)):
                start_rc, note_start = run_profess(case_dir, f"{case.case}_start_scf", args.timeout_s)
                note += f"start_scf: {note_start}".strip()
            else:
                note += "start_scf: reused existing output"

            if not (args.resume and relax_out.exists() and parse_total_energies(relax_out)):
                relax_rc, note_relax = run_profess(case_dir, f"{case.case}_ion_relax", args.timeout_s)
                note += " | " + f"ion_relax: {note_relax}".strip()
            else:
                note += " | ion_relax: reused existing output"

            row = summarize_row(
                kedf=kedf,
                case=case,
                meta=meta,
                start_rc=start_rc,
                relax_rc=relax_rc,
                start_out=start_out,
                relax_out=relax_out,
                note=note,
            )
            rows.append(row)
            write_rows(summary_path, rows)
            print(
                f"{kedf:16s} {case.case:36s} {row['status']:18s} "
                f"dE={row['relax_delta_vs_start_scf_eV']} "
                f"F={row['final_ion_max_force_eV_A']}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(summary_path, index=False)
    md_lines = [
        "# Local PROFESS Fixed-Cell Ionic Relax Pilot",
        "",
        "Mode: `MINI ion`, `method ion bfgs`; simulation cell is fixed.",
        "",
        "This is intentionally not full cell relaxation because the vacancy",
        "nanostructure cells contain x/y vacuum regions.",
        "",
        "## Completion",
        "",
        df.groupby("kedf")["status"].value_counts().to_string() if not df.empty else "No rows.",
        "",
    ]
    (outdir / "README_fixed_cell_relax_pilot.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
