#!/usr/bin/env python3
"""Collect PROFESS periodic vacancy fixed-cell ion-relax outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HA_PER_BOHR_TO_EV_PER_A = 51.4220674763


TOTAL_ENERGY_RE = re.compile(r"TOTAL ENERGY\s*=\s*([-+0-9.Ee]+)\s*eV")
ION_RE = re.compile(
    r"\(Ion-Relax\).*?Iter=\s*(?P<step>\d+),\s*"
    r"totEnergy=\s*(?P<energy>[-+0-9.Ee]+)\s*\(Ha\),\s*"
    r"maxForce=\s*(?P<force>[-+0-9.Ee]+)"
)
SUCCESS_RE = re.compile(
    r"\(Ion-Relax\):\s*Max Force=\s*(?P<force>[-+0-9.Ee]+)\s*<\s*"
    r"(?P<tolf>[-+0-9.Ee]+).*?ion-relax is successful",
    re.IGNORECASE,
)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def has_end_marker(path: Path) -> bool:
    text = read_text(path)
    return "END OF PROFESS" in text or "Total Run Time" in text


def parse_total_energies(path: Path) -> list[float]:
    return [float(m.group(1)) for m in TOTAL_ENERGY_RE.finditer(read_text(path))]


def parse_ion_rows(path: Path) -> list[dict[str, float | int]]:
    rows = []
    for m in ION_RE.finditer(read_text(path)):
        force = float(m.group("force"))
        rows.append(
            {
                "step": int(m.group("step")),
                "energy_Ha": float(m.group("energy")),
                "force_Ha_bohr": force,
                "force_eV_A": force * HA_PER_BOHR_TO_EV_PER_A,
            }
        )
    return rows


def parse_success(path: Path) -> dict[str, object]:
    text = read_text(path)
    for m in SUCCESS_RE.finditer(text):
        force = float(m.group("force"))
        tolf = float(m.group("tolf"))
        return {
            "profess_success": True,
            "success_force_Ha_bohr": force,
            "success_force_eV_A": force * HA_PER_BOHR_TO_EV_PER_A,
            "success_tolf_Ha_bohr": tolf,
            "success_tolf_eV_A": tolf * HA_PER_BOHR_TO_EV_PER_A,
        }
    return {
        "profess_success": False,
        "success_force_Ha_bohr": math.nan,
        "success_force_eV_A": math.nan,
        "success_tolf_Ha_bohr": math.nan,
        "success_tolf_eV_A": math.nan,
    }


def summarize_out(path: Path) -> dict[str, object]:
    energies = parse_total_energies(path)
    ion_rows = parse_ion_rows(path)
    final_ion = ion_rows[-1] if ion_rows else {}
    success = parse_success(path)
    return {
        "out_path": str(path),
        "has_end_marker": has_end_marker(path),
        "n_total_energy_lines": len(energies),
        "initial_total_energy_eV": energies[0] if energies else math.nan,
        "final_total_energy_eV": energies[-1] if energies else math.nan,
        "ion_steps_observed": len(ion_rows),
        "final_ion_step": final_ion.get("step", math.nan),
        "final_force_eV_A": final_ion.get("force_eV_A", math.nan),
        **success,
    }


def status_from(pristine: dict[str, object], vacancy: dict[str, object], target: float) -> str:
    if math.isnan(float(pristine["final_total_energy_eV"])) or math.isnan(float(vacancy["final_total_energy_eV"])):
        return "NO_ENERGY"
    p_force = float(pristine["final_force_eV_A"]) if pristine["final_force_eV_A"] == pristine["final_force_eV_A"] else math.nan
    v_force = float(vacancy["final_force_eV_A"]) if vacancy["final_force_eV_A"] == vacancy["final_force_eV_A"] else math.nan
    p_ok = bool(pristine["profess_success"]) or (not math.isnan(p_force) and p_force <= target)
    v_ok = bool(vacancy["profess_success"]) or (not math.isnan(v_force) and v_force <= target)
    if p_ok and v_ok:
        return "OK_FORCE"
    if bool(pristine["has_end_marker"]) and bool(vacancy["has_end_marker"]):
        return "ENDED_FORCE_NOT_MET"
    return "PARTIAL"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rootdir", required=True, type=Path)
    ap.add_argument("--target-force", type=float, default=0.002)
    args = ap.parse_args()

    root = args.rootdir.resolve()
    manifests = sorted(root.glob("*/*/manifest.json"))
    rows: list[dict[str, object]] = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        case_dir = manifest_path.parent
        pristine = summarize_out(case_dir / "pristine_relax.out")
        vacancy = summarize_out(case_dir / "vacancy_relax.out")
        n_pristine = int(manifest["N_pristine"])
        n_vacancy = int(manifest["N_vacancy"])
        p_energy = float(pristine["final_total_energy_eV"])
        v_energy = float(vacancy["final_total_energy_eV"])
        ef = (
            v_energy - (n_vacancy / n_pristine) * p_energy
            if not math.isnan(p_energy) and not math.isnan(v_energy)
            else math.nan
        )
        rows.append(
            {
                "case": manifest["case"],
                "kedf": manifest["kedf"],
                "family": manifest["family"],
                "shape": manifest["shape"],
                "orientation": manifest["orientation"],
                "diameter_nm": manifest["diameter_nm"],
                "N_pristine": n_pristine,
                "N_vacancy": n_vacancy,
                "vacancy_concentration_percent": manifest["vacancy_concentration_percent"],
                "vacancy_position": manifest["vacancy_position"],
                "vacancy_radial_fraction": manifest["vacancy_radial_fraction"],
                "ecut_eV": manifest["ecut_eV"],
                "ion_tolf_eV_A": manifest["ion_tolf_eV_A"],
                "E_pristine_relax_eV": p_energy,
                "E_vacancy_relax_eV": v_energy,
                "Ef_vac_eV": ef,
                "pristine_final_force_eV_A": pristine["final_force_eV_A"],
                "vacancy_final_force_eV_A": vacancy["final_force_eV_A"],
                "pristine_end_marker": pristine["has_end_marker"],
                "vacancy_end_marker": vacancy["has_end_marker"],
                "pristine_profess_success": pristine["profess_success"],
                "vacancy_profess_success": vacancy["profess_success"],
                "status": status_from(pristine, vacancy, float(args.target_force)),
                "case_dir": str(case_dir),
            }
        )

    out_csv = root / "profess_periodic_vacancy_relax_summary.csv"
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        out_csv.write_text("", encoding="utf-8")

    df = pd.DataFrame(rows)
    if not df.empty:
        completion = (
            df.groupby(["kedf", "family", "status"])
            .size()
            .reset_index(name="count")
            .sort_values(["kedf", "family", "status"])
        )
        completion.to_csv(root / "profess_periodic_vacancy_relax_completion.csv", index=False)
        ok = df[df["status"].eq("OK_FORCE") & df["Ef_vac_eV"].notna()].copy()
        if not ok.empty:
            fig, axes = plt.subplots(1, len(sorted(ok["family"].unique())), figsize=(6.0 * len(sorted(ok["family"].unique())), 4.2), squeeze=False)
            for ax, fam in zip(axes.ravel(), sorted(ok["family"].unique())):
                subfam = ok[ok["family"].eq(fam)]
                for (kedf, pos), sub in subfam.groupby(["kedf", "vacancy_position"]):
                    sub = sub.sort_values("diameter_nm")
                    ax.plot(sub["diameter_nm"], sub["Ef_vac_eV"], marker="o", label=f"{kedf} {pos}")
                ax.set_title(fam)
                ax.set_xlabel("Diameter (nm)")
                ax.set_ylabel("Vacancy formation energy (eV)")
                ax.grid(True, alpha=0.25)
                ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(root / "profess_periodic_vacancy_relax_Ef.png", dpi=250)
            plt.close(fig)

    print("============================================================")
    print("PROFESS periodic vacancy relax collection completed")
    print("============================================================")
    print(f"Root : {root}")
    print(f"Rows : {len(rows)}")
    print(f"CSV  : {out_csv}")
    if rows:
        print(df.groupby(["kedf", "status"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
