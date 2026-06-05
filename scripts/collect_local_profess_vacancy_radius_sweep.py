#!/usr/bin/env python3
"""Collect local PROFESS vacancy radius sweep outputs into clean tables/plots."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ase.io import read


CASE_RE = re.compile(
    r"Al_vac_(?P<direction>\d+)_r(?P<radius>[\d.]+)_zr(?P<zr>[\d.]+)_n(?P<natoms>\d+)"
)


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


def has_profess_end(out_path: Path) -> bool:
    if not out_path.exists():
        return False
    text = out_path.read_text(encoding="utf-8", errors="ignore")
    return "END OF PROFESS" in text or "Total Run Time" in text


def read_structure_meta(vasp_path: Path) -> dict[str, float | int]:
    atoms = read(str(vasp_path), format="vasp")
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


def collect(outdir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for kedf_dir in sorted(p for p in outdir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        kedf = kedf_dir.name
        for case_dir in sorted(p for p in kedf_dir.iterdir() if p.is_dir()):
            match = CASE_RE.match(case_dir.name)
            if not match:
                continue
            stem = case_dir.name
            vasp_path = case_dir / f"{stem}.vasp"
            out_path = case_dir / f"{stem}.out"
            err_path = case_dir / f"{stem}.err"
            inpt_path = case_dir / f"{stem}.inpt"
            energy = parse_total_energy(out_path)
            meta: dict[str, float | int] = {}
            if vasp_path.exists():
                meta = read_structure_meta(vasp_path)
            status = "OK" if not math.isnan(energy) else "NO_TOTAL_ENERGY"
            if err_path.exists() and err_path.stat().st_size:
                status = "ERR_WITH_TOTAL_ENERGY" if status == "OK" else "ERR_NO_TOTAL_ENERGY"
            rows.append(
                {
                    "kedf": kedf,
                    "case": stem,
                    "direction": match.group("direction"),
                    "radius_A": float(match.group("radius")),
                    "zr": float(match.group("zr")),
                    "natoms_from_name": int(match.group("natoms")),
                    **meta,
                    "energy_eV": energy,
                    "energy_eV_per_atom": energy / meta["natoms"]
                    if meta and not math.isnan(energy)
                    else math.nan,
                    "status": status,
                    "has_profess_end": has_profess_end(out_path),
                    "inpt_path": str(inpt_path),
                    "out_path": str(out_path),
                    "err_path": str(err_path),
                    "vasp_path": str(vasp_path),
                }
            )
    return pd.DataFrame(rows)


def write_outputs(df: pd.DataFrame, analysis_dir: Path) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    all_csv = analysis_dir / "profess_vacancy_radius_all_cases.csv"
    df.to_csv(all_csv, index=False)

    ok = df[df["status"].eq("OK") & df["energy_eV_per_atom"].notna()].copy()
    if not ok.empty:
        ok["relative_eV_atom"] = ok.groupby(["kedf", "direction"])["energy_eV_per_atom"].transform(
            lambda s: s - s.min()
        )
        ok["relative_meV_atom"] = ok["relative_eV_atom"] * 1000.0
        ok.sort_values(["kedf", "direction", "radius_A"]).to_csv(
            analysis_dir / "profess_vacancy_radius_relative_energy.csv", index=False
        )

        spread = (
            ok.groupby("kedf")["energy_eV_per_atom"]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
            .sort_values("kedf")
        )
        spread.to_csv(analysis_dir / "profess_vacancy_kedf_spread_summary.csv", index=False)
        plot_relative(ok, analysis_dir)

    completion = (
        df.assign(is_ok=df["status"].eq("OK"))
        .groupby("kedf")
        .agg(total_cases=("case", "count"), completed_cases=("is_ok", "sum"))
        .reset_index()
    )
    completion["completion_fraction"] = completion["completed_cases"] / completion["total_cases"]
    completion.sort_values("kedf").to_csv(
        analysis_dir / "profess_vacancy_kedf_completion_summary.csv", index=False
    )


def plot_relative(ok: pd.DataFrame, analysis_dir: Path) -> None:
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
    directions = sorted(ok["direction"].unique())
    fig, axes = plt.subplots(1, len(directions), figsize=(5.2 * len(directions), 4.2), sharey=True)
    if len(directions) == 1:
        axes = [axes]
    for ax, direction in zip(axes, directions):
        subdir = ok[ok["direction"].eq(direction)]
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
        ax.set_title(f"[{direction}]")
        ax.set_xlabel("Radius (A)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Relative energy (meV/atom)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Local PROFESS vacancy radius sweep: completed KEDF cases", fontsize=14)
    fig.tight_layout()
    fig.savefig(analysis_dir / "profess_vacancy_radius_relative_energy.png", dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        default=Path(r"C:\Users\dawso\Desktop\LOCAL_PROFESS_VACANCY_RADIUS_SWEEP_20260604"),
        type=Path,
    )
    parser.add_argument("--analysis-subdir", default="analysis", type=str)
    args = parser.parse_args()

    outdir = args.outdir.resolve()
    df = collect(outdir)
    if df.empty:
        raise SystemExit(f"No PROFESS case outputs found under {outdir}")
    analysis_dir = outdir / args.analysis_subdir
    write_outputs(df, analysis_dir)
    completion = analysis_dir / "profess_vacancy_kedf_completion_summary.csv"
    print(f"Wrote: {analysis_dir}")
    print(pd.read_csv(completion).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
