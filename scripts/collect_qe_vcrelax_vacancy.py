from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


RY_TO_EV = 13.605693122994


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""


def job_done(path: Path) -> bool:
    return "JOB DONE" in read_text(path)


def last_energy_ry(path: Path) -> float:
    values: list[float] = []
    for line in read_text(path).splitlines():
        if line.strip().startswith("!"):
            values.append(float(line.split("=")[1].split()[0]))
    return values[-1] if values else math.nan


def parse_nat(path: Path) -> int | None:
    match = re.search(r"\bnat\s*=\s*(\d+)", read_text(path), re.IGNORECASE)
    return int(match.group(1)) if match else None


def parse_ecut_ev(path: Path) -> float:
    match = re.search(r"\becutwfc\s*=\s*([0-9.EeDd+-]+)", read_text(path), re.IGNORECASE)
    if not match:
        return math.nan
    return float(match.group(1).replace("D", "E").replace("d", "e")) * RY_TO_EV


def parse_kmesh(path: Path) -> str:
    lines = read_text(path).splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("k_points") and i + 1 < len(lines):
            nums = lines[i + 1].split()
            if len(nums) >= 3:
                return f"{nums[0]}x{nums[1]}x{nums[2]}"
    return "unknown"


def parse_last_total_force(path: Path) -> float:
    values: list[float] = []
    for line in read_text(path).splitlines():
        match = re.search(r"Total force\s*=\s*([0-9.EeDd+-]+)", line)
        if match:
            values.append(float(match.group(1).replace("D", "E").replace("d", "e")))
    return values[-1] if values else math.nan


def infer_mode(path: Path) -> str:
    parts = set(path.parts)
    if "ecut_scan" in parts:
        return "ecut_scan"
    if "kmesh_scan" in parts:
        return "kmesh_scan"
    return "other"


def collect(rootdir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pristine_out in sorted(rootdir.rglob("pristine_vcrelax/vc-relax.out")):
        base = pristine_out.parent.parent
        vacancy_out = base / "vacancy_vcrelax" / "vc-relax.out"
        pristine_in = base / "pristine_vcrelax" / "vc-relax.in"
        vacancy_in = base / "vacancy_vcrelax" / "vc-relax.in"
        if not vacancy_out.exists():
            continue

        p_done = job_done(pristine_out)
        v_done = job_done(vacancy_out)
        ep_ry = last_energy_ry(pristine_out)
        ev_ry = last_energy_ry(vacancy_out)
        np_atoms = parse_nat(pristine_in)
        nv_atoms = parse_nat(vacancy_in)
        if p_done and v_done and np_atoms and nv_atoms and not math.isnan(ep_ry) and not math.isnan(ev_ry):
            ef_ev = (ev_ry - (nv_atoms / np_atoms) * ep_ry) * RY_TO_EV
        else:
            ef_ev = math.nan

        rows.append(
            {
                "path": str(base.relative_to(rootdir)),
                "mode": infer_mode(base),
                "ecut_eV": parse_ecut_ev(pristine_in),
                "kmesh": parse_kmesh(pristine_in),
                "N_pristine": np_atoms,
                "N_vacancy": nv_atoms,
                "vacancy_concentration_percent": (100.0 / np_atoms) if np_atoms else math.nan,
                "pristine_done": p_done,
                "vacancy_done": v_done,
                "E_pristine_Ry": ep_ry,
                "E_vacancy_Ry": ev_ry,
                "Ef_vac_eV": ef_ev,
                "pristine_total_force_Ry_bohr": parse_last_total_force(pristine_out),
                "vacancy_total_force_Ry_bohr": parse_last_total_force(vacancy_out),
            }
        )
    return sorted(rows, key=lambda r: (str(r["mode"]), str(r["kmesh"]), float(r["ecut_eV"]), str(r["path"])))


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect QE vc-relax vacancy formation energies.")
    ap.add_argument("--rootdir", default=".")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rootdir = Path(args.rootdir).expanduser().resolve()
    rows = collect(rootdir)
    out = Path(args.out).expanduser().resolve() if args.out else rootdir / "processed_vcrelax" / "qe_vcrelax_vacancy_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("============================================================")
    print("QE vc-relax vacancy collection completed")
    print("============================================================")
    print(f"Output: {out}")
    for row in rows:
        print(
            f"{row['mode']:10s} {row['kmesh']:8s} "
            f"ecut={float(row['ecut_eV']):8.1f} eV  "
            f"Ef={float(row['Ef_vac_eV']):10.6f} eV  "
            f"P={row['pristine_done']} V={row['vacancy_done']}  {row['path']}"
        )


if __name__ == "__main__":
    main()
