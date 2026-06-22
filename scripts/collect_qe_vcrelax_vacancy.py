from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


RY_TO_EV = 13.605693122994
BOHR_TO_ANGSTROM = 0.529177210903
RY_BOHR_TO_EV_ANGSTROM = RY_TO_EV / BOHR_TO_ANGSTROM
ATOM_FORCE_RE = re.compile(
    r"atom\s+\d+\s+type\s+\d+\s+force\s*=\s*"
    r"([0-9.EeDd+-]+)\s+([0-9.EeDd+-]+)\s+([0-9.EeDd+-]+)",
    re.IGNORECASE,
)


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


def parse_final_atomic_fmax(path: Path, nat: int | None) -> tuple[float, float]:
    """Return the maximum atomic-force norm from the final QE force block."""
    if not nat:
        return math.nan, math.nan
    text = read_text(path)
    if "Total force" not in text:
        return math.nan, math.nan
    before_last_total = text.rsplit("Total force", 1)[0]
    matches = list(ATOM_FORCE_RE.finditer(before_last_total))[-nat:]
    if len(matches) != nat:
        return math.nan, math.nan
    norms: list[float] = []
    for match in matches:
        components = [
            float(match.group(index).replace("D", "E").replace("d", "e"))
            for index in range(1, 4)
        ]
        norms.append(math.sqrt(sum(component * component for component in components)))
    fmax_ry_bohr = max(norms)
    return fmax_ry_bohr, fmax_ry_bohr * RY_BOHR_TO_EV_ANGSTROM


def infer_mode(path: Path) -> str:
    parts = set(path.parts)
    if "pair_scan" in parts:
        return "pair_scan"
    if "ecut_scan" in parts:
        return "ecut_scan"
    if "kmesh_scan" in parts:
        return "kmesh_scan"
    return "other"


def read_manifest(path: Path) -> dict[str, object]:
    for name in ("pair_manifest.json", "manifest.json"):
        candidate = path / name
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def collect(rootdir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pristine_out in sorted(rootdir.rglob("pristine_vcrelax/vc-relax.out")):
        base = pristine_out.parent.parent
        defect_dir_name = "divacancy_vcrelax" if (base / "divacancy_vcrelax").exists() else "vacancy_vcrelax"
        vacancy_out = base / defect_dir_name / "vc-relax.out"
        pristine_in = base / "pristine_vcrelax" / "vc-relax.in"
        vacancy_in = base / defect_dir_name / "vc-relax.in"
        if not vacancy_out.exists():
            continue
        manifest = read_manifest(base)

        p_done = job_done(pristine_out)
        v_done = job_done(vacancy_out)
        ep_ry = last_energy_ry(pristine_out)
        ev_ry = last_energy_ry(vacancy_out)
        np_atoms = parse_nat(pristine_in)
        nv_atoms = parse_nat(vacancy_in)
        pristine_fmax_ry_bohr, pristine_fmax_ev_a = parse_final_atomic_fmax(pristine_out, np_atoms)
        vacancy_fmax_ry_bohr, vacancy_fmax_ev_a = parse_final_atomic_fmax(vacancy_out, nv_atoms)
        vacancy_count = (np_atoms - nv_atoms) if np_atoms and nv_atoms else math.nan
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
                "vacancy_count": vacancy_count,
                "defect_label": "divacancy" if defect_dir_name == "divacancy_vcrelax" else "vacancy",
                "vacancy_concentration_percent": (100.0 * vacancy_count / np_atoms) if np_atoms and nv_atoms else math.nan,
                "pair_distance_A": manifest.get("pair_distance_A", math.nan),
                "pristine_done": p_done,
                "vacancy_done": v_done,
                "E_pristine_Ry": ep_ry,
                "E_vacancy_Ry": ev_ry,
                "Ef_vac_eV": ef_ev,
                "pristine_total_force_Ry_bohr": parse_last_total_force(pristine_out),
                "vacancy_total_force_Ry_bohr": parse_last_total_force(vacancy_out),
                "pristine_final_atomic_fmax_Ry_bohr": pristine_fmax_ry_bohr,
                "vacancy_final_atomic_fmax_Ry_bohr": vacancy_fmax_ry_bohr,
                "pristine_final_atomic_fmax_eV_A": pristine_fmax_ev_a,
                "vacancy_final_atomic_fmax_eV_A": vacancy_fmax_ev_a,
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
