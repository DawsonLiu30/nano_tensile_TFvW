from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


RY_TO_EV = 13.605693122994


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Collect all QE vacancy convergence cases recursively into one CSV.")
    ap.add_argument("--rootdir", default=".", help="QE vacancy workflow root directory")
    ap.add_argument("--outdir", default="", help="Output directory. Defaults to <rootdir>/processed_vacancy_convergence")
    return ap.parse_args()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _job_done(path: Path) -> bool:
    return "JOB DONE" in _read_text(path)


def _all_energies_ry(path: Path) -> list[float]:
    vals: list[float] = []
    for line in _read_text(path).splitlines():
        if line.strip().startswith("!"):
            try:
                vals.append(float(line.split("=")[1].split()[0]))
            except Exception:
                pass
    return vals


def _first_energy_ry(path: Path) -> float:
    vals = _all_energies_ry(path)
    return vals[0] if vals else math.nan


def _last_energy_ry(path: Path) -> float:
    vals = _all_energies_ry(path)
    return vals[-1] if vals else math.nan


def _parse_nat(path: Path) -> int | None:
    m = re.search(r"nat\s*=\s*(\d+)", _read_text(path))
    return int(m.group(1)) if m else None


def _parse_ecut_ev(path: Path) -> float:
    m = re.search(r"ecutwfc\s*=\s*([0-9.EeDd+-]+)", _read_text(path))
    if not m:
        return math.nan
    return float(m.group(1).replace("D", "E").replace("d", "e")) * RY_TO_EV


def _parse_kmesh(path: Path) -> str:
    lines = _read_text(path).splitlines()
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith("k_points") and idx + 1 < len(lines):
            nums = lines[idx + 1].split()
            if len(nums) >= 3:
                try:
                    return f"{int(nums[0])}x{int(nums[1])}x{int(nums[2])}"
                except Exception:
                    return f"{nums[0]}x{nums[1]}x{nums[2]}"
    return "unknown"


def _infer_mode(base: Path) -> str:
    parts = set(base.parts)
    if "ecut_scan" in parts:
        return "ecut_scan"
    if "kmesh_scan" in parts:
        return "kmesh_scan"
    return "other"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rootdir = Path(args.rootdir).expanduser().resolve()
    outdir = (
        Path(args.outdir).expanduser().resolve()
        if str(args.outdir).strip()
        else (rootdir / "processed_vacancy_convergence").resolve()
    )
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for pristine_out in sorted(rootdir.rglob("pristine_scf/scf.out")):
        base = pristine_out.parent.parent
        vacancy_out = base / "vacancy_relax" / "relax.out"
        pristine_in = base / "pristine_scf" / "scf.in"
        vacancy_in = base / "vacancy_relax" / "relax.in"
        if not vacancy_out.exists() or not pristine_in.exists() or not vacancy_in.exists():
            continue

        pristine_done = _job_done(pristine_out)
        vacancy_done = _job_done(vacancy_out)
        e_pristine_ry = _first_energy_ry(pristine_out)
        e_vacancy_ry = _last_energy_ry(vacancy_out)
        n_pristine = _parse_nat(pristine_in)
        n_vacancy = _parse_nat(vacancy_in)

        ef_vac = math.nan
        if (
            pristine_done
            and vacancy_done
            and n_pristine
            and n_vacancy
            and not math.isnan(e_pristine_ry)
            and not math.isnan(e_vacancy_ry)
        ):
            ef_vac = (e_vacancy_ry - (n_vacancy / n_pristine) * e_pristine_ry) * RY_TO_EV

        rows.append(
            {
                "path": str(base.relative_to(rootdir)),
                "mode": _infer_mode(base),
                "ecut_eV": _parse_ecut_ev(pristine_in),
                "kmesh": _parse_kmesh(pristine_in),
                "N_pristine": n_pristine,
                "N_vacancy": n_vacancy,
                "pristine_done": pristine_done,
                "vacancy_done": vacancy_done,
                "E_pristine_Ry": e_pristine_ry,
                "E_vacancy_Ry": e_vacancy_ry,
                "Ef_vac_eV": ef_vac,
            }
        )

    rows.sort(
        key=lambda r: (
            str(r["mode"]),
            str(r["kmesh"]),
            float(r["ecut_eV"]) if not math.isnan(float(r["ecut_eV"])) else 0.0,
            str(r["path"]),
        )
    )

    outcsv = outdir / "qe_vacancy_all_recursive_summary.csv"
    _write_csv(outcsv, rows)

    print("============================================================")
    print("QE vacancy recursive collection completed")
    print("============================================================")
    print(f"Output: {outcsv}")
    print()
    for row in rows:
        ef_text = f"{float(row['Ef_vac_eV']):10.6f}" if not math.isnan(float(row["Ef_vac_eV"])) else "       nan"
        ecut = float(row["ecut_eV"])
        ecut_text = f"{ecut:8.1f}" if not math.isnan(ecut) else "     nan"
        print(
            f"{str(row['mode']):10s} {str(row['kmesh']):8s} "
            f"ecut={ecut_text} eV  Ef={ef_text} eV  "
            f"P={row['pristine_done']} V={row['vacancy_done']}  {row['path']}"
        )


if __name__ == "__main__":
    main()
