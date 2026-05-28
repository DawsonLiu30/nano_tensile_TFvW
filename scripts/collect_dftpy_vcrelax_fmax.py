from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_ase_fmax(log_path: Path) -> tuple[int | None, float]:
    final_step: int | None = None
    final_fmax = math.nan
    if not log_path.exists():
        return final_step, final_fmax
    for line in log_path.read_text(errors="ignore").splitlines():
        if line.strip().startswith("BFGS:"):
            parts = line.split()
            try:
                final_step = int(parts[1])
                final_fmax = float(parts[-1])
            except Exception:
                pass
    return final_step, final_fmax


def manifest_or_empty(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def result_or_empty(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect pristine/vacancy fmax for DFTpy full-cell-relax vacancy cases.")
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", default="")
    ap.add_argument("--target-fmax", type=float, default=0.002)
    args = ap.parse_args()

    rows: list[dict[str, object]] = []
    for root in [Path(x).expanduser().resolve() for x in args.roots]:
        for manifest_path in sorted(root.rglob("point_manifest.json")):
            case_dir = manifest_path.parent
            manifest = manifest_or_empty(manifest_path)
            result = result_or_empty(case_dir / "result.json")
            p_step, p_fmax = parse_ase_fmax(case_dir / "pristine_relax.log")
            v_step, v_fmax = parse_ase_fmax(case_dir / "vacancy_relax.log")
            rows.append(
                {
                    "series": root.name,
                    "setting": manifest.get("setting", case_dir.name),
                    "scan_type": manifest.get("scan_type", ""),
                    "relaxation_mode": result.get("relaxation_mode", ""),
                    "N_pristine": manifest.get("pristine_n_atoms", ""),
                    "N_vacancy": manifest.get("vacancy_n_atoms", ""),
                    "spacing_A": manifest.get("spacing_A", ""),
                    "Ef_vac_eV": result.get("vacancy_formation_energy_eV", math.nan),
                    "pristine_final_step": p_step,
                    "pristine_final_fmax_eV_A": p_fmax,
                    "vacancy_final_step": v_step,
                    "vacancy_final_fmax_eV_A": v_fmax,
                    "target_fmax_eV_A": float(args.target_fmax),
                    "pristine_meets_target": bool(p_fmax < float(args.target_fmax)) if math.isfinite(p_fmax) else False,
                    "vacancy_meets_target": bool(v_fmax < float(args.target_fmax)) if math.isfinite(v_fmax) else False,
                    "done": (case_dir / "result.json").exists(),
                    "case_dir": str(case_dir),
                }
            )

    out = Path(args.out).expanduser().resolve() if args.out else Path("dftpy_vcrelax_fmax_summary.csv").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote: {out}")
    for row in rows:
        print(
            f"{row['series']} {row['setting']} "
            f"Ef={float(row['Ef_vac_eV']):.6f} "
            f"P_fmax={row['pristine_final_fmax_eV_A']} "
            f"V_fmax={row['vacancy_final_fmax_eV_A']} "
            f"done={row['done']}"
        )


if __name__ == "__main__":
    main()
