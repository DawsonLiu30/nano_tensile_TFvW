from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _parse_float_list(text: str, *, label: str) -> list[float]:
    values = []
    for chunk in str(text).split(","):
        token = chunk.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError(f"No valid {label} values were provided.")
    return values


def _read_a0_from_summary(path: Path) -> float:
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line.startswith("a0_ref_A="):
            return float(line.split("=", 1)[1].strip())
    raise ValueError(f"Could not find a0_ref_A in {path}")


def _latest_bulk_summary_path() -> Path:
    candidates = sorted(
        (ROOT / "results").glob("bulk_Al_fcc_TFVW*/summary.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No bulk TFVW summary.txt was found under results/.")
    return candidates[0]


def _format_radius_tag(diameter_nm: float) -> str:
    rounded = round(float(diameter_nm), 6)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return str(rounded).replace(".", "p")


def _family(shape: str) -> str:
    return "nanocolumn" if str(shape).lower() == "circle" else "nanocrystal"


def _normalize_vacancy_position(value: str) -> str:
    key = str(value).strip().lower()
    aliases = {
        "inner": "inner",
        "core": "inner",
        "center": "inner",
        "centre": "inner",
        "middle": "middle",
        "mid": "middle",
        "outer": "outer",
        "surface": "outer",
        "edge": "outer",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported vacancy radial position '{value}'. Use inner, middle, or outer.")
    return aliases[key]


def _parse_position_list(text: str) -> list[str]:
    positions = []
    for chunk in str(text).split(","):
        token = chunk.strip()
        if token:
            positions.append(_normalize_vacancy_position(token))
    if not positions:
        raise ValueError("No vacancy radial positions were provided.")
    return list(dict.fromkeys(positions))


def _case_name(diameter_nm: float, shape: str, orientation: str, vacancy_position: str) -> str:
    shape_key = str(shape).lower()
    pos_key = _normalize_vacancy_position(vacancy_position)
    return f"{_family(shape_key)}_{shape_key}_periodic_{orientation}_{float(diameter_nm):.1f}nm_vac_{pos_key}_tfvw"


def _run_name(diameter_nm: float, shape: str, vacancy_position: str) -> str:
    shape_key = str(shape).lower()
    pos_key = _normalize_vacancy_position(vacancy_position)
    return f"{_family(shape_key)}_{shape_key}_r{_format_radius_tag(diameter_nm)}_vac_{pos_key}_tfvw"


def _run_command(args: list[str]) -> None:
    print(f"[{_ts()}] RUN {' '.join(args)}")
    subprocess.run(args, cwd=str(ROOT), check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run vacancy periodic nanocolumn/nanocrystal preparation and tensile cases sequentially."
    )
    ap.add_argument("--diameters", required=True, help="Comma-separated diameters in nm, e.g. 1.0,2.0,3.0,4.0")
    ap.add_argument("--bulk-summary", default="", help="Optional bulk summary.txt used to read a0_ref_A")
    ap.add_argument("--orientation", default="111")
    ap.add_argument(
        "--cross-section-shape",
        choices=["circle", "hexagon", "triangle"],
        default="circle",
        help="circle = nanocolumn; hexagon/triangle = nanocrystal.",
    )
    ap.add_argument("--shape-rotation-deg", type=float, default=0.0)
    ap.add_argument("--vacuum", type=float, default=10.0)
    ap.add_argument("--min-short-lz", type=float, default=10.0)
    ap.add_argument("--short-repeat-z", type=int, default=0)
    ap.add_argument("--target-long-lz", type=float, default=200.0)
    ap.add_argument("--scan-scales", default="0.94,0.95,0.96,0.97,0.98,0.99,1.00,1.01,1.02")
    ap.add_argument("--vacancy-z-window-fraction", type=float, default=0.25)
    ap.add_argument(
        "--vacancy-radial-positions",
        default="outer",
        help="Comma-separated vacancy positions to run: inner,middle,outer. Default keeps the previous outer/surface vacancy behavior.",
    )
    ap.add_argument("--pp", default="al.gga.recpot")
    ap.add_argument("--kedf", default="TFVW")
    ap.add_argument("--ecut", type=float, default=1000.0)
    ap.add_argument("--fmax", type=float, default=0.002)
    ap.add_argument("--prep-relax-steps", type=int, default=120)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--tensile-relax-steps", type=int, default=80)
    args = ap.parse_args()

    diameters = _parse_float_list(args.diameters, label="diameters")
    vacancy_positions = _parse_position_list(args.vacancy_radial_positions)
    bulk_summary = _latest_bulk_summary_path() if not str(args.bulk_summary).strip() else (ROOT / args.bulk_summary).resolve()
    a0_ref = _read_a0_from_summary(bulk_summary)

    print(f"[{_ts()}] Sequential vacancy periodic series")
    print(f"[{_ts()}] cross_section_shape={args.cross_section_shape}")
    print(f"[{_ts()}] diameters_nm={', '.join(f'{d:.1f}' for d in diameters)}")
    print(f"[{_ts()}] vacancy_positions={', '.join(vacancy_positions)}")
    print(f"[{_ts()}] bulk_summary={bulk_summary}")
    print(f"[{_ts()}] a0_ref_A={a0_ref:.12f}")

    python = sys.executable
    for diameter_nm in diameters:
        for vacancy_position in vacancy_positions:
            case = _case_name(diameter_nm, str(args.cross_section_shape), str(args.orientation), vacancy_position)
            run_name = _run_name(diameter_nm, str(args.cross_section_shape), vacancy_position)

            _run_command(
                [
                    python,
                    "scripts/prepare_vacancy_periodic_wire.py",
                    "--case",
                    case,
                    "--diameter-nm",
                    f"{float(diameter_nm):.1f}",
                    "--cross-section-shape",
                    str(args.cross_section_shape),
                    "--shape-rotation-deg",
                    f"{float(args.shape_rotation_deg):.6f}",
                    "--orientation",
                    str(args.orientation),
                    "--a0",
                    f"{a0_ref:.12f}",
                    "--vacuum",
                    f"{float(args.vacuum):.6f}",
                    "--min-short-lz",
                    f"{float(args.min_short_lz):.6f}",
                    "--short-repeat-z",
                    str(int(args.short_repeat_z)),
                    "--target-long-lz",
                    f"{float(args.target_long_lz):.6f}",
                    "--scan-scales",
                    str(args.scan_scales),
                    "--vacancy-z-window-fraction",
                    f"{float(args.vacancy_z_window_fraction):.6f}",
                    "--vacancy-radial-position",
                    vacancy_position,
                    "--pp",
                    str(args.pp),
                    "--kedf",
                    str(args.kedf),
                    "--ecut",
                    f"{float(args.ecut):.6f}",
                    "--fmax",
                    f"{float(args.fmax):.6f}",
                    "--relax-steps",
                    str(int(args.prep_relax_steps)),
                ]
            )

            _run_command(
                [
                    python,
                    "scripts/run_periodic_tensile.py",
                    "--case",
                    run_name,
                    "--workdir",
                    str(Path("cases") / case),
                    "--init",
                    "inputs/vacancy_equilibrium.vasp",
                    "--pp",
                    str(args.pp),
                    "--kedf",
                    str(args.kedf),
                    "--ecut",
                    f"{float(args.ecut):.6f}",
                    "--step",
                    f"{float(args.step):.6f}",
                    "--cycles",
                    str(int(args.cycles)),
                    "--fmax",
                    f"{float(args.fmax):.6f}",
                    "--relax-steps",
                    str(int(args.tensile_relax_steps)),
                    "--plot-summary",
                ]
            )

    print(f"[{_ts()}] Vacancy series completed.")


if __name__ == "__main__":
    main()
