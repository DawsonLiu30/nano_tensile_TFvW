from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _parse_float_list(text: str, *, label: str) -> list[float]:
    values = []
    for chunk in str(text).split(","):
        token = chunk.strip()
        if token:
            values.append(float(token))
    if not values:
        raise ValueError(f"No valid {label} values were provided.")
    return values


def _read_a0_from_summary(path: Path) -> float:
    for line in path.read_text(encoding="utf-8").splitlines():
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


def _case_name(diameter_nm: float) -> str:
    return f"finite_grip_111_{float(diameter_nm):.1f}nm_vacancy_tfvw"


def _run_name(diameter_nm: float) -> str:
    return f"grip_r{_format_radius_tag(diameter_nm)}_vacancy_tfvw"


def _run_command(args: list[str]) -> None:
    print(f"[{_ts()}] RUN {' '.join(args)}", flush=True)
    subprocess.run(args, cwd=str(ROOT), check=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run finite-grip vacancy nanowire cases sequentially.")
    ap.add_argument("--diameters", required=True, help="Comma-separated diameters in nm, e.g. 1.0,2.0,3.0,4.0")
    ap.add_argument("--bulk-summary", default="", help="Optional bulk summary.txt used to read a0_ref_A")
    ap.add_argument("--orientation", default="111")
    ap.add_argument("--wire-length", type=float, default=21.0)
    ap.add_argument("--min-wire-span", type=float, default=10.0)
    ap.add_argument("--xy-vacuum", type=float, default=10.0)
    ap.add_argument("--z-vacuum", type=float, default=10.0)
    ap.add_argument("--grip-thickness", type=float, default=3.0)
    ap.add_argument("--vacancy-z-window-fraction", type=float, default=0.35)
    ap.add_argument("--pp", default="al.gga.recpot")
    ap.add_argument("--kedf", default="TFVW")
    ap.add_argument("--ecut", type=float, default=1000.0)
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--prep-relax-steps", type=int, default=120)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--tensile-relax-steps", type=int, default=80)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    diameters = _parse_float_list(args.diameters, label="diameters")
    bulk_summary = _latest_bulk_summary_path() if not str(args.bulk_summary).strip() else (ROOT / args.bulk_summary).resolve()
    a0_ref = _read_a0_from_summary(bulk_summary)
    python = sys.executable

    print(f"[{_ts()}] Sequential finite-grip vacancy series", flush=True)
    print(f"[{_ts()}] diameters_nm={', '.join(f'{d:.1f}' for d in diameters)}", flush=True)
    print(f"[{_ts()}] bulk_summary={bulk_summary}", flush=True)
    print(f"[{_ts()}] a0_ref_A={a0_ref:.12f}", flush=True)

    for diameter_nm in diameters:
        case = _case_name(diameter_nm)
        run_name = _run_name(diameter_nm)
        _run_command(
            [
                python,
                "scripts/prepare_grip_vacancy_wire.py",
                "--case",
                case,
                "--diameter-nm",
                f"{float(diameter_nm):.1f}",
                "--orientation",
                str(args.orientation),
                "--a0",
                f"{a0_ref:.12f}",
                "--wire-length",
                f"{float(args.wire_length):.6f}",
                "--min-wire-span",
                f"{float(args.min_wire_span):.6f}",
                "--xy-vacuum",
                f"{float(args.xy_vacuum):.6f}",
                "--z-vacuum",
                f"{float(args.z_vacuum):.6f}",
                "--grip-thickness",
                f"{float(args.grip_thickness):.6f}",
                "--vacancy-z-window-fraction",
                f"{float(args.vacancy_z_window_fraction):.6f}",
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
                "scripts/run_grip_tensile.py",
                "--case",
                run_name,
                "--workdir",
                str(Path("cases") / case),
                "--init",
                "inputs/vacancy_equilibrium.vasp",
                "--metadata",
                "inputs/grip_metadata.json",
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

    print(f"[{_ts()}] Finite-grip vacancy series completed.", flush=True)


if __name__ == "__main__":
    main()
