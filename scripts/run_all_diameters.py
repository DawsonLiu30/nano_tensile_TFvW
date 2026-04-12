#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _tag_float(value: float) -> str:
    return f"{float(value):.1f}".replace(".", "p")


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    shown = " ".join(cmd)
    print(f"\n[run] {shown}")
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def _case_name(prefix: str, orientation: str, diameter_nm: float, vac_pct: float) -> str:
    return (
        f"{prefix}_{orientation}_d{_tag_float(diameter_nm)}nm_"
        f"vac{_tag_float(vac_pct)}"
    )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Prepare and optionally run finite-length faceted tensile cases."
    )
    ap.add_argument("--orientations", nargs="+", default=["111", "100", "110"])
    ap.add_argument("--diameters", nargs="+", type=float, default=[1.0, 2.0])
    ap.add_argument("--length-z", type=float, default=200.0)
    ap.add_argument("--vacuum", type=float, default=10.0)
    ap.add_argument("--gamma100", type=float, default=1.00)
    ap.add_argument("--gamma110", type=float, default=1.06)
    ap.add_argument("--vacancy-pct", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--case-prefix", default="paperfinite")
    ap.add_argument("--pp", default=str(ROOT / "al.gga.recpot"))
    ap.add_argument(
        "--init-name",
        default="init.vasp",
        help="Input structure filename under cases/<case>/inputs/ to pass to main.py.",
    )

    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--cycles", type=int, default=40)
    ap.add_argument("--ecut", type=float, default=1000.0)
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--relax-steps", type=int, default=200)
    ap.add_argument("--axial-vacuum", type=float, default=10.0)
    ap.add_argument("--fracture-gap-factor", type=float, default=3.0)
    ap.add_argument(
        "--init-state",
        choices=["auto", "raw", "relaxed", "checkpoint"],
        default="raw",
        help="How main.py should treat the provided --init file.",
    )
    ap.add_argument(
        "--rebuild-cases",
        action="store_true",
        help="Force regenerate case inputs even if the case directory already exists.",
    )
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--plot-summary", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    for orientation in args.orientations:
        for diameter_nm in args.diameters:
            size_A = float(diameter_nm) * 5.0
            case_name = _case_name(
                prefix=str(args.case_prefix),
                orientation=str(orientation),
                diameter_nm=float(diameter_nm),
                vac_pct=float(args.vacancy_pct),
            )
            case_dir = ROOT / "cases" / case_name

            print("\n" + "=" * 72)
            print(
                f"[case] orientation={orientation} diameter={diameter_nm:.1f} nm "
                f"size_A={size_A:.3f} case={case_name}"
            )
            print("=" * 72)

            init_input = case_dir / "inputs" / str(args.init_name)
            need_create = bool(args.rebuild_cases) or not init_input.exists()
            if need_create:
                create_cmd = [
                    PYTHON,
                    str(ROOT / "scripts" / "create_case.py"),
                    "--case",
                    case_name,
                    "--a0",
                    "4.05",
                    "--size",
                    f"{size_A}",
                    "--length-z",
                    f"{float(args.length_z)}",
                    "--vacuum",
                    f"{float(args.vacuum)}",
                    "--orientation",
                    str(orientation),
                    "--gamma100",
                    f"{float(args.gamma100)}",
                    "--gamma110",
                    f"{float(args.gamma110)}",
                    "--min-thickness",
                    "4.0",
                    "--max-thickness",
                    "4.0",
                    "--vacancy",
                    "--vac-mode",
                    "conc",
                    "--vac-conc-pct",
                    f"{float(args.vacancy_pct)}",
                    "--vac-conc-basis",
                    "free",
                    "--vac-region",
                    "free",
                    "--seed",
                    f"{int(args.seed)}",
                ]
                if bool(args.rebuild_cases):
                    create_cmd.insert(4, "--force")
                _run(create_cmd, cwd=ROOT)
            else:
                print(f"[case] Reusing existing inputs: {case_dir}")

            if args.prepare_only:
                continue

            run_cmd = [
                PYTHON,
                str(ROOT / "main.py"),
                "--case",
                case_name,
                "--workdir",
                str(case_dir),
                "--init",
                str(init_input),
                "--pp",
                str(args.pp),
                "--bottom-idx",
                str(case_dir / "inputs" / "bottom_idx.npy"),
                "--top-idx",
                str(case_dir / "inputs" / "top_idx.npy"),
                "--init-state",
                str(args.init_state),
                "--ecut",
                f"{float(args.ecut)}",
                "--step",
                f"{float(args.step)}",
                "--cycles",
                f"{int(args.cycles)}",
                "--fmax",
                f"{float(args.fmax)}",
                "--relax-steps",
                f"{int(args.relax_steps)}",
                "--axial-vacuum",
                f"{float(args.axial_vacuum)}",
                "--fracture-gap-factor",
                f"{float(args.fracture_gap_factor)}",
            ]
            if args.plot_summary:
                run_cmd.append("--plot-summary")
            _run(run_cmd, cwd=ROOT)


if __name__ == "__main__":
    main()
