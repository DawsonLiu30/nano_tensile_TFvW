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


def _is_pid_alive(pid: int) -> bool:
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    )
    line = (result.stdout or "").strip()
    if not line:
        return False
    if line.upper().startswith("INFO:"):
        return False
    return f'"{int(pid)}"' in line


def _wait_for_pid(pid: int, poll_seconds: float) -> None:
    print(f"[{_ts()}] Waiting for PID {pid} to finish...")
    while _is_pid_alive(int(pid)):
        time.sleep(float(poll_seconds))
    print(f"[{_ts()}] PID {pid} has finished.")


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


def _case_name(diameter_nm: float, shape: str, orientation: str) -> str:
    shape_key = str(shape).lower()
    return f"{_family(shape_key)}_{shape_key}_periodic_{orientation}_{float(diameter_nm):.1f}nm_tfvw"


def _run_name(diameter_nm: float, shape: str) -> str:
    shape_key = str(shape).lower()
    return f"{_family(shape_key)}_{shape_key}_r{_format_radius_tag(diameter_nm)}_tfvw"


def _run_command(args: list[str]) -> None:
    print(f"[{_ts()}] RUN {' '.join(args)}")
    subprocess.run(args, cwd=str(ROOT), check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run axially periodic nanocolumn/nanocrystal preparation and tensile cases sequentially."
    )
    ap.add_argument("--diameters", required=True, help="Comma-separated diameters in nm, e.g. 3.0,4.0")
    ap.add_argument("--wait-pid", type=int, default=0, help="Optional PID to wait for before starting the series")
    ap.add_argument(
        "--bulk-summary",
        default="",
        help="Optional bulk validation summary.txt used to read a0_ref_A; defaults to the latest TFVW bulk summary under results/",
    )
    ap.add_argument("--orientation", default="111")
    ap.add_argument(
        "--cross-section-shape",
        choices=["circle", "hexagon", "triangle"],
        default="circle",
        help="circle = nanocolumn; hexagon/triangle = nanocrystal.",
    )
    ap.add_argument("--shape-rotation-deg", type=float, default=0.0)
    ap.add_argument("--vacuum", type=float, default=10.0)
    ap.add_argument("--replicate-z", type=int, default=30)
    ap.add_argument("--scan-scales", default="0.95,0.96,0.97,0.98,0.99,1.00,1.01")
    ap.add_argument("--pp", default="al.gga.recpot")
    ap.add_argument("--kedf", default="TFVW")
    ap.add_argument("--ecut", type=float, default=1000.0)
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--prep-relax-steps", type=int, default=120)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--tensile-relax-steps", type=int, default=80)
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    args = ap.parse_args()

    diameters = _parse_float_list(args.diameters, label="diameters")
    bulk_summary = _latest_bulk_summary_path() if not str(args.bulk_summary).strip() else (ROOT / args.bulk_summary).resolve()
    a0_ref = _read_a0_from_summary(bulk_summary)

    print(f"[{_ts()}] Sequential periodic series")
    print(f"[{_ts()}] cross_section_shape={args.cross_section_shape}")
    print(f"[{_ts()}] diameters_nm={', '.join(f'{d:.1f}' for d in diameters)}")
    print(f"[{_ts()}] bulk_summary={bulk_summary}")
    print(f"[{_ts()}] a0_ref_A={a0_ref:.12f}")

    if int(args.wait_pid) > 0:
        _wait_for_pid(int(args.wait_pid), poll_seconds=float(args.poll_seconds))

    python = sys.executable
    for diameter_nm in diameters:
        case = _case_name(diameter_nm, str(args.cross_section_shape), str(args.orientation))
        run_name = _run_name(diameter_nm, str(args.cross_section_shape))

        _run_command(
            [
                python,
                "scripts/prepare_paper_periodic_wire.py",
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
                "--replicate-z",
                str(int(args.replicate_z)),
                "--scan-scales",
                str(args.scan_scales),
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
                "inputs/short_equilibrium.vasp",
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

    print(f"[{_ts()}] Series completed.")


if __name__ == "__main__":
    main()
