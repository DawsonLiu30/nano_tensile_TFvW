from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS_FILE_OVERRIDE: Path | None = None


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _parse_float_list(text: str, *, label: str) -> list[float]:
    values: list[float] = []
    for chunk in str(text).split(","):
        token = chunk.strip()
        if token:
            values.append(float(token))
    if not values:
        raise ValueError(f"No valid {label} values were provided.")
    return values


def _format_tag(value: float) -> str:
    rounded = round(float(value), 6)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return str(rounded).replace(".", "p")


def _case_name(diameter_nm: float) -> str:
    return f"finite_grip_111_{float(diameter_nm):.1f}nm_vacancy_tfvw"


def _run_name(diameter_nm: float) -> str:
    return f"grip_r{_format_tag(diameter_nm)}_vacancy_tfvw"


def _read_a0_from_summary(path: Path) -> float:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("a0_ref_A="):
            return float(line.split("=", 1)[1].strip())
    raise ValueError(f"Could not find a0_ref_A in {path}")


def _inputs_ready(case_dir: Path) -> bool:
    required = [
        case_dir / "inputs" / "vacancy_equilibrium.vasp",
        case_dir / "inputs" / "grip_metadata.json",
        case_dir / "inputs" / "grip_vacancy_manifest.json",
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def _latest_result(case_dir: Path, run_name: str) -> Path | None:
    results_dir = case_dir / "results"
    if not results_dir.exists():
        return None
    candidates = [
        path
        for path in results_dir.glob(f"{run_name}*")
        if path.is_dir() and (path / "summary.csv").exists()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _run(command: list[str], *, required: bool = True) -> bool:
    print(f"[{_ts()}] RUN {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=str(ROOT))
    if completed.returncode != 0:
        message = f"[{_ts()}] command failed with exit code {completed.returncode}: {' '.join(command)}"
        if required:
            raise RuntimeError(message)
        print(f"[WARN] {message}", flush=True)
        return False
    return True


def _status_csv() -> Path:
    out = STATUS_FILE_OVERRIDE or (ROOT / "results" / "hpc_finite_grip_series_status.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        out.write_text(
            "time,case,diameter_nm,stage,status,result_dir,last_cycle,last_strain_pct,"
            "fractured,max_gap_all_A,gap_threshold_A,message\n",
            encoding="utf-8",
        )
    return out


def _latest_summary(result_dir: Path | None) -> dict[str, str]:
    if result_dir is None:
        return {}
    summary = result_dir / "summary.csv"
    if not summary.exists():
        return {}
    with open(summary, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def _latest_fracture(result_dir: Path | None) -> dict[str, str]:
    if result_dir is None:
        return {}
    status = result_dir / "fracture_status.csv"
    if not status.exists():
        return {}
    with open(status, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def _fractured_flag(result_dir: Path | None) -> bool:
    fracture = _latest_fracture(result_dir)
    value = str(fracture.get("fractured", "")).strip().lower()
    return value in {"true", "1", "yes"}


def _append_status(
    *,
    case: str,
    diameter_nm: float,
    stage: str,
    status: str,
    result_dir: Path | None = None,
    message: str = "",
) -> None:
    summary = _latest_summary(result_dir)
    fracture = _latest_fracture(result_dir)
    last_cycle = summary.get("cycle", "")
    last_strain_pct = ""
    if summary.get("strain"):
        last_strain_pct = f"{float(summary['strain']) * 100.0:.6f}"
    row = [
        _ts(),
        case,
        f"{float(diameter_nm):.1f}",
        stage,
        status,
        str(result_dir or ""),
        last_cycle,
        last_strain_pct,
        fracture.get("fractured", ""),
        fracture.get("max_gap_all_A", ""),
        fracture.get("gap_threshold_A", ""),
        str(message).replace("\n", " "),
    ]
    with open(_status_csv(), "a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(row)


def _postprocess(args: argparse.Namespace, *, case_dir: Path, run_name: str, diameter_nm: float) -> None:
    if args.skip_postprocess:
        return
    latest = _latest_result(case_dir, run_name)
    if latest is None:
        print(f"[WARN] No result folder found for postprocess: {case_dir}", flush=True)
        return

    required = bool(args.strict_postprocess)
    _run([sys.executable, "scripts/analyze_tensile_events.py", "--results-dir", str(latest)], required=required)
    if args.skip_plots:
        return

    tag = _format_tag(diameter_nm)
    professor_dir = ROOT / "results" / "professor_review"
    professor_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "scripts/plot_grip_tensile_curve.py",
            "--results-dir",
            str(latest),
            "--cycles-target",
            str(int(args.cycles)),
            "--out",
            str(professor_dir / f"r{tag}_finite_grip_vacancy_tensile.png"),
            "--copy-csv",
        ],
        required=required,
    )
    _run(
        [
            sys.executable,
            "scripts/plot_tensile_presentation.py",
            "--results-dir",
            str(latest),
            "--out",
            str(professor_dir / f"r{tag}_finite_grip_vacancy_tensile_presentation.png"),
            "--title",
            f"Finite-grip vacancy tensile, d = {float(diameter_nm):.1f} nm",
        ],
        required=required,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Prepare, resume, run, analyze, and plot finite-grip vacancy tensile cases sequentially."
    )
    ap.add_argument("--diameters", required=True, help="Comma-separated diameters/r labels, e.g. 2,3,4,5,6,7,8")
    ap.add_argument("--a0", type=float, default=4.118877004246)
    ap.add_argument("--bulk-summary", default="", help="Optional summary.txt containing a0_ref_A; overrides --a0.")
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
    ap.add_argument("--spacing", type=float, default=None, help="Optional grid spacing in Angstrom; overrides --ecut.")
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--prep-relax-steps", type=int, default=120)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--cycles", type=int, default=80)
    ap.add_argument(
        "--cycles-per-launch",
        type=int,
        default=0,
        help="If > 0, run tensile in small subprocess chunks to avoid long-lived Python/DFTpy memory buildup.",
    )
    ap.add_argument("--tensile-relax-steps", type=int, default=80)
    ap.add_argument("--fracture-gap-factor", type=float, default=3.0)
    ap.add_argument("--force-prepare", action="store_true")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--resume-latest", action="store_true")
    ap.add_argument("--skip-postprocess", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--strict-postprocess", action="store_true")
    ap.add_argument("--status-file", default="", help="Optional per-job status CSV path.")
    return ap.parse_args()


def main() -> None:
    global STATUS_FILE_OVERRIDE
    args = parse_args()
    if str(args.status_file).strip():
        status_file = Path(args.status_file)
        STATUS_FILE_OVERRIDE = status_file.resolve() if status_file.is_absolute() else (ROOT / status_file).resolve()
    diameters = _parse_float_list(args.diameters, label="diameters")
    a0 = float(args.a0)
    if str(args.bulk_summary).strip():
        bulk_summary = Path(args.bulk_summary)
        bulk_summary = bulk_summary.resolve() if bulk_summary.is_absolute() else (ROOT / bulk_summary).resolve()
        a0 = _read_a0_from_summary(bulk_summary)
        print(f"[{_ts()}] a0 from bulk summary: {bulk_summary} -> {a0:.12f} A", flush=True)
    else:
        print(f"[{_ts()}] a0 from command/default: {a0:.12f} A", flush=True)

    print(f"[{_ts()}] finite-grip vacancy HPC series", flush=True)
    print(f"[{_ts()}] diameters_nm={', '.join(f'{d:.1f}' for d in diameters)}", flush=True)
    print(f"[{_ts()}] step={float(args.step):.6f}, cycles={int(args.cycles)}", flush=True)
    if args.spacing is not None:
        print(f"[{_ts()}] spacing={float(args.spacing):.6f} A (overrides ecut={float(args.ecut):.6f} eV)", flush=True)
    if int(args.cycles_per_launch) > 0:
        print(f"[{_ts()}] cycles_per_launch={int(args.cycles_per_launch)}", flush=True)

    for diameter_nm in diameters:
        case = _case_name(diameter_nm)
        run_name = _run_name(diameter_nm)
        case_dir = ROOT / "cases" / case
        print("=" * 72, flush=True)
        print(f"[{_ts()}] CASE {case}", flush=True)
        print("=" * 72, flush=True)

        try:
            if args.force_prepare or not _inputs_ready(case_dir):
                _append_status(case=case, diameter_nm=diameter_nm, stage="prepare", status="start")
                prepare_command = [
                    sys.executable,
                    "scripts/prepare_grip_vacancy_wire.py",
                    "--case",
                    case,
                    "--diameter-nm",
                    f"{float(diameter_nm):.1f}",
                    "--orientation",
                    str(args.orientation),
                    "--a0",
                    f"{a0:.12f}",
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
                    "--fmax",
                    f"{float(args.fmax):.6f}",
                    "--relax-steps",
                    str(int(args.prep_relax_steps)),
                ]
                if args.spacing is not None:
                    prepare_command += ["--spacing", f"{float(args.spacing):.6f}"]
                else:
                    prepare_command += ["--ecut", f"{float(args.ecut):.6f}"]
                _run(prepare_command)
                _append_status(case=case, diameter_nm=diameter_nm, stage="prepare", status="done")
            else:
                print(f"[{_ts()}] Existing relaxed inputs found; skipping prepare.", flush=True)
                _append_status(case=case, diameter_nm=diameter_nm, stage="prepare", status="skipped")

            if args.prepare_only:
                continue

            def build_tensile_command(*, cycles_target: int, resume_dir: Path | None) -> list[str]:
                command = [
                    sys.executable,
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
                    "--a0",
                    f"{a0:.12f}",
                    "--step",
                    f"{float(args.step):.6f}",
                    "--cycles",
                    str(int(cycles_target)),
                    "--fmax",
                    f"{float(args.fmax):.6f}",
                    "--relax-steps",
                    str(int(args.tensile_relax_steps)),
                    "--fracture-gap-factor",
                    f"{float(args.fracture_gap_factor):.6f}",
                    "--plot-summary",
                ]
                if args.spacing is not None:
                    command += ["--spacing", f"{float(args.spacing):.6f}"]
                else:
                    command += ["--ecut", f"{float(args.ecut):.6f}"]
                if resume_dir is not None:
                    command += ["--resume-results", str(resume_dir)]
                return command

            latest = _latest_result(case_dir, run_name) if args.resume_latest else None
            if latest is not None:
                print(f"[{_ts()}] Resuming latest result: {latest}", flush=True)
            else:
                print(f"[{_ts()}] No previous result for {run_name}; starting fresh.", flush=True)

            _append_status(case=case, diameter_nm=diameter_nm, stage="tensile", status="start", result_dir=latest)
            cycles_per_launch = int(args.cycles_per_launch)
            if cycles_per_launch > 0:
                while True:
                    latest = _latest_result(case_dir, run_name)
                    summary = _latest_summary(latest)
                    last_cycle = int(summary.get("cycle", "0")) if summary else -1
                    if last_cycle >= int(args.cycles):
                        print(f"[{_ts()}] Target cycles reached for {run_name}: {last_cycle}", flush=True)
                        break
                    if latest is not None and _fractured_flag(latest):
                        print(f"[{_ts()}] Fracture already detected for {run_name}; stopping chunk loop.", flush=True)
                        break

                    next_target = min(int(args.cycles), max(last_cycle, 0) + cycles_per_launch)
                    print(
                        f"[{_ts()}] Launching tensile chunk for {run_name}: "
                        f"last_cycle={last_cycle}, next_target={next_target}",
                        flush=True,
                    )
                    _run(build_tensile_command(cycles_target=next_target, resume_dir=latest))
                    latest = _latest_result(case_dir, run_name)
                    _append_status(
                        case=case,
                        diameter_nm=diameter_nm,
                        stage="tensile_chunk",
                        status="done",
                        result_dir=latest,
                        message=f"target_cycle={next_target}",
                    )
                    summary_after = _latest_summary(latest)
                    if not summary_after:
                        raise RuntimeError(f"No summary rows were found after tensile chunk for {run_name}.")
                    actual_cycle = int(summary_after.get("cycle", "-1"))
                    if actual_cycle < next_target and not _fractured_flag(latest):
                        raise RuntimeError(
                            f"Tensile chunk for {run_name} ended early at cycle {actual_cycle} "
                            f"before target {next_target} without fracture."
                        )
            else:
                _run(build_tensile_command(cycles_target=int(args.cycles), resume_dir=latest))

            latest_after = _latest_result(case_dir, run_name)
            _append_status(
                case=case,
                diameter_nm=diameter_nm,
                stage="tensile",
                status="done",
                result_dir=latest_after,
            )
            _postprocess(args, case_dir=case_dir, run_name=run_name, diameter_nm=diameter_nm)
            _append_status(
                case=case,
                diameter_nm=diameter_nm,
                stage="postprocess",
                status="done",
                result_dir=latest_after,
            )
        except Exception as exc:
            latest_error = _latest_result(case_dir, run_name)
            _append_status(
                case=case,
                diameter_nm=diameter_nm,
                stage="case",
                status="failed",
                result_dir=latest_error,
                message=str(exc),
            )
            raise

    print(f"[{_ts()}] finite-grip vacancy HPC series finished.", flush=True)


if __name__ == "__main__":
    main()
