from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path(
    r"C:\Users\dawso\Desktop\DFTPY_VACLM_PROF_DELIVERY_20260629_ISERVICE_ONLY\03_RAW_CASES\03_runs"
)
DEFAULT_OUT_ROOT = Path(
    r"C:\Users\dawso\Desktop\LOCAL_DFTPY_OFFICIAL_RELAX_VACLM_10X10_20260629"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run the 10x10 VACLM matrix locally, one official-style DFTpy/ASE relaxation case at a time."
    )
    ap.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--optimizer", default="BFGS", choices=["BFGS", "LBFGS", "BFGSLineSearch", "SciPyFminBFGS", "SciPyFminCG", "MDMin"])
    ap.add_argument("--fmax", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--abort-filter-fmax", type=float, default=25.0)
    ap.add_argument("--abort-after-steps", type=int, default=50)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit for smoke tests.")
    return ap.parse_args()


def load_cases(source_root: Path) -> list[Path]:
    cases = sorted(p for p in source_root.iterdir() if p.is_dir() and p.name.startswith("tfvw_"))
    if not cases:
        raise RuntimeError(f"No tfvw_* case folders found in {source_root}")
    return cases


def write_progress(out_root: Path, payload: dict) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "local_official_rerun_progress.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def append_summary(out_root: Path, row: dict) -> None:
    path = out_root / "local_official_rerun_summary.csv"
    exists = path.exists()
    fieldnames = [
        "timestamp",
        "setting",
        "status",
        "returncode",
        "elapsed_s",
        "result_json",
        "stdout_log",
        "stderr_log",
    ]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve().parent / "run_dftpy_official_relax_vaclm_one.py"
    if not script.exists():
        raise FileNotFoundError(script)

    cases = load_cases(source_root)
    if args.limit is not None:
        cases = cases[: int(args.limit)]

    env = os.environ.copy()
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[key] = str(int(args.threads))
    env["PYTHONUNBUFFERED"] = "1"

    started = datetime.now().isoformat(timespec="seconds")
    progress = {
        "workflow": "local_official_dftpy_relax_vaclm_10x10",
        "official_reference": "https://dftpy.rutgers.edu/tutorials/ofdft/relax.html",
        "source_root": str(source_root),
        "out_root": str(out_root),
        "started": started,
        "optimizer": str(args.optimizer),
        "fmax_override": args.fmax,
        "steps_override": args.steps,
        "abort_filter_fmax": float(args.abort_filter_fmax),
        "abort_after_steps": int(args.abort_after_steps),
        "threads": int(args.threads),
        "total_cases": len(cases),
        "attempted": 0,
        "completed_ok": 0,
        "failed": 0,
        "current": None,
        "last_update": started,
    }
    write_progress(out_root, progress)

    (out_root / "RUN_DESCRIPTION.md").write_text(
        "\n".join(
            [
                "# Local DFTpy official-style VACLM rerun",
                "",
                "This run follows the DFTpy relaxation tutorial pattern:",
                "",
                "- Build `DefaultOption` / `OptionFormat` in Python.",
                "- Use `DFTpyCalculator(config=conf)` with `calctype = Energy Force Stress`.",
                "- Attach the calculator to ASE atoms.",
                "- Relax with ASE `UnitCellFilter` and the selected ASE optimizer.",
                "- Write `.traj`, relax logs, final VASP/XYZ structures, DFTpy energy summaries, and result.json.",
                "",
                "Official DFTpy reference:",
                "https://dftpy.rutgers.edu/tutorials/ofdft/relax.html",
                "",
                f"Source root: `{source_root}`",
                f"Output root: `{out_root}`",
        f"Optimizer: `{args.optimizer}`",
        f"Abort guard: filter fmax > `{args.abort_filter_fmax}` after `{args.abort_after_steps}` optimizer steps",
        f"Threads: `{args.threads}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    for idx, source_case in enumerate(cases):
        setting = source_case.name
        out_case = out_root / "03_runs" / setting
        stdout_log = out_case / "official_case_stdout.log"
        stderr_log = out_case / "official_case_stderr.log"
        result_json = out_case / "result.json"

        if result_json.exists() and not args.force:
            status = "skipped_existing"
            progress["completed_ok"] += 1
            append_summary(
                out_root,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "setting": setting,
                    "status": status,
                    "returncode": 0,
                    "elapsed_s": 0,
                    "result_json": str(result_json),
                    "stdout_log": str(stdout_log),
                    "stderr_log": str(stderr_log),
                },
            )
            continue

        out_case.mkdir(parents=True, exist_ok=True)
        progress["attempted"] += 1
        progress["current"] = {"index": idx, "setting": setting, "source_case": str(source_case), "out_case": str(out_case)}
        progress["last_update"] = datetime.now().isoformat(timespec="seconds")
        write_progress(out_root, progress)

        cmd = [
            sys.executable,
            "-u",
            str(script),
            "--source-case",
            str(source_case),
            "--out-case",
            str(out_case),
            "--optimizer",
            str(args.optimizer),
            "--abort-filter-fmax",
            str(args.abort_filter_fmax),
            "--abort-after-steps",
            str(args.abort_after_steps),
        ]
        if args.fmax is not None:
            cmd.extend(["--fmax", str(args.fmax)])
        if args.steps is not None:
            cmd.extend(["--steps", str(args.steps)])
        if args.force:
            cmd.append("--force")

        start = time.time()
        with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
            proc = subprocess.run(cmd, stdout=out, stderr=err, env=env)
        elapsed = time.time() - start

        if proc.returncode == 0 and result_json.exists():
            status = "ok"
            progress["completed_ok"] += 1
        else:
            status = "failed"
            progress["failed"] += 1
            (out_case / "LOCAL_OFFICIAL_RERUN_FAILED.txt").write_text(
                f"returncode={proc.returncode}\nelapsed_s={elapsed:.3f}\n",
                encoding="utf-8",
            )

        append_summary(
            out_root,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "setting": setting,
                "status": status,
                "returncode": proc.returncode,
                "elapsed_s": f"{elapsed:.3f}",
                "result_json": str(result_json),
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
            },
        )
        progress["last_update"] = datetime.now().isoformat(timespec="seconds")
        write_progress(out_root, progress)

    progress["current"] = None
    progress["finished"] = datetime.now().isoformat(timespec="seconds")
    write_progress(out_root, progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
