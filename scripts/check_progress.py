#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _latest_summary_row(summary_file: Path):
    with summary_file.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    return rows[-1], len(rows)


def _pick_stress(row: dict[str, str]) -> tuple[float | None, str]:
    for key in ("eng_stress_top_GPa", "sigma_zz_GPa"):
        value = (row.get(key) or "").strip()
        if value:
            try:
                return float(value), key
            except ValueError:
                continue
    return None, "n/a"


_OPT_LINE_RE = re.compile(
    r"^(?P<method>BFGS|FIRE):\s+"
    r"(?P<step>\d+)\s+"
    r"(?P<time>\S+)\s+"
    r"(?P<energy>[-+0-9.eE]+)\s+"
    r"(?P<fmax>[-+0-9.eE]+)\s*$"
)


def _latest_optimizer_status(case_path: Path):
    candidate_names = ["init_relax.log"]
    candidate_names.extend(
        sorted(
            [p.name for p in case_path.glob("cycle_*_relax.log")],
            reverse=True,
        )
    )
    seen = set()
    for name in candidate_names:
        if name in seen:
            continue
        seen.add(name)
        log_path = case_path / name
        if not log_path.exists():
            continue
        lines = [line.strip() for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines()]
        matches = [_OPT_LINE_RE.match(line) for line in lines]
        matches = [m for m in matches if m is not None]
        if not matches:
            traj_path = log_path.with_suffix(".traj")
            if name == "init_relax.log" and (
                log_path.stat().st_size == 0
                or (traj_path.exists() and traj_path.stat().st_mtime >= log_path.stat().st_mtime)
            ):
                return {
                    "log_name": name,
                    "method": "INIT",
                    "step": None,
                    "time": None,
                    "energy": None,
                    "fmax": None,
                }
            continue
        m = matches[-1]
        return {
            "log_name": name,
            "method": m.group("method"),
            "step": int(m.group("step")),
            "time": m.group("time"),
            "energy": float(m.group("energy")),
            "fmax": float(m.group("fmax")),
        }
    return None


def main() -> None:
    results_roots = [ROOT / "results", ROOT / "cases"]
    case_dirs: list[Path] = []

    if (ROOT / "results").exists():
        case_dirs.extend(sorted((ROOT / "results").glob("*")))

    if (ROOT / "cases").exists():
        for case_dir in sorted((ROOT / "cases").glob("*")):
            results_dir = case_dir / "results"
            if results_dir.exists():
                case_dirs.extend(sorted(results_dir.glob("*")))

    print("\n" + "=" * 72)
    print("Finite-Wire Tensile Progress")
    print("=" * 72)

    shown = 0
    for case_path in sorted(case_dirs, key=lambda p: p.stat().st_mtime, reverse=True):
        if not case_path.is_dir():
            continue
        summary_file = case_path / "summary.csv"
        if not summary_file.exists():
            opt = _latest_optimizer_status(case_path)
            if opt is None:
                print(f"[wait] {case_path.name}")
                print("  summary.csv not written yet")
            else:
                print(f"[live] {case_path.name}")
                if opt["method"] == "INIT":
                    print(f"  {opt['log_name']} started; waiting for first optimizer step")
                else:
                    print(
                        f"  {opt['log_name']} {opt['method']} step={opt['step']:03d} "
                        f"fmax={opt['fmax']:.6f} eV/A energy={opt['energy']:.6f}"
                    )
            continue

        latest = _latest_summary_row(summary_file)
        if latest is None:
            opt = _latest_optimizer_status(case_path)
            if opt is None:
                print(f"[init] {case_path.name}")
                print("  summary.csv exists but has no completed cycle yet")
            else:
                print(f"[live] {case_path.name}")
                if opt["method"] == "INIT":
                    print(f"  {opt['log_name']} started; waiting for first optimizer step")
                else:
                    print(
                        f"  {opt['log_name']} {opt['method']} step={opt['step']:03d} "
                        f"fmax={opt['fmax']:.6f} eV/A energy={opt['energy']:.6f}"
                    )
            continue

        row, cycle = latest
        strain = float((row.get("strain") or "0").strip() or 0.0)
        stress, stress_key = _pick_stress(row)
        stress_text = "n/a" if stress is None else f"{stress:9.3f} GPa"
        print(f"[run ] {case_path.name}")
        print(
            f"  cycle={cycle:03d} strain={strain * 100.0:7.2f}% "
            f"stress={stress_text} ({stress_key})"
        )
        shown += 1

    if shown == 0:
        print("No active result folders found.")

    print("-" * 72)


if __name__ == "__main__":
    main()
