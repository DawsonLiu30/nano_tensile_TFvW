#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from ase.io import write

from ase_nanocrystal import build_wulff_nanocrystal
from auto_prep import get_grip_indices


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Create a reproducible tensile case under cases/<case>/inputs."
    )
    ap.add_argument("--case", required=True, help="Case name, e.g. nc_large_vac01")
    ap.add_argument("--force", action="store_true", help="Overwrite existing case directory")

    # Geometry
    ap.add_argument("--a0", type=float, default=4.05)
    ap.add_argument("--size", type=float, default=3.5)
    ap.add_argument("--length-z", type=float, default=40.0)
    ap.add_argument("--vacuum", type=float, default=5.0)
    ap.add_argument("--gamma100", type=float, default=1.00)
    ap.add_argument("--gamma110", type=float, default=1.06)

    # Grip selection
    ap.add_argument("--end-frac", type=float, default=0.10)
    ap.add_argument("--min-thickness", type=float, default=2.0)
    ap.add_argument("--max-thickness", type=float, default=8.0)
    ap.add_argument("--min-layers", type=int, default=2)
    ap.add_argument("--max-layers", type=int, default=0, help="0 means unlimited")
    ap.add_argument("--eps", type=float, default=1e-3)

    # Vacancy setup (optional)
    ap.add_argument("--vacancy", action="store_true", help="Enable vacancy generation")
    ap.add_argument("--vac-mode", choices=["one", "count", "conc"], default="conc")
    ap.add_argument("--vac-n", type=int, default=1)
    ap.add_argument("--vac-conc-pct", type=float, default=0.1)
    ap.add_argument("--vac-conc-basis", choices=["total", "free"], default="free")
    ap.add_argument("--vac-region", choices=["free", "all"], default="free")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--write-run-script",
        action="store_true",
        help="Also write cases/<case>/run_main.sh",
    )
    return ap.parse_args()


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    cases_dir = root / "cases"
    case_dir = cases_dir / args.case
    inputs_dir = case_dir / "inputs"

    if case_dir.exists():
        if not args.force:
            raise RuntimeError(f"Case already exists: {case_dir}. Use --force to overwrite.")
        shutil.rmtree(case_dir)

    inputs_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build structure
    atoms = build_wulff_nanocrystal(
        a0=float(args.a0),
        size=float(args.size),
        length_z=float(args.length_z),
        vacuum=float(args.vacuum),
        gamma100=float(args.gamma100),
        gamma110=float(args.gamma110),
    )
    write(str(inputs_dir / "raw_structure.xyz"), atoms)
    write(str(inputs_dir / "raw_structure.vasp"), atoms, vasp5=True, direct=True)

    # 2) Build grips
    max_layers = None if int(args.max_layers) == 0 else int(args.max_layers)
    bottom_idx, top_idx = get_grip_indices(
        atoms,
        end_frac=float(args.end_frac),
        min_thickness=float(args.min_thickness),
        max_thickness=float(args.max_thickness),
        min_layers=int(args.min_layers),
        max_layers=max_layers,
        eps=float(args.eps),
        debug=False,
    )
    np.save(str(inputs_dir / "raw_bottom_idx.npy"), bottom_idx)
    np.save(str(inputs_dir / "raw_top_idx.npy"), top_idx)

    # 3) Vacancy (optional)
    if bool(args.vacancy):
        cmd = [
            sys.executable,
            str(root / "make_vacancy.py"),
            "--input",
            "raw_structure.xyz",
            "--bottom",
            "raw_bottom_idx.npy",
            "--top",
            "raw_top_idx.npy",
            "--seed",
            str(int(args.seed)),
            "--tag",
            "init",
            "--mode",
            str(args.vac_mode),
            "--n",
            str(int(args.vac_n)),
            "--conc-pct",
            str(float(args.vac_conc_pct)),
            "--conc-basis",
            str(args.vac_conc_basis),
            "--region",
            str(args.vac_region),
        ]
        _run(cmd, cwd=inputs_dir)
        shutil.copy2(inputs_dir / "bottom_idx_init.npy", inputs_dir / "bottom_idx.npy")
        shutil.copy2(inputs_dir / "top_idx_init.npy", inputs_dir / "top_idx.npy")
    else:
        shutil.copy2(inputs_dir / "raw_structure.xyz", inputs_dir / "init.xyz")
        shutil.copy2(inputs_dir / "raw_structure.vasp", inputs_dir / "init.vasp")
        shutil.copy2(inputs_dir / "raw_bottom_idx.npy", inputs_dir / "bottom_idx.npy")
        shutil.copy2(inputs_dir / "raw_top_idx.npy", inputs_dir / "top_idx.npy")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "case": args.case,
        "geometry": {
            "a0": args.a0,
            "size": args.size,
            "length_z": args.length_z,
            "vacuum": args.vacuum,
            "gamma100": args.gamma100,
            "gamma110": args.gamma110,
        },
        "grips": {
            "end_frac": args.end_frac,
            "min_thickness": args.min_thickness,
            "max_thickness": args.max_thickness,
            "min_layers": args.min_layers,
            "max_layers": args.max_layers,
            "eps": args.eps,
        },
        "vacancy": {
            "enabled": bool(args.vacancy),
            "mode": args.vac_mode,
            "n": args.vac_n,
            "conc_pct": args.vac_conc_pct,
            "conc_basis": args.vac_conc_basis,
            "region": args.vac_region,
            "seed": args.seed,
        },
        "artifacts": {
            "inputs_dir": str(inputs_dir),
            "init_structure": str(inputs_dir / "init.vasp"),
            "bottom_idx": str(inputs_dir / "bottom_idx.npy"),
            "top_idx": str(inputs_dir / "top_idx.npy"),
        },
    }
    (case_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if bool(args.write_run_script):
        run_script = case_dir / "run_main.sh"
        run_script.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    'ROOT="$(cd "$(dirname "$0")/../.." && pwd)"',
                    f'CASE="{args.case}"',
                    'python3 "$ROOT/main.py" \\',
                    '  --case "$CASE" \\',
                    '  --workdir "$ROOT/cases/$CASE" \\',
                    '  --init "$ROOT/cases/$CASE/inputs/init.vasp" \\',
                    '  --pp "$ROOT/al.gga.psp" \\',
                    '  --bottom-idx "$ROOT/cases/$CASE/inputs/bottom_idx.npy" \\',
                    '  --top-idx "$ROOT/cases/$CASE/inputs/top_idx.npy" \\',
                    "  --step 0.005 \\",
                    "  --cycles 120 \\",
                    "  --relax-steps 80 \\",
                    "  --plot-summary",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        run_script.chmod(0o755)

    print(f"[case] Created: {case_dir}")
    print(f"[case] Inputs : {inputs_dir}")
    print(f"[case] Init   : {inputs_dir / 'init.vasp'}")
    print(f"[case] Grips  : {inputs_dir / 'bottom_idx.npy'} , {inputs_dir / 'top_idx.npy'}")
    print(f"[case] Manifest: {case_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
