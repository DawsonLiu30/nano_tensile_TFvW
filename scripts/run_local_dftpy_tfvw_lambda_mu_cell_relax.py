from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ONE_CASE = ROOT / "scripts" / "run_dftpy_tfvw_lambda_mu_bulk_one.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a prepared DFTpy lambda-mu cell-relaxation scan locally."
    )
    parser.add_argument("--rootdir", required=True)
    parser.add_argument("--rerun-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rootdir = Path(args.rootdir).expanduser().resolve()
    settings = [
        line.strip()
        for line in (rootdir / "settings_lambda_mu_scan.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    stable = 0
    unstable = 0
    failed = 0
    for index, setting in enumerate(settings, start=1):
        case_dir = rootdir / "lambda_mu_scan" / setting
        result_path = case_dir / "result.json"
        if result_path.exists() and not args.rerun_existing:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("stable_fcc_equilibrium"):
                stable += 1
            elif result.get("relaxation_converged"):
                unstable += 1
            else:
                failed += 1
            print(
                f"[{index:03d}/{len(settings):03d}] {setting} "
                f"existing status={result.get('status', 'UNKNOWN')}"
            )
            continue

        completed = subprocess.run(
            [
                sys.executable,
                str(ONE_CASE),
                "--rootdir",
                str(rootdir),
                "--setting",
                setting,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (case_dir / "local_runner.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        print(
            f"[{index:03d}/{len(settings):03d}] {setting} rc={completed.returncode} "
            f"{completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ''}"
        )
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("stable_fcc_equilibrium"):
                stable += 1
            elif result.get("relaxation_converged"):
                unstable += 1
            else:
                failed += 1
        else:
            failed += 1

    print("============================================================")
    print(f"Stable fcc equilibria : {stable}/{len(settings)}")
    print(f"Relaxed but unstable  : {unstable}/{len(settings)}")
    print(f"Failed/unconverged    : {failed}/{len(settings)}")
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
