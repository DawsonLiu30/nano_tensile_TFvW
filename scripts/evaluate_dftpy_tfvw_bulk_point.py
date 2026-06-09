from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ase.io import write


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_dftpy_tfvw_lambda_mu_bulk_one import evaluate_point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one isolated DFTpy bulk point.")
    parser.add_argument("--request", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    atoms, row = evaluate_point(
        a0_A=float(request["a0_A"]),
        repeat=tuple(int(value) for value in request["repeat"]),
        pp_file=Path(request["pp_file"]),
        spacing_A=float(request["spacing_A"]),
        xc=str(request["xc"]),
        lambda_tf=float(request["lambda_tf"]),
        mu_vw=float(request["mu_vw"]),
        opt_method=str(request["opt_method"]),
        opt_maxiter=int(request["opt_maxiter"]),
        opt_maxfun=int(request["opt_maxfun"]),
        outfile=Path(request["dftpy_outfile"]),
        scf_logfile=Path(request["scf_logfile"]),
    )
    write(Path(request["structure_path"]), atoms, direct=True, vasp5=True)
    Path(request["result_path"]).write_text(
        json.dumps(row, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    if row["status"] != "OK":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
