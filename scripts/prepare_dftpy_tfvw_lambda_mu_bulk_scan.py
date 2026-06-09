from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_float_list(text: str, *, label: str) -> list[float]:
    values = sorted({float(token.strip()) for token in str(text).split(",") if token.strip()})
    if not values:
        raise ValueError(f"No {label} values were provided.")
    return values


def float_token(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def setting_name(lambda_tf: float, mu_vw: float) -> str:
    return f"lambda_{float_token(lambda_tf)}_mu_{float_token(mu_vw)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a DFTpy TF+vW lambda-mu full cell-relaxation scan."
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--lambda-list",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    parser.add_argument(
        "--mu-list",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    parser.add_argument("--initial-a0", type=float, default=4.039848)
    parser.add_argument("--spacing", type=float, default=0.20)
    parser.add_argument("--pp", default=str(ROOT / "al.lda.recpot"))
    parser.add_argument("--xc", default="LDA")
    parser.add_argument("--fmax", type=float, default=0.002)
    parser.add_argument("--relax-steps", type=int, default=500)
    parser.add_argument("--opt-method", default="CG-HS")
    parser.add_argument("--opt-maxiter", type=int, default=500)
    parser.add_argument("--opt-maxfun", type=int, default=500)
    parser.add_argument("--min-solid-a0", type=float, default=3.0)
    parser.add_argument("--max-solid-a0", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    pp_source = Path(args.pp).expanduser().resolve()
    if not pp_source.exists():
        raise FileNotFoundError(f"Missing pseudopotential: {pp_source}")

    lambda_values = parse_float_list(args.lambda_list, label="lambda")
    mu_values = parse_float_list(args.mu_list, label="mu")
    outdir.mkdir(parents=True, exist_ok=True)
    pseudo_dir = outdir / "pseudopotential"
    pseudo_dir.mkdir(exist_ok=True)
    pp_copy = pseudo_dir / pp_source.name
    shutil.copy2(pp_source, pp_copy)

    settings: list[str] = []
    for lambda_tf in lambda_values:
        for mu_vw in mu_values:
            setting = setting_name(lambda_tf, mu_vw)
            settings.append(setting)
            case_dir = outdir / "lambda_mu_scan" / setting
            case_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "setting": setting,
                "scan_type": "lambda_mu_bulk_full_cell_relaxation",
                "material": "fcc Al",
                "cell_basis": "conventional cubic fcc",
                "n_atoms": 4,
                "kedf": "TFVW",
                "lambda_tf": lambda_tf,
                "mu_vw": mu_vw,
                "dftpy_kedf_x": lambda_tf,
                "dftpy_kedf_y": mu_vw,
                "coefficient_definition": "T_s = lambda_TF * T_TF + mu_vW * T_vW",
                "coefficient_constraint": "lambda_TF and mu_vW are independent",
                "xc": str(args.xc).strip().upper(),
                "spacing_A": float(args.spacing),
                "initial_a0_A": float(args.initial_a0),
                "fmax_eV_A": float(args.fmax),
                "relax_steps": int(args.relax_steps),
                "hydrostatic_strain": True,
                "target_pressure_GPa": 0.0,
                "pp_file": str(pp_copy),
                "opt_method": str(args.opt_method),
                "opt_maxiter": int(args.opt_maxiter),
                "opt_maxfun": int(args.opt_maxfun),
                "min_solid_a0_A": float(args.min_solid_a0),
                "max_solid_a0_A": float(args.max_solid_a0),
            }
            (case_dir / "point_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )

    (outdir / "settings_lambda_mu_scan.txt").write_text(
        "\n".join(settings) + "\n", encoding="utf-8"
    )
    top_manifest = {
        "workflow": "dftpy_tfvw_lambda_mu_bulk_full_cell_relaxation",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "equation": "T_s[n] = lambda_TF T_TF[n] + mu_vW T_vW[n]",
        "lambda_tf_values": lambda_values,
        "mu_vw_values": mu_values,
        "initial_a0_A": float(args.initial_a0),
        "spacing_A": float(args.spacing),
        "xc": str(args.xc).strip().upper(),
        "pseudopotential": str(pp_copy),
        "relaxation_mode": "full atom-and-hydrostatic-cell relaxation",
        "fmax_eV_A": float(args.fmax),
        "relax_steps": int(args.relax_steps),
        "valid_solid_a0_range_A": [
            float(args.min_solid_a0),
            float(args.max_solid_a0),
        ],
        "n_settings": len(settings),
    }
    (outdir / "manifest.json").write_text(
        json.dumps(top_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("============================================================")
    print("DFTpy TF+vW lambda-mu full cell-relaxation scan prepared")
    print("============================================================")
    print(f"Root        : {outdir}")
    print(f"Cases       : {len(settings)}")
    print(f"Initial a0  : {args.initial_a0:.6f} A")
    print(f"Target fmax : {args.fmax:.6f} eV/A")


if __name__ == "__main__":
    main()
