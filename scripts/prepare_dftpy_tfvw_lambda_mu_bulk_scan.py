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
        description="Prepare a two-dimensional DFTpy TF+vW lambda-mu bulk Al scan."
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--lambda-list",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Thomas-Fermi coefficients.",
    )
    parser.add_argument(
        "--mu-list",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="von Weizsaecker coefficients.",
    )
    parser.add_argument(
        "--a0-list",
        default=(
            "2.20,2.30,2.40,2.50,2.60,2.70,2.80,2.90,3.00,3.10,"
            "3.20,3.30,3.40,3.50,3.60,3.70,3.80,3.90,4.00,4.10,"
            "4.20,4.30,4.40,4.50,4.60,4.70,4.80,4.90,5.00"
        ),
        help="fcc lattice constants used for each EOS scan, in Angstrom.",
    )
    parser.add_argument("--spacing", type=float, default=0.20)
    parser.add_argument("--repeat", default="1x1x1")
    parser.add_argument("--pp", default=str(ROOT / "al.lda.recpot"))
    parser.add_argument("--xc", default="LDA")
    parser.add_argument("--opt-method", default="CG-HS")
    parser.add_argument("--opt-maxiter", type=int, default=500)
    parser.add_argument("--opt-maxfun", type=int, default=500)
    return parser.parse_args()


def parse_repeat(text: str) -> tuple[int, int, int]:
    normalized = str(text).lower().replace(",", "x")
    parts = [int(token.strip()) for token in normalized.split("x") if token.strip()]
    if len(parts) != 3 or any(value <= 0 for value in parts):
        raise ValueError(f"repeat must contain three positive integers, got {text!r}")
    return tuple(parts)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    pp_source = Path(args.pp).expanduser().resolve()
    if not pp_source.exists():
        raise FileNotFoundError(f"Missing pseudopotential: {pp_source}")

    lambda_values = parse_float_list(args.lambda_list, label="lambda")
    mu_values = parse_float_list(args.mu_list, label="mu")
    a0_values = parse_float_list(args.a0_list, label="a0")
    repeat = parse_repeat(args.repeat)

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
                "scan_type": "lambda_mu_bulk_eos",
                "material": "fcc Al",
                "kedf": "TFVW",
                "lambda_tf": lambda_tf,
                "mu_vw": mu_vw,
                "dftpy_kedf_x": lambda_tf,
                "dftpy_kedf_y": mu_vw,
                "coefficient_definition": "T_s = lambda_TF * T_TF + mu_vW * T_vW",
                "coefficient_constraint": "lambda_TF and mu_vW are independent; no sum-to-one constraint",
                "xc": str(args.xc).strip().upper(),
                "spacing_A": float(args.spacing),
                "repeat": list(repeat),
                "a0_scan_A": a0_values,
                "pp_file": str(pp_copy),
                "opt_method": str(args.opt_method),
                "opt_maxiter": int(args.opt_maxiter),
                "opt_maxfun": int(args.opt_maxfun),
            }
            (case_dir / "point_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )

    (outdir / "settings_lambda_mu_scan.txt").write_text(
        "\n".join(settings) + "\n", encoding="utf-8"
    )
    top_manifest = {
        "workflow": "dftpy_tfvw_lambda_mu_bulk_eos_scan",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": (
            "Map total energy, KEDF energy, and equilibrium fcc lattice constant "
            "while independently varying the Thomas-Fermi and von Weizsaecker coefficients."
        ),
        "equation": "T_s[n] = lambda_TF T_TF[n] + mu_vW T_vW[n]",
        "lambda_tf_values": lambda_values,
        "mu_vw_values": mu_values,
        "a0_scan_A": a0_values,
        "spacing_A": float(args.spacing),
        "repeat": list(repeat),
        "xc": str(args.xc).strip().upper(),
        "pseudopotential": str(pp_copy),
        "opt_method": str(args.opt_method),
        "opt_maxiter": int(args.opt_maxiter),
        "opt_maxfun": int(args.opt_maxfun),
        "n_settings": len(settings),
    }
    (outdir / "manifest.json").write_text(
        json.dumps(top_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (outdir / "README.md").write_text(
        "# DFTpy TF+vW lambda-mu bulk scan\n\n"
        "This workflow varies the Thomas-Fermi coefficient `lambda_TF` and the "
        "von Weizsaecker coefficient `mu_vW` independently. It does not impose "
        "`lambda_TF + mu_vW = 1`.\n\n"
        "Each setting performs an fcc Al lattice-constant/EOS scan and records "
        "the equilibrium total energy, kinetic (KEDF) energy, and lattice constant.\n",
        encoding="utf-8",
    )

    print("============================================================")
    print("DFTpy TF+vW lambda-mu bulk scan prepared")
    print("============================================================")
    print(f"Root        : {outdir}")
    print(f"Lambda rows : {len(lambda_values)}")
    print(f"Mu columns  : {len(mu_values)}")
    print(f"Cases       : {len(settings)}")
    print(f"a0 points   : {len(a0_values)}")
    print(f"Settings    : {outdir / 'settings_lambda_mu_scan.txt'}")


if __name__ == "__main__":
    main()
