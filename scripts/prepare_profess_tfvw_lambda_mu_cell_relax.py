from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def parse_values(text: str) -> list[float]:
    return sorted({float(token.strip()) for token in text.split(",") if token.strip()})


def token(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def setting_name(lambda_tf: float, mu_vw: float) -> str:
    return f"lambda_{token(lambda_tf)}_mu_{token(mu_vw)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local PROFESS TF+vW cell relaxations.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--profess-bin", required=True)
    parser.add_argument("--pp", required=True)
    parser.add_argument(
        "--lambda-list", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
    )
    parser.add_argument(
        "--mu-list", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
    )
    parser.add_argument("--initial-a0", type=float, default=4.039848)
    parser.add_argument("--ecut", type=float, default=1600.0)
    parser.add_argument("--min-solid-a0", type=float, default=3.0)
    parser.add_argument("--max-solid-a0", type=float, default=6.0)
    parser.add_argument("--max-stress-gpa", type=float, default=0.1)
    return parser.parse_args()


def write_ion(path: Path, a0: float, pp_name: str) -> None:
    path.write_text(
        f"""%BLOCK LATTICE_CART
  {a0:.12f} 0.0 0.0
  0.0 {a0:.12f} 0.0
  0.0 0.0 {a0:.12f}
%END BLOCK LATTICE_CART
%BLOCK POSITIONS_FRAC
  Al 0.0 0.0 0.0
  Al 0.0 0.5 0.5
  Al 0.5 0.0 0.5
  Al 0.5 0.5 0.0
%END BLOCK POSITIONS_FRAC
%BLOCK SPECIES_POT
  Al {pp_name}
%END BLOCK SPECIES_POT
""",
        encoding="ascii",
    )


def write_input(path: Path, ecut: float, lambda_tf: float, mu_vw: float) -> None:
    path.write_text(
        f"""ecut {ecut:g}
MINI cell
method ntn
method ion non
KINE TF+
PARA LAMB {lambda_tf:.12g}
PARA MU {mu_vw:.12g}
exch lda
geometryfile bulk.ion

print minimizer density 2
print minimizer geom 2
calculate stresses
""",
        encoding="ascii",
    )


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    profess_source = Path(args.profess_bin).expanduser().resolve()
    pp_source = Path(args.pp).expanduser().resolve()
    if not profess_source.exists():
        raise FileNotFoundError(profess_source)
    if not pp_source.exists():
        raise FileNotFoundError(pp_source)

    outdir.mkdir(parents=True, exist_ok=True)
    profess_copy = outdir / "PROFESS"
    pp_copy = outdir / pp_source.name
    shutil.copy2(profess_source, profess_copy)
    shutil.copy2(pp_source, pp_copy)

    lambda_values = parse_values(args.lambda_list)
    mu_values = parse_values(args.mu_list)
    settings = []
    for lambda_tf in lambda_values:
        for mu_vw in mu_values:
            setting = setting_name(lambda_tf, mu_vw)
            settings.append(setting)
            case_dir = outdir / "lambda_mu_scan" / setting
            case_dir.mkdir(parents=True, exist_ok=True)
            write_input(case_dir / "bulk.inpt", args.ecut, lambda_tf, mu_vw)
            write_ion(case_dir / "bulk.ion", args.initial_a0, pp_copy.name)
            shutil.copy2(pp_copy, case_dir / pp_copy.name)
            manifest = {
                "setting": setting,
                "code": "PROFESS 3",
                "relaxation_mode": "MINI cell; perfect fcc ions fixed by symmetry",
                "lambda_tf": lambda_tf,
                "mu_vw": mu_vw,
                "equation": "T_s = lambda_TF * T_TF + mu_vW * T_vW",
                "initial_a0_A": args.initial_a0,
                "ecut_eV": args.ecut,
                "n_atoms": 4,
                "xc": "LDA",
                "pseudopotential": pp_copy.name,
                "min_solid_a0_A": args.min_solid_a0,
                "max_solid_a0_A": args.max_solid_a0,
                "max_stress_GPa": args.max_stress_gpa,
            }
            (case_dir / "point_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )

    (outdir / "settings_lambda_mu_scan.txt").write_text(
        "\n".join(settings) + "\n", encoding="utf-8"
    )
    (outdir / "manifest.json").write_text(
        json.dumps(
            {
                "workflow": "profess_tfvw_lambda_mu_bulk_cell_relaxation",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "lambda_tf_values": lambda_values,
                "mu_vw_values": mu_values,
                "initial_a0_A": args.initial_a0,
                "ecut_eV": args.ecut,
                "n_settings": len(settings),
                "profess_binary": str(profess_copy),
                "pseudopotential": str(pp_copy),
                "valid_solid_a0_range_A": [
                    args.min_solid_a0,
                    args.max_solid_a0,
                ],
                "max_stress_GPa": args.max_stress_gpa,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(settings)} PROFESS cell-relaxation cases at {outdir}")


if __name__ == "__main__":
    main()
