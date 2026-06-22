from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
TF_RE = re.compile(rf"KEDF-TF energy \(eV\):\s*({FLOAT_PATTERN})")
VW_RE = re.compile(rf"KEDF-VW energy \(eV\):\s*({FLOAT_PATTERN})")
TOTAL_RE = re.compile(rf"TOTAL energy \(eV\):\s*({FLOAT_PATTERN})")
SETTING_RE = re.compile(r"(tfvw_lam[0-9p]+_mu[0-9p]+)")

NUMERIC_MATRIX_FIELDS = {
    "pristine_total_energy_eV_per_atom": "Final pristine total energy (eV/atom)",
    "pristine_kedf_energy_eV_per_atom": "Final pristine KEDF energy (eV/atom)",
    "pristine_lattice_constant_A": "Final pristine lattice constant (A)",
    "vacancy_formation_energy_eV": "Vacancy formation energy (eV)",
    "kedf_vacancy_formation_energy_eV": "KEDF contribution to vacancy formation energy (eV)",
    "pristine_final_fmax_eV_A": "Pristine final fmax (eV/A)",
    "vacancy_final_fmax_eV_A": "Vacancy final fmax (eV/A)",
    "pristine_max_abs_stress_GPa": "Pristine maximum absolute stress (GPa)",
    "vacancy_max_abs_stress_GPa": "Vacancy maximum absolute stress (GPa)",
}


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_finite(value: object) -> bool:
    return math.isfinite(as_float(value))


def final_fmax(path: Path) -> float:
    if not path.exists():
        return math.nan
    value = math.nan
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            value = float(parts[-1])
        except ValueError:
            continue
    return value


def product(values: list[float]) -> float:
    result = 1.0
    for value in values:
        result *= float(value)
    return result


def cell_volume(result: dict[str, object], prefix: str) -> float:
    direct = result.get(f"{prefix}_volume_A3")
    if direct is not None:
        return as_float(direct)
    lengths = result.get(f"{prefix}_cell_lengths_A")
    angles = result.get(f"{prefix}_cell_angles_deg")
    if isinstance(lengths, list) and len(lengths) == 3:
        if not isinstance(angles, list) or len(angles) != 3:
            return product([float(value) for value in lengths])
        alpha, beta, gamma = np.radians(np.asarray(angles, dtype=float))
        factor = math.sqrt(
            max(
                0.0,
                1.0
                + 2.0 * math.cos(alpha) * math.cos(beta) * math.cos(gamma)
                - math.cos(alpha) ** 2
                - math.cos(beta) ** 2
                - math.cos(gamma) ** 2,
            )
        )
        return product([float(value) for value in lengths]) * factor
    return math.nan


def mean_cell_length(result: dict[str, object], prefix: str) -> float:
    lengths = result.get(f"{prefix}_cell_lengths_A")
    if isinstance(lengths, list) and len(lengths) == 3:
        return float(np.mean(np.asarray(lengths, dtype=float)))
    return math.nan


def conventional_lattice_constant(
    result: dict[str, object],
    prefix: str,
    manifest: dict[str, object],
) -> tuple[float, float]:
    repeat = result.get("conventional_repeat", manifest.get("conventional_repeat", [3, 3, 3]))
    if not isinstance(repeat, list) or len(repeat) != 3:
        repeat = [3, 3, 3]
    repeat_values = np.asarray(repeat, dtype=float)
    lengths = result.get(f"{prefix}_cell_lengths_A")
    mean_based = math.nan
    if isinstance(lengths, list) and len(lengths) == 3:
        mean_based = float(np.mean(np.asarray(lengths, dtype=float) / repeat_values))
    volume = cell_volume(result, prefix)
    volume_based = math.nan
    if math.isfinite(volume) and volume > 0.0:
        volume_based = float((volume / product(repeat_values.tolist())) ** (1.0 / 3.0))
    return volume_based, mean_based


def max_abs_stress(result: dict[str, object], prefix: str) -> float:
    stress = result.get(f"{prefix}_stress_GPa")
    if not isinstance(stress, list):
        return math.nan
    array = np.asarray(stress, dtype=float)
    return float(np.max(np.abs(array))) if array.size else math.nan


def settings(rootdir: Path) -> list[str]:
    setting_file = rootdir / "settings_weight_scan.txt"
    if setting_file.exists():
        output = []
        for line in setting_file.read_text(encoding="utf-8").splitlines():
            tokens = line.split()
            if tokens:
                output.append(tokens[-1])
        return output
    return sorted(path.name for path in (rootdir / "weight_scan").iterdir() if path.is_dir())


def status_for(case_dir: Path, result_path: Path) -> str:
    if result_path.exists() and result_path.stat().st_size:
        return "DONE"
    if list(case_dir.glob("RESCUE_TIMEOUT_*.txt")):
        return "TIMEOUT"
    if (case_dir / "RESCUE_FAILED.txt").exists():
        return "FAILED"
    return "MISSING"


def parse_scheduler_energy_blocks(log_root: Path) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    if not log_root.exists():
        return blocks
    sequence = 0
    for path in sorted(log_root.rglob("*.out")):
        tf_energy = math.nan
        vw_energy = math.nan
        current_setting = ""
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            setting_match = SETTING_RE.search(line)
            if setting_match and (
                "CASE_DIR=" in line
                or "setting=" in line
                or '"setting":' in line
            ):
                current_setting = setting_match.group(1)
            match = TF_RE.search(line)
            if match:
                tf_energy = float(match.group(1))
                continue
            match = VW_RE.search(line)
            if match:
                vw_energy = float(match.group(1))
                continue
            match = TOTAL_RE.search(line)
            if not match or not math.isfinite(tf_energy) or not math.isfinite(vw_energy):
                continue
            blocks.append(
                {
                    "total_energy_eV": float(match.group(1)),
                    "tf_energy_eV": tf_energy,
                    "vw_energy_eV": vw_energy,
                    "kedf_energy_eV": tf_energy + vw_energy,
                    "source_log": str(path),
                    "source_line": line_number,
                    "sequence": sequence,
                    "setting": current_setting,
                    "source_type": "remote_scheduler_log",
                }
            )
            sequence += 1
            tf_energy = math.nan
            vw_energy = math.nan
    return blocks


def match_energy_block(
    blocks: list[dict[str, object]],
    target_energy_eV: float,
    tolerance_eV: float,
    setting: str,
) -> dict[str, object]:
    if not math.isfinite(target_energy_eV):
        return {}
    matches = [
        block
        for block in blocks
        if str(block.get("setting", "")) == setting
        and abs(as_float(block["total_energy_eV"]) - target_energy_eV) <= tolerance_eV
    ]
    if not matches:
        return {}
    matches.sort(
        key=lambda block: (
            abs(as_float(block["total_energy_eV"]) - target_energy_eV),
            -int(block["sequence"]),
        )
    )
    selected = dict(matches[0])
    selected["match_count"] = len(matches)
    selected["total_energy_match_delta_eV"] = (
        as_float(selected["total_energy_eV"]) - target_energy_eV
    )
    selected["kedf_match_spread_eV"] = (
        max(as_float(block["kedf_energy_eV"]) for block in matches)
        - min(as_float(block["kedf_energy_eV"]) for block in matches)
    )
    return selected


def local_final_structure_kedf_fallback(
    case_dir: Path,
    manifest: dict[str, object],
    result: dict[str, object],
    pp_file: Path,
) -> dict[str, dict[str, object]]:
    cache_path = case_dir / "final_relaxed_kedf_scf_fallback.json"
    if cache_path.exists() and cache_path.stat().st_size:
        cached = read_json(cache_path)
        return {
            prefix: dict(cached.get(prefix, {}))
            for prefix in ("pristine", "vacancy")
        }

    from ase.io import read

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from app.dft_engine import evaluate_atoms_with_energy_components

    output: dict[str, dict[str, object]] = {}
    log_path = case_dir / "final_relaxed_kedf_scf_fallback.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
            for prefix in ("pristine", "vacancy"):
                structure_path = case_dir / f"{prefix}_vc_relaxed.vasp"
                atoms = read(structure_path)
                _, energy_eV, _, terms = evaluate_atoms_with_energy_components(
                    atoms,
                    pp_file=pp_file,
                    spacing=as_float(manifest.get("spacing_A"), 0.2),
                    kedf=str(manifest.get("kedf", "TFVW")),
                    xc=str(manifest.get("xc", "LDA")),
                    kedf_x=as_float(manifest.get("kedf_x")),
                    kedf_y=as_float(manifest.get("kedf_y")),
                    dftpy_outfile=str(
                        case_dir / f"{prefix}_final_relaxed_kedf_scf_fallback.out"
                    ),
                )
                target_energy = as_float(result.get(f"{prefix}_energy_eV"))
                output[prefix] = {
                    "total_energy_eV": energy_eV,
                    "tf_energy_eV": as_float(terms.get("KEDF-TF")),
                    "vw_energy_eV": as_float(terms.get("KEDF-VW")),
                    "kedf_energy_eV": as_float(terms.get("KEDF")),
                    "source_log": str(log_path),
                    "source_line": "",
                    "sequence": 0,
                    "setting": str(manifest.get("setting", case_dir.name)),
                    "source_type": "local_final_structure_scf_fallback",
                    "match_count": 1,
                    "total_energy_match_delta_eV": energy_eV - target_energy,
                    "kedf_match_spread_eV": 0.0,
                }
    cache_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clean_generated_outputs(tables_dir: Path, figures_dir: Path) -> None:
    table_patterns = (
        "matrix_*.csv",
        "professor_*.csv",
        "lambda_mu_*.csv",
        "incomplete_cases.csv",
        "quality_flagged_cases.csv",
        "final_kedf_*.csv",
        "top20_*.csv",
        "analysis_key_metrics.csv",
    )
    for pattern in table_patterns:
        for path in tables_dir.glob(pattern):
            path.unlink()
    for path in figures_dir.glob("heatmap_*.png"):
        path.unlink()


def matrix_rows(
    rows: list[dict[str, object]],
    lambdas: list[float],
    mus: list[float],
    field: str,
) -> list[dict[str, object]]:
    lookup = {
        (as_float(row["lambda_tf"]), as_float(row["mu_vw"])): row.get(field, math.nan)
        for row in rows
        if row["status"] == "DONE"
    }
    output = []
    for lambda_tf in lambdas:
        row: dict[str, object] = {"lambda/mu": lambda_tf}
        for mu_vw in mus:
            row[f"{mu_vw:.1f}"] = lookup.get((lambda_tf, mu_vw), math.nan)
        output.append(row)
    return output


def status_matrix_rows(
    rows: list[dict[str, object]],
    lambdas: list[float],
    mus: list[float],
) -> list[dict[str, object]]:
    lookup = {
        (as_float(row["lambda_tf"]), as_float(row["mu_vw"])): row["quality_status"]
        for row in rows
    }
    output = []
    for lambda_tf in lambdas:
        row: dict[str, object] = {"lambda/mu": lambda_tf}
        for mu_vw in mus:
            row[f"{mu_vw:.1f}"] = lookup.get((lambda_tf, mu_vw), "MISSING")
        output.append(row)
    return output


def write_three_panel_table(
    path: Path,
    matrices: dict[str, list[dict[str, object]]],
    mus: list[float],
) -> None:
    selected = [
        ("Final pristine total energy (eV/atom)", matrices["pristine_total_energy_eV_per_atom"]),
        ("Final pristine KEDF energy (eV/atom)", matrices["pristine_kedf_energy_eV_per_atom"]),
        ("Final pristine lattice constant (A)", matrices["pristine_lattice_constant_A"]),
    ]
    width = len(mus) + 1
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        title_row: list[object] = []
        header_row: list[object] = []
        for index, (title, _) in enumerate(selected):
            if index:
                title_row.append("")
                header_row.append("")
            title_row.extend([title] + [""] * (width - 1))
            header_row.extend(["lambda/mu"] + [f"{value:.1f}" for value in mus])
        writer.writerow(title_row)
        writer.writerow(header_row)
        for row_index in range(len(selected[0][1])):
            combined: list[object] = []
            for index, (_, matrix) in enumerate(selected):
                if index:
                    combined.append("")
                row = matrix[row_index]
                combined.extend([row["lambda/mu"]] + [row[f"{value:.1f}"] for value in mus])
            writer.writerow(combined)


def plot_heatmap(
    path: Path,
    rows: list[dict[str, object]],
    lambdas: list[float],
    mus: list[float],
    field: str,
    title: str,
) -> None:
    lookup = {
        (as_float(row["lambda_tf"]), as_float(row["mu_vw"])): as_float(row.get(field))
        for row in rows
        if row["status"] == "DONE"
    }
    matrix = np.asarray(
        [[lookup.get((lambda_tf, mu_vw), math.nan) for mu_vw in mus] for lambda_tf in lambdas]
    )
    fig, ax = plt.subplots(figsize=(8.2, 6.6), constrained_layout=True)
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(mus)), [f"{value:.1f}" for value in mus])
    ax.set_yticks(range(len(lambdas)), [f"{value:.1f}" for value in lambdas])
    ax.set_xlabel("mu_vW")
    ax.set_ylabel("lambda_TF")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def quality_status(
    status: str,
    force_ok: bool,
    stress_ok: bool,
    collapsed: bool,
    kedf_available: bool,
) -> str:
    if status != "DONE":
        return status
    flags = []
    if collapsed:
        flags.append("COLLAPSED")
    if not force_ok:
        flags.append("FORCE_FAIL")
    if not stress_ok:
        flags.append("STRESS_FAIL")
    if not kedf_available:
        flags.append("KEDF_MISSING")
    return "+".join(flags) if flags else "PASS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect a DFTpy TFvW lambda-mu vacancy vc-relax matrix."
    )
    parser.add_argument("--rootdir", required=True)
    parser.add_argument(
        "--scheduler-log-root",
        default=None,
        help="Scheduler log root. Defaults to ROOT/_scheduler_logs.",
    )
    parser.add_argument("--energy-match-tolerance-eV", type=float, default=2.0e-6)
    parser.add_argument("--stress-limit-GPa", type=float, default=0.5)
    parser.add_argument("--collapse-lattice-limit-A", type=float, default=3.0)
    parser.add_argument("--qe-reference-eV", type=float, default=0.6389122264065549)
    parser.add_argument("--lattice-reference-A", type=float, default=4.039848)
    parser.add_argument(
        "--fallback-pp",
        default=None,
        help=(
            "Optional local pseudopotential. If final KEDF terms are absent from "
            "scheduler logs, run SCF-only evaluations on the final relaxed VASP files."
        ),
    )
    parser.add_argument(
        "--fallback-total-energy-tolerance-eV",
        type=float,
        default=5.0e-3,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rootdir = Path(args.rootdir).expanduser().resolve()
    scan_root = rootdir / "weight_scan"
    if not scan_root.exists():
        raise FileNotFoundError(scan_root)
    log_root = (
        Path(args.scheduler_log_root).expanduser().resolve()
        if args.scheduler_log_root
        else rootdir / "_scheduler_logs"
    )
    energy_blocks = parse_scheduler_energy_blocks(log_root)
    fallback_pp = (
        Path(args.fallback_pp).expanduser().resolve() if args.fallback_pp else None
    )
    if fallback_pp is not None and not fallback_pp.exists():
        raise FileNotFoundError(fallback_pp)

    rows: list[dict[str, object]] = []
    for index, setting in enumerate(settings(rootdir)):
        case_dir = scan_root / setting
        manifest_path = case_dir / "point_manifest.json"
        result_path = case_dir / "result.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        result = read_json(result_path) if result_path.exists() and result_path.stat().st_size else {}
        status = status_for(case_dir, result_path)
        timeout_markers = sorted(path.name for path in case_dir.glob("RESCUE_TIMEOUT_*.txt"))
        n_pristine = as_float(
            result.get("pristine_n_atoms", manifest.get("pristine_n_atoms"))
        )
        n_vacancy = as_float(
            result.get("vacancy_n_atoms", manifest.get("vacancy_n_atoms"))
        )
        pristine_energy = as_float(result.get("pristine_energy_eV"))
        vacancy_energy = as_float(result.get("vacancy_energy_eV"))
        pristine_match = match_energy_block(
            energy_blocks, pristine_energy, args.energy_match_tolerance_eV, setting
        )
        vacancy_match = match_energy_block(
            energy_blocks, vacancy_energy, args.energy_match_tolerance_eV, setting
        )
        if (
            status == "DONE"
            and fallback_pp is not None
            and (not pristine_match or not vacancy_match)
        ):
            fallback = local_final_structure_kedf_fallback(
                case_dir, manifest, result, fallback_pp
            )
            if not pristine_match:
                candidate = fallback["pristine"]
                if (
                    abs(as_float(candidate.get("total_energy_match_delta_eV")))
                    <= args.fallback_total_energy_tolerance_eV
                ):
                    pristine_match = candidate
            if not vacancy_match:
                candidate = fallback["vacancy"]
                if (
                    abs(as_float(candidate.get("total_energy_match_delta_eV")))
                    <= args.fallback_total_energy_tolerance_eV
                ):
                    vacancy_match = candidate
        pristine_kedf = as_float(pristine_match.get("kedf_energy_eV"))
        vacancy_kedf = as_float(vacancy_match.get("kedf_energy_eV"))
        pristine_tf = as_float(pristine_match.get("tf_energy_eV"))
        vacancy_tf = as_float(vacancy_match.get("tf_energy_eV"))
        pristine_vw = as_float(pristine_match.get("vw_energy_eV"))
        vacancy_vw = as_float(vacancy_match.get("vw_energy_eV"))
        scale = n_vacancy / n_pristine if n_pristine > 0.0 else math.nan
        kedf_formation = (
            vacancy_kedf - scale * pristine_kedf
            if math.isfinite(vacancy_kedf) and math.isfinite(pristine_kedf)
            else math.nan
        )
        tf_formation = (
            vacancy_tf - scale * pristine_tf
            if math.isfinite(vacancy_tf) and math.isfinite(pristine_tf)
            else math.nan
        )
        vw_formation = (
            vacancy_vw - scale * pristine_vw
            if math.isfinite(vacancy_vw) and math.isfinite(pristine_vw)
            else math.nan
        )
        pristine_fmax = final_fmax(case_dir / "pristine_relax.log")
        vacancy_fmax = final_fmax(case_dir / "vacancy_relax.log")
        target_fmax = as_float(
            result.get("fmax_eV_per_A", manifest.get("fmax_eV_per_A"))
        )
        force_ok = (
            status == "DONE"
            and math.isfinite(pristine_fmax)
            and math.isfinite(vacancy_fmax)
            and pristine_fmax <= target_fmax
            and vacancy_fmax <= target_fmax
        )
        pristine_stress = max_abs_stress(result, "pristine")
        vacancy_stress = max_abs_stress(result, "vacancy")
        stress_ok = (
            status == "DONE"
            and math.isfinite(pristine_stress)
            and math.isfinite(vacancy_stress)
            and pristine_stress <= args.stress_limit_GPa
            and vacancy_stress <= args.stress_limit_GPa
        )
        pristine_a_volume, pristine_a_mean = conventional_lattice_constant(
            result, "pristine", manifest
        )
        vacancy_a_volume, vacancy_a_mean = conventional_lattice_constant(
            result, "vacancy", manifest
        )
        collapsed = status == "DONE" and (
            not math.isfinite(pristine_a_volume)
            or pristine_a_volume < args.collapse_lattice_limit_A
        )
        kedf_available = math.isfinite(pristine_kedf) and math.isfinite(vacancy_kedf)
        row = {
            "index": index,
            "setting": setting,
            "status": status,
            "quality_status": quality_status(
                status, force_ok, stress_ok, collapsed, kedf_available
            ),
            "physically_usable": bool(
                status == "DONE" and force_ok and stress_ok and not collapsed
            ),
            "lambda_tf": as_float(result.get("kedf_x", manifest.get("kedf_x"))),
            "mu_vw": as_float(result.get("kedf_y", manifest.get("kedf_y"))),
            "xc": result.get("xc", manifest.get("xc", "")),
            "kedf": result.get("kedf", manifest.get("kedf", "")),
            "N_pristine": n_pristine,
            "N_vacancy": n_vacancy,
            "vacancy_concentration_percent": result.get(
                "vacancy_concentration_percent", math.nan
            ),
            "spacing_A": result.get("spacing_A", manifest.get("spacing_A", math.nan)),
            "target_fmax_eV_A": target_fmax,
            "pristine_energy_eV": pristine_energy,
            "pristine_total_energy_eV_per_atom": (
                pristine_energy / n_pristine
                if math.isfinite(pristine_energy) and n_pristine > 0.0
                else math.nan
            ),
            "vacancy_energy_eV": vacancy_energy,
            "vacancy_total_energy_eV_per_atom": (
                vacancy_energy / n_vacancy
                if math.isfinite(vacancy_energy) and n_vacancy > 0.0
                else math.nan
            ),
            "vacancy_formation_energy_eV": result.get(
                "vacancy_formation_energy_eV", math.nan
            ),
            "pristine_kedf_energy_eV": pristine_kedf,
            "pristine_kedf_energy_eV_per_atom": (
                pristine_kedf / n_pristine
                if math.isfinite(pristine_kedf) and n_pristine > 0.0
                else math.nan
            ),
            "vacancy_kedf_energy_eV": vacancy_kedf,
            "vacancy_kedf_energy_eV_per_atom": (
                vacancy_kedf / n_vacancy
                if math.isfinite(vacancy_kedf) and n_vacancy > 0.0
                else math.nan
            ),
            "pristine_kedf_tf_energy_eV": pristine_tf,
            "pristine_kedf_vw_energy_eV": pristine_vw,
            "vacancy_kedf_tf_energy_eV": vacancy_tf,
            "vacancy_kedf_vw_energy_eV": vacancy_vw,
            "kedf_vacancy_formation_energy_eV": kedf_formation,
            "kedf_tf_vacancy_formation_energy_eV": tf_formation,
            "kedf_vw_vacancy_formation_energy_eV": vw_formation,
            "pristine_lattice_constant_A": pristine_a_volume,
            "pristine_lattice_constant_mean_length_A": pristine_a_mean,
            "vacancy_lattice_constant_A": vacancy_a_volume,
            "vacancy_lattice_constant_mean_length_A": vacancy_a_mean,
            "pristine_final_fmax_eV_A": pristine_fmax,
            "vacancy_final_fmax_eV_A": vacancy_fmax,
            "force_converged": force_ok,
            "pristine_max_abs_stress_GPa": pristine_stress,
            "vacancy_max_abs_stress_GPa": vacancy_stress,
            "stress_converged": stress_ok,
            "collapsed_pristine_cell": collapsed,
            "pristine_volume_A3": cell_volume(result, "pristine"),
            "vacancy_volume_A3": cell_volume(result, "vacancy"),
            "pristine_cell_lengths_A": json.dumps(
                result.get("pristine_cell_lengths_A", [])
            ),
            "vacancy_cell_lengths_A": json.dumps(
                result.get("vacancy_cell_lengths_A", [])
            ),
            "pristine_kedf_source_log": pristine_match.get("source_log", ""),
            "pristine_kedf_source_type": pristine_match.get("source_type", ""),
            "pristine_kedf_source_line": pristine_match.get("source_line", ""),
            "pristine_kedf_match_count": pristine_match.get("match_count", 0),
            "pristine_kedf_total_match_delta_eV": pristine_match.get(
                "total_energy_match_delta_eV", math.nan
            ),
            "pristine_kedf_match_spread_eV": pristine_match.get(
                "kedf_match_spread_eV", math.nan
            ),
            "vacancy_kedf_source_log": vacancy_match.get("source_log", ""),
            "vacancy_kedf_source_type": vacancy_match.get("source_type", ""),
            "vacancy_kedf_source_line": vacancy_match.get("source_line", ""),
            "vacancy_kedf_match_count": vacancy_match.get("match_count", 0),
            "vacancy_kedf_total_match_delta_eV": vacancy_match.get(
                "total_energy_match_delta_eV", math.nan
            ),
            "vacancy_kedf_match_spread_eV": vacancy_match.get(
                "kedf_match_spread_eV", math.nan
            ),
            "timeout_marker_count": len(timeout_markers),
            "timeout_markers": ";".join(timeout_markers),
            "failed_marker": (case_dir / "RESCUE_FAILED.txt").exists(),
            "case_dir": str(case_dir),
        }
        rows.append(row)

    rows.sort(key=lambda row: int(row["index"]))
    lambdas = sorted(
        {as_float(row["lambda_tf"]) for row in rows if is_finite(row["lambda_tf"])}
    )
    mus = sorted({as_float(row["mu_vw"]) for row in rows if is_finite(row["mu_vw"])})
    analysis = rootdir / "analysis"
    tables_dir = analysis / "tables"
    figures_dir = analysis / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    clean_generated_outputs(tables_dir, figures_dir)
    write_csv(tables_dir / "lambda_mu_vacancy_long_summary.csv", rows)

    matrices: dict[str, list[dict[str, object]]] = {}
    for field, title in NUMERIC_MATRIX_FIELDS.items():
        matrix = matrix_rows(rows, lambdas, mus, field)
        matrices[field] = matrix
        write_csv(tables_dir / f"matrix_{field}.csv", matrix)
        plot_heatmap(
            figures_dir / f"heatmap_{field}.png",
            rows,
            lambdas,
            mus,
            field,
            title,
        )
    write_csv(
        tables_dir / "matrix_quality_status.csv",
        status_matrix_rows(rows, lambdas, mus),
    )
    write_three_panel_table(
        tables_dir / "professor_three_panel_final_vcrelax.csv",
        matrices,
        mus,
    )

    completed_rows = [row for row in rows if row["status"] == "DONE"]
    incomplete_rows = [row for row in rows if row["status"] != "DONE"]
    issue_rows = [
        row
        for row in completed_rows
        if not bool(row["force_converged"])
        or not bool(row["stress_converged"])
        or bool(row["collapsed_pristine_cell"])
    ]
    kedf_missing_rows = [
        row
        for row in completed_rows
        if not is_finite(row["pristine_kedf_energy_eV"])
        or not is_finite(row["vacancy_kedf_energy_eV"])
    ]
    fallback_rows = [
        row
        for row in completed_rows
        if row["pristine_kedf_source_type"] == "local_final_structure_scf_fallback"
        or row["vacancy_kedf_source_type"] == "local_final_structure_scf_fallback"
    ]
    scheduler_kedf_rows = [
        row
        for row in completed_rows
        if row["pristine_kedf_source_type"] == "remote_scheduler_log"
        and row["vacancy_kedf_source_type"] == "remote_scheduler_log"
    ]
    closest_rows = sorted(
        completed_rows,
        key=lambda row: abs(
            as_float(row["vacancy_formation_energy_eV"]) - args.qe_reference_eV
        ),
    )
    for row in closest_rows:
        row["abs_difference_from_QE_reference_eV"] = abs(
            as_float(row["vacancy_formation_energy_eV"]) - args.qe_reference_eV
        )
        row["abs_difference_from_lattice_reference_A"] = abs(
            as_float(row["pristine_lattice_constant_A"]) - args.lattice_reference_A
        )
    closest_lattice_rows = sorted(
        completed_rows,
        key=lambda row: as_float(row["abs_difference_from_lattice_reference_A"]),
    )
    joint_close_count = sum(
        as_float(row["abs_difference_from_QE_reference_eV"]) <= 0.1
        and as_float(row["abs_difference_from_lattice_reference_A"]) <= 0.05
        and bool(row["physically_usable"])
        for row in completed_rows
    )
    write_csv(tables_dir / "incomplete_cases.csv", incomplete_rows)
    write_csv(tables_dir / "quality_flagged_cases.csv", issue_rows)
    write_csv(tables_dir / "final_kedf_missing_cases.csv", kedf_missing_rows)
    write_csv(tables_dir / "final_kedf_local_fallback_cases.csv", fallback_rows)
    write_csv(tables_dir / "top20_closest_to_QE_reference.csv", closest_rows[:20])
    write_csv(
        tables_dir / "top20_closest_to_lattice_reference.csv",
        closest_lattice_rows[:20],
    )

    counts = Counter(str(row["status"]) for row in rows)
    quality_counts = Counter(str(row["quality_status"]) for row in rows)
    force_pass = sum(bool(row["force_converged"]) for row in completed_rows)
    stress_pass = sum(bool(row["stress_converged"]) for row in completed_rows)
    physical_pass = sum(bool(row["physically_usable"]) for row in completed_rows)
    collapsed_count = sum(bool(row["collapsed_pristine_cell"]) for row in completed_rows)
    ef_reasonable = sum(
        0.0 <= as_float(row["vacancy_formation_energy_eV"]) <= 1.0
        for row in completed_rows
    )
    ef_high = sum(
        as_float(row["vacancy_formation_energy_eV"]) > 5.0 for row in completed_rows
    )
    write_csv(
        tables_dir / "analysis_key_metrics.csv",
        [
            {"metric": "expected_points", "value": len(rows), "unit": "points"},
            {"metric": "completed_points", "value": len(completed_rows), "unit": "points"},
            {"metric": "timeout_points", "value": len(incomplete_rows), "unit": "points"},
            {
                "metric": "remote_kedf_complete",
                "value": len(scheduler_kedf_rows),
                "unit": "points",
            },
            {
                "metric": "local_kedf_fallback",
                "value": len(fallback_rows),
                "unit": "points",
            },
            {"metric": "force_pass", "value": force_pass, "unit": "points"},
            {"metric": "stress_pass", "value": stress_pass, "unit": "points"},
            {
                "metric": "numerically_usable_noncollapsed",
                "value": physical_pass,
                "unit": "points",
            },
            {"metric": "collapsed_cells", "value": collapsed_count, "unit": "points"},
            {"metric": "Ef_0_to_1_eV", "value": ef_reasonable, "unit": "points"},
            {"metric": "Ef_above_5_eV", "value": ef_high, "unit": "points"},
            {
                "metric": "joint_close_Ef_0p1eV_a0_0p05A",
                "value": joint_close_count,
                "unit": "points",
            },
            {
                "metric": "QE_reference_Ef",
                "value": args.qe_reference_eV,
                "unit": "eV",
            },
            {
                "metric": "lattice_reference",
                "value": args.lattice_reference_A,
                "unit": "A",
            },
        ],
    )
    note = [
        "# DFTpy TFvW Lambda-Mu Vacancy Matrix Audit",
        "",
        "Definitions:",
        "",
        "- Professor Total table: final relaxed pristine total energy per atom.",
        "- Professor KEDF table: final relaxed pristine KEDF kinetic energy per atom.",
        "- Professor lattice table: final relaxed pristine conventional fcc lattice constant.",
        "- Vacancy formation energy is reported separately and is not labelled total energy.",
        "- KEDF vacancy contribution = K_vac - (N_vac/N_pristine) K_pristine.",
        "",
        f"- Expected settings: {len(rows)}",
        f"- DONE: {counts['DONE']}",
        f"- TIMEOUT: {counts['TIMEOUT']}",
        f"- FAILED: {counts['FAILED']}",
        f"- MISSING: {counts['MISSING']}",
        f"- Final KEDF recovered from scheduler logs: {len(scheduler_kedf_rows)}/{len(completed_rows)}",
        f"- Final KEDF locally reconstructed from final VASP structures: {len(fallback_rows)}/{len(completed_rows)}",
        f"- Force pass (both structures): {force_pass}/{len(completed_rows)}",
        f"- Stress pass <= {args.stress_limit_GPa:g} GPa (both structures): {stress_pass}/{len(completed_rows)}",
        f"- Force/stress usable and non-collapsed: {physical_pass}/{len(completed_rows)}",
        f"- Collapsed pristine cells (a0 < {args.collapse_lattice_limit_A:g} A): {collapsed_count}/{len(completed_rows)}",
        f"- Vacancy formation energy in 0-1 eV: {ef_reasonable}/{len(completed_rows)}",
        f"- Vacancy formation energy > 5 eV: {ef_high}/{len(completed_rows)}",
        (
            "- Points simultaneously within 0.1 eV of the QE vacancy reference "
            f"and 0.05 A of the lattice reference: {joint_close_count}/{len(completed_rows)}"
        ),
        f"- Parsed scheduler energy blocks: {len(energy_blocks)}",
        f"- KEDF total-energy match tolerance: {args.energy_match_tolerance_eV:g} eV",
        "",
        "Quality status counts:",
        "",
    ]
    note.extend(f"- {name}: {count}" for name, count in sorted(quality_counts.items()))
    note.extend(["", "Incomplete settings:", ""])
    note.extend(
        f"- index {int(row['index']):02d}: {row['setting']} ({row['status']})"
        for row in incomplete_rows
    )
    note.extend(["", "Completed settings without final KEDF log match:", ""])
    note.extend(f"- {row['setting']}" for row in kedf_missing_rows)
    note.extend(["", "Closest completed points to the QE reference:", ""])
    note.extend(
        "- lambda={lambda_tf:g}, mu={mu_vw:g}: Ef={ef:.6f} eV, "
        "a0={a0:.6f} A, quality={quality}".format(
            lambda_tf=as_float(row["lambda_tf"]),
            mu_vw=as_float(row["mu_vw"]),
            ef=as_float(row["vacancy_formation_energy_eV"]),
            a0=as_float(row["pristine_lattice_constant_A"]),
            quality=row["quality_status"],
        )
        for row in closest_rows[:5]
    )
    note.extend(["", "Closest completed points to the lattice reference:", ""])
    note.extend(
        "- lambda={lambda_tf:g}, mu={mu_vw:g}: a0={a0:.6f} A, "
        "Ef={ef:.6f} eV, quality={quality}".format(
            lambda_tf=as_float(row["lambda_tf"]),
            mu_vw=as_float(row["mu_vw"]),
            a0=as_float(row["pristine_lattice_constant_A"]),
            ef=as_float(row["vacancy_formation_energy_eV"]),
            quality=row["quality_status"],
        )
        for row in closest_lattice_rows[:5]
    )
    note.extend(
        [
            "",
            "Important:",
            "",
            "- SCF_3tables_for_professor_Total_KEDF_lattice_constant.csv is an older fixed-structure SCF table.",
            "- Use professor_three_panel_final_vcrelax.csv for the final relaxed 96-point snapshot.",
            "- Always send matrix_quality_status.csv with the three-panel table.",
        ]
    )
    (analysis / "COMPLETION_AUDIT.md").write_text("\n".join(note) + "\n", encoding="utf-8")

    print("============================================================")
    print("DFTpy TFvW lambda-mu vacancy matrix collected")
    print("============================================================")
    print(f"Root        : {rootdir}")
    print(f"Expected    : {len(rows)}")
    for status in ("DONE", "TIMEOUT", "FAILED", "MISSING"):
        print(f"{status:11s} : {counts[status]}")
    print(f"KEDF remote : {len(scheduler_kedf_rows)}/{len(completed_rows)}")
    print(f"KEDF fallback: {len(fallback_rows)}/{len(completed_rows)}")
    print(f"KEDF missing: {len(kedf_missing_rows)}/{len(completed_rows)}")
    print(f"Force pass  : {force_pass}/{len(completed_rows)}")
    print(f"Stress pass : {stress_pass}/{len(completed_rows)}")
    print(f"Usable      : {physical_pass}/{len(completed_rows)}")
    print(f"Collapsed   : {collapsed_count}/{len(completed_rows)}")
    print(f"Summary     : {tables_dir / 'lambda_mu_vacancy_long_summary.csv'}")
    print(f"3-panel     : {tables_dir / 'professor_three_panel_final_vcrelax.csv'}")


if __name__ == "__main__":
    main()
