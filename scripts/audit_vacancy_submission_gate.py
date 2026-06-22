from __future__ import annotations

import argparse
import configparser
import json
import math
import re
from pathlib import Path

import numpy as np
from ase.io import read


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual: float, expected: float, tolerance: float = 1.0e-8) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)


def minimum_distance(atoms) -> float:
    distances = atoms.get_all_distances(mic=True)
    distances[distances < 1.0e-12] = np.inf
    return float(np.min(distances))


def read_ini(path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(path, encoding="utf-8")
    return config


def audit_dftpy(root: Path, errors: list[str], warnings: list[str]) -> dict[str, object]:
    manifest_path = root / "manifest.json"
    settings_path = root / "settings_weight_scan.txt"
    submit_candidates = [
        root / "02_submission_scripts" / "active" / "submit_dftpy_lambda_mu_fine_one_case_ct56.sh",
        root / "submit_dftpy_lambda_mu_fine_array.sh",
        root / "submit_dftpy_lambda_mu_fine_workers.sh",
    ]
    if not manifest_path.exists():
        errors.append(f"DFTpy missing manifest: {manifest_path}")
        return {}
    if not settings_path.exists():
        errors.append(f"DFTpy missing settings list: {settings_path}")
        return {}

    workflow = load_json(manifest_path)
    settings = [line.strip() for line in settings_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(settings) != len(set(settings)):
        errors.append("DFTpy settings list contains duplicate names")
    if len(settings) != int(workflow.get("point_count", -1)):
        errors.append(f"DFTpy settings count {len(settings)} != manifest point_count {workflow.get('point_count')}")

    submit = next((path for path in submit_candidates if path.exists()), None)
    if submit:
        text = submit.read_text(encoding="utf-8")
        required = ["run_dftpy_vcrelax_vacancy_one.py", "--ase-optimizer BFGS", "OMP_NUM_THREADS=1"]
        for token in required:
            if token not in text:
                errors.append(f"DFTpy submit script missing: {token}")
        if "#SBATCH --cpus-per-task=1" not in text:
            errors.append("DFTpy submit script must request one CPU per serial case")
    else:
        warnings.append("DFTpy submit script not found inside series root")

    lambda_values: set[float] = set()
    mu_values: set[float] = set()
    completed = 0
    for setting in settings:
        case = root / "weight_scan" / setting
        required_files = [
            "point_manifest.json",
            "README_CASE.txt",
            "pristine_raw.vasp",
            "vacancy_start.vasp",
            "dftpy_pristine_input.ini",
            "dftpy_vacancy_input.ini",
        ]
        for name in required_files:
            if not (case / name).exists():
                errors.append(f"{setting}: missing {name}")
        if any(not (case / name).exists() for name in required_files[:1] + required_files[2:]):
            continue

        point = load_json(case / "point_manifest.json")
        pristine = read(case / "pristine_raw.vasp")
        vacancy = read(case / "vacancy_start.vasp")
        expected_np = int(point["pristine_n_atoms"])
        expected_nv = int(point["vacancy_n_atoms"])
        if len(pristine) != expected_np or len(vacancy) != expected_nv or expected_np - expected_nv != 1:
            errors.append(f"{setting}: atom counts are not the expected single vacancy")
        if min(pristine.cell.lengths()) <= 10.0:
            errors.append(f"{setting}: a cell side is not greater than 10 A")
        if max(abs(float(angle) - 90.0) for angle in pristine.cell.angles()) > 1.0e-5:
            errors.append(f"{setting}: starting cell is not orthogonal")
        if minimum_distance(pristine) < 2.0 or minimum_distance(vacancy) < 2.0:
            errors.append(f"{setting}: suspiciously short Al-Al distance")
        removed_scaled = np.asarray(point.get("removed_atom_scaled", []), dtype=float)
        if removed_scaled.size != 3 or np.linalg.norm(removed_scaled - 0.5) > 1.0e-6:
            errors.append(f"{setting}: vacancy site is not centered")

        lambda_tf = float(point["kedf_x"])
        mu_vw = float(point["kedf_y"])
        lambda_values.add(lambda_tf)
        mu_values.add(mu_vw)
        pp_name = Path(str(point["pp_file"])).name
        if not (case / pp_name).exists():
            errors.append(f"{setting}: case-local pseudopotential {pp_name} is missing")

        for role, structure_name in (("pristine", "pristine_raw.vasp"), ("vacancy", "vacancy_start.vasp")):
            ini = read_ini(case / f"dftpy_{role}_input.ini")
            expected = {
                ("JOB", "task"): "Optdensity",
                ("JOB", "calctype"): "Energy Force Stress",
                ("PP", "al"): pp_name,
                ("CELL", "cellfile"): structure_name,
                ("EXC", "xc"): "LDA",
                ("KEDF", "kedf"): "TFVW",
            }
            for (section, option), value in expected.items():
                if ini.get(section, option, fallback="").strip() != value:
                    errors.append(f"{setting}: {role} INI {section}.{option} mismatch")
            if not close(ini.getfloat("GRID", "spacing"), float(point["spacing_A"])):
                errors.append(f"{setting}: {role} INI spacing mismatch")
            if not close(ini.getfloat("KEDF", "x"), lambda_tf) or not close(ini.getfloat("KEDF", "y"), mu_vw):
                errors.append(f"{setting}: {role} INI lambda/mu mismatch")

        result = case / "result.json"
        if result.exists() and result.stat().st_size:
            completed += 1

    return {
        "cases": len(settings),
        "completed_results": completed,
        "lambda_values": sorted(lambda_values),
        "mu_values": sorted(mu_values),
    }


def parse_qe_nat(text: str) -> int | None:
    match = re.search(r"\bnat\s*=\s*(\d+)", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def audit_qe(root: Path, require_outputs: bool, errors: list[str], warnings: list[str]) -> dict[str, object]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        errors.append(f"QE missing manifest: {manifest_path}")
        return {}
    workflow = load_json(manifest_path)
    cases = 0
    completed = 0
    for pristine_input in sorted(root.rglob("pristine_vcrelax/vc-relax.in")):
        case = pristine_input.parent.parent
        vacancy_input = case / "vacancy_vcrelax" / "vc-relax.in"
        if not vacancy_input.exists():
            errors.append(f"{case}: missing vacancy vc-relax input")
            continue
        cases += 1
        for path, expected_nat in (
            (pristine_input, int(workflow["pristine_n_atoms"])),
            (vacancy_input, int(workflow["vacancy_n_atoms"])),
        ):
            text = path.read_text(encoding="utf-8")
            if "calculation = 'vc-relax'" not in text:
                errors.append(f"{path}: calculation is not vc-relax")
            if parse_qe_nat(text) != expected_nat:
                errors.append(f"{path}: nat mismatch")
            for token in ("ion_dynamics = 'bfgs'", "cell_dynamics = 'bfgs'", "Al_PAW_PBE.UPF"):
                if token not in text:
                    errors.append(f"{path}: missing {token}")
        outputs = [
            case / "pristine_vcrelax" / "vc-relax.out",
            case / "vacancy_vcrelax" / "vc-relax.out",
        ]
        done = all(path.exists() and "JOB DONE" in path.read_text(errors="ignore") for path in outputs)
        if done:
            completed += 1
        elif require_outputs:
            errors.append(f"{case}: completed QE output pair is missing")
        else:
            warnings.append(f"{case}: outputs not complete yet")
    pseudo = root / "psp" / "Al_PAW_PBE.UPF"
    if not pseudo.exists() or not pseudo.stat().st_size:
        errors.append(f"QE pseudopotential missing: {pseudo}")
    return {"cases": cases, "completed_output_pairs": completed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-fast structure/input gate for DFTpy and QE vacancy workflows.")
    parser.add_argument("--dftpy-root", default="")
    parser.add_argument("--qe-root", default="")
    parser.add_argument("--require-qe-outputs", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    if not args.dftpy_root and not args.qe_root:
        parser.error("Provide --dftpy-root and/or --qe-root")

    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, object] = {}
    if args.dftpy_root:
        report["dftpy"] = audit_dftpy(Path(args.dftpy_root).expanduser().resolve(), errors, warnings)
    if args.qe_root:
        report["qe"] = audit_qe(
            Path(args.qe_root).expanduser().resolve(), args.require_qe_outputs, errors, warnings
        )
    report["errors"] = errors
    report["warnings"] = warnings
    report["status"] = "PASS" if not errors else "FAIL"
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).expanduser().resolve().write_text(text + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
