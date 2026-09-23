"""Evidence checks shared by collection, analysis and the teaching notebooks.

``qualified`` means the recorded calculation passes the checks below. It is not
a claim of electronic-density convergence, finite-size convergence, model
validation, or thesis acceptance. *_dftpy.out is a saved calculator energy/
stress summary, not a complete electronic-density iteration trace.
Source files are never modified. Missing evidence is never silently accepted.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from ase.io import read

ENERGY_DEFINITION = "E_defect-(N_defect/N_pristine)*E_pristine;total_eV"
DISTANCE_CONVENTION = "initial minimum-image distance under PBC"
ENERGY_ATOL = 1e-5  # eV; ASE logs print six decimal places.


def as_float(value: object) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return math.nan


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def optimizer_last_record(path: Path) -> dict:
    """Read the final ASE optimizer record (including LBFGS and SciPy variants).

    These logs are emitted by the optimizer acting on FrechetCellFilter and
    therefore include cell degrees of freedom. Atomic-only result fmax is not
    a substitute. Repeated headers/restarts are allowed; the final record wins.
    """
    record = {"optimizer": "", "step": None, "energy_eV": math.nan, "fmax_eV_A": math.nan}
    if not path.is_file():
        return record
    for line in path.read_text(errors="replace").splitlines():
        match = re.match(r"^\s*([A-Za-z][A-Za-z0-9_]*):\s+(\d+)\s+\S+\s+(\S+)\s+(\S+)\s*$", line)
        if match:
            record = {"optimizer": match[1], "step": int(match[2]),
                      "energy_eV": as_float(match[3]), "fmax_eV_A": as_float(match[4])}
    return record


def output_energy(path: Path) -> float:
    value = math.nan
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            match = re.search(r"total energy \(eV\)\s*:\s*(\S+)", line, re.I)
            if match:
                value = as_float(match[1])
    return value


def _choose(case: Path, primary: str, legacy: str | None = None) -> Path:
    path = case / primary
    return case / legacy if not path.exists() and legacy and (case / legacy).exists() else path


def _axis_label(vector: np.ndarray) -> str:
    # Preserve a signed crystal axis up to global reversal, not absolute values
    # component-by-component. In particular [110] and [1 -1 0] stay distinct.
    from fractions import Fraction
    parts = [Fraction(float(x)).limit_denominator(96) for x in vector]
    scale = math.lcm(*(p.denominator for p in parts))
    indices = [int(p * scale) for p in parts]
    divisor = math.gcd(*indices)
    if not divisor:
        return "unknown"
    indices = [x // divisor for x in indices]
    if next(x for x in indices if x) < 0:
        indices = [-x for x in indices]
    return "[" + ("".join(map(str, indices)) if all(0 <= x < 10 for x in indices) else " ".join(map(str, indices))) + "]"


def qualify_case(case: Path) -> dict:
    """Return explicit state, evidence, and reasons for a single vacancy case."""
    case = Path(case)
    checked = {"status": "missing", "qualified": False, "qualification_reasons": "",
               "thesis_acceptance": "not_assessed", "electronic_convergence_status": "not_independently_verified",
               "energy_definition": ENERGY_DEFINITION,
               "distance_convention": DISTANCE_CONVENTION, "pair_direction_verified": "unknown",
               "pair_distance_verified_A": math.nan, "initial_geometry_sha256": "",
               "initial_pristine_geometry_sha256": "", "reference_consistency_verified": False,
               "pseudopotential_sha256": "", "calculation_code": "DFTpy",
               "pristine_combined_fmax_eV_A": math.nan,
               "vacancy_combined_fmax_eV_A": math.nan, "Ef_recomputed_eV": math.nan}
    missing, failed, unconverged = [], [], []
    try:
        manifest = load_json(case / "point_manifest.json")
    except (OSError, ValueError) as exc:
        checked["status"] = "missing" if not (case / "point_manifest.json").exists() else "failed"
        checked["qualification_reasons"] = f"manifest unavailable: {exc}"
        return checked
    if not (case / "result.json").exists():
        checked["qualification_reasons"] = "result.json missing; no completed result"
        return checked
    try:
        result = load_json(case / "result.json")
    except (OSError, ValueError) as exc:
        checked.update(status="failed", qualification_reasons=f"invalid result.json: {exc}")
        return checked

    def close(a, b, reason, atol=ENERGY_ATOL):
        if not math.isfinite(as_float(a)) or not math.isfinite(as_float(b)):
            failed.append(f"non-finite {reason}")
        elif not math.isclose(float(a), float(b), rel_tol=0, abs_tol=atol):
            failed.append(f"inconsistent {reason}")

    n, nd = as_float(manifest.get("pristine_n_atoms")), as_float(manifest.get("vacancy_n_atoms"))
    if not (math.isfinite(n) and math.isfinite(nd) and n > nd > 0 and n.is_integer() and nd.is_integer()):
        checked.update(status="failed", qualification_reasons="invalid pristine/defect atom counts")
        return checked
    is_pair = manifest.get("scan_type") == "pair" or n - nd == 2
    if is_pair and n - nd != 2:
        failed.append("pair case must remove exactly two atoms")
    defect = "divacancy" if is_pair else "vacancy"
    for key in ("pristine_n_atoms", "vacancy_n_atoms", "spacing_A", "kedf_x", "kedf_y", "fmax_eV_per_A"):
        if key not in manifest or key not in result:
            missing.append(f"{key} missing from manifest/result")
        else:
            close(manifest[key], result[key], key, 1e-10)
    for key in ("xc", "kedf", "cell_basis"):
        if not manifest.get(key) or not result.get(key):
            missing.append(f"{key} missing from manifest/result")
        elif str(manifest[key]).upper() != str(result[key]).upper():
            failed.append(f"inconsistent {key}")
    if result.get("relaxation_mode") != "full_atom_and_cell_relaxation_vc_relax_equivalent":
        missing.append("full atom-and-cell relaxation provenance unavailable")
    pressure = as_float(result.get("target_pressure_GPa"))
    if not math.isfinite(pressure):
        missing.append("target pressure unavailable")
    elif abs(pressure) > 1e-12:
        failed.append("nonzero pressure needs an explicitly validated enthalpy definition")
    target = as_float(manifest.get("fmax_eV_per_A"))
    if not math.isfinite(target) or target <= 0:
        failed.append("invalid convergence threshold")
    ep, ed, ef = (as_float(result.get(k)) for k in
                  ("pristine_energy_eV", "vacancy_energy_eV", "vacancy_formation_energy_eV"))
    checked["Ef_recomputed_eV"] = ed - nd / n * ep
    close(ef, checked["Ef_recomputed_eV"], "total defect formation energy")
    if "divacancy_energy_eV" in result:
        close(ed, result["divacancy_energy_eV"], "defect energy aliases")

    configurations = []
    for label in ("pristine", defect):
        config_path = _choose(case, f"dftpy_{label}_calculator_config.json", "dftpy_vacancy_calculator_config.json" if label == "divacancy" else None)
        if not config_path.exists():
            missing.append(f"{label} calculator/reference provenance missing")
            continue
        try:
            config = load_json(config_path)
            calc, relax = config["dftpy_calculator"], config["ase_full_relaxation"]
            configurations.append(calc)
            if relax.get("cell_filter") != "FrechetCellFilter":
                failed.append(f"{label} optimizer log is not a documented FrechetCellFilter log")
            close(relax.get("fmax_eV_A"), target, f"{label} config convergence threshold", 1e-10)
            close(relax.get("scalar_pressure_GPa"), pressure, f"{label} config pressure", 1e-10)
            close(calc.get("GRID", {}).get("spacing"), manifest.get("spacing_A"), f"{label} config spacing", 1e-10)
            for config_key, manifest_key in (("x", "kedf_x"), ("y", "kedf_y")):
                close(calc.get("KEDF", {}).get(config_key), manifest.get(manifest_key), f"{label} config {manifest_key}", 1e-10)
            if str(calc.get("EXC", {}).get("xc", "")).upper() != str(manifest.get("xc", "")).upper():
                failed.append(f"{label} config XC mismatch")
            if str(calc.get("KEDF", {}).get("kedf", "")).upper() != str(manifest.get("kedf", "")).upper():
                failed.append(f"{label} config KEDF mismatch")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failed.append(f"{label} calculator config invalid: {exc}")
    if len(configurations) == 2:
        if configurations[0] != configurations[1]:
            failed.append("pristine/defect calculator settings or pseudopotentials differ")
        else:
            checked["reference_consistency_verified"] = True
            pp_names = configurations[0].get("PP", {})
            if not pp_names:
                missing.append("pseudopotential identity unavailable")
            else:
                hashes = {}
                for element, name in pp_names.items():
                    # Historical absolute paths are provenance; resolve a
                    # portable package copy without substituting another PP.
                    pp = Path(str(name))
                    candidates = [Path(str(configurations[0].get("PATH", {}).get("pppath", ""))) / pp,
                                  case / pp.name, case.parent.parent / pp.name,
                                  case.parent.parent / "pseudopotentials" / pp.name]
                    existing = next((p for p in candidates if p.is_file()), None)
                    if existing is None:
                        missing.append(f"{element} pseudopotential bytes unavailable for SHA-256")
                    else:
                        hashes[element] = hashlib.sha256(existing.read_bytes()).hexdigest()
                if len(hashes) == len(pp_names):
                    checked["pseudopotential_sha256"] = json.dumps(hashes, sort_keys=True)

    for label, energy, expected_count in (("pristine", ep, n), (defect, ed, nd)):
        legacy = "vacancy" if label == "divacancy" else label
        log = _choose(case, f"{label}_relax.log", f"{legacy}_relax.log")
        record = optimizer_last_record(log)
        name = "pristine" if label == "pristine" else "vacancy"
        checked[f"{name}_combined_fmax_eV_A"] = record["fmax_eV_A"]
        checked[f"{name}_optimizer"] = record["optimizer"]
        checked[f"{name}_optimizer_step"] = record["step"]
        exact_key = 'pristine_relaxation_evidence' if label == 'pristine' else 'defect_relaxation_evidence'
        if exact_key in result:
            exact = result[exact_key]
            if not isinstance(exact, dict):
                failed.append(f'{label} exact convergence metadata malformed')
            else:
                close(exact.get('target_fmax_eV_A'), target, f'{label} exact force target', 1e-12)
                if exact.get('optimizer_converged') is not True:
                    unconverged.append(f'{label} optimizer did not report convergence')
                precise_force = as_float(exact.get('combined_filter_fmax_eV_A'))
                if not math.isfinite(precise_force) or precise_force < 0:
                    failed.append(f'{label} exact combined force unavailable or invalid')
                elif precise_force >= target:
                    unconverged.append(f'{label} exact combined force exceeds target')
        if not record["optimizer"]:
            missing.append(f"{label} combined-filter optimizer record missing")
        else:
            close(energy, record["energy_eV"], f"{label} optimizer energy")
            force = record["fmax_eV_A"]
            if not math.isfinite(force) or force < 0:
                failed.append(f"invalid {label} combined force")
            elif force >= target:
                # ASE uses strict '< fmax'. A boundary rounded to the target
                # cannot establish convergence, so it is deliberately excluded.
                unconverged.append(f"{label} combined fmax {force:g} >= {target:g} eV/A")
        out = _choose(case, f"{label}_dftpy.out", f"{legacy}_dftpy.out")
        if not out.exists():
            missing.append(f"{label} raw energy output missing")
        else:
            close(energy, output_energy(out), f"{label} raw output energy")
        trajectory = _choose(case, f"{label}_relax.traj", f"{legacy}_relax.traj")
        final = _choose(case, f"{label}_vc_relaxed.vasp", f"{legacy}_vc_relaxed.vasp")
        if not trajectory.exists() or not final.exists():
            missing.append(f"{label} final trajectory/structure missing")
        else:
            try:
                atoms, saved = read(trajectory, index=-1), read(final)
                if len(atoms) != expected_count or len(saved) != expected_count:
                    failed.append(f"{label} final atom-count mismatch")
                close(energy, atoms.get_potential_energy(), f"{label} trajectory energy")
                if not np.all(np.isfinite(atoms.get_forces())):
                    failed.append(f"{label} non-finite trajectory forces")
                if not np.allclose(atoms.cell.array, saved.cell.array, atol=1e-7, rtol=0):
                    failed.append(f"{label} final cell differs from trajectory")
                elif len(atoms) == len(saved):
                    _, distances = find_mic(atoms.positions - saved.positions, atoms.cell, atoms.pbc)
                    if np.max(distances) > 1e-6:
                        failed.append(f"{label} final positions differ from trajectory")
            except Exception as exc:
                failed.append(f"{label} trajectory/structure unreadable: {exc}")

    pristine_path = case / "pristine_raw.vasp"
    start_path = _choose(case, f"{defect}_start.vasp", "vacancy_start.vasp")
    if not pristine_path.exists() or not start_path.exists():
        missing.append("initial pristine/defect structures missing")
    else:
        try:
            pristine, start = read(pristine_path), read(start_path)
            if len(pristine) != n or len(start) != nd:
                failed.append("initial atom-count mismatch")
            if not np.allclose(pristine.cell.array, start.cell.array, atol=1e-7, rtol=0):
                failed.append("initial pristine/defect cells differ")
            # Match every surviving atom under PBC, not just manifest indices.
            delta = pristine.positions[:, None, :] - start.positions[None, :, :]
            _, distances = find_mic(delta.reshape(-1, 3), pristine.cell, pristine.pbc)
            distances = distances.reshape(len(pristine), len(start))
            matches = distances < 1e-6
            removed = np.where(~matches.any(axis=1))[0]
            if not np.all(matches.sum(axis=0) == 1) or len(removed) != n - nd:
                failed.append("defect does not match pristine lattice with expected vacancies")
            # Order-independent fractional coordinates and cell encode starting
            # geometry for cross-method comparisons. The hash is provenance,
            # not a tolerance-based proof on its own.
            for key, atoms in (("initial_geometry_sha256", start), ("initial_pristine_geometry_sha256", pristine)):
                entries = sorted((symbol, *np.round(frac, 8)) for symbol, frac in
                                 zip(atoms.get_chemical_symbols(), atoms.get_scaled_positions(wrap=True)))
                payload = json.dumps({"cell": np.round(atoms.cell.array, 8).tolist(), "atoms": entries})
                checked[key] = hashlib.sha256(payload.encode()).hexdigest()
            if is_pair and len(removed) == 2:
                declared = {int(manifest[k]) for k in ("first_vacancy_index", "second_vacancy_index") if k in manifest}
                if declared and declared != set(map(int, removed)):
                    failed.append("manifest vacancy indices do not match removed atoms")
                # np.rint tie convention matches the generator at half-cell
                # boundaries; the norm is cross-checked using ASE find_mic.
                frac = pristine.get_scaled_positions(wrap=True)[removed[1]] - pristine.get_scaled_positions(wrap=True)[removed[0]]
                frac -= np.rint(frac)
                vector = frac @ pristine.cell.array
                _, minimum = find_mic(vector, pristine.cell, pristine.pbc)
                close(np.linalg.norm(vector), minimum, "minimum-image pair distance", 1e-7)
                checked["pair_distance_verified_A"] = float(minimum)
                # Conventional repeats can be anisotropic. Convert supercell
                # fractions into underlying conventional lattice coordinates.
                repeat = np.asarray(manifest.get("conventional_repeat", [1, 1, 1]), dtype=float)
                direction = _axis_label(frac * repeat)
                checked["pair_direction_verified"] = direction
                close(manifest.get("pair_distance_A"), minimum, "manifest pair distance", 1e-6)
                close(result.get("pair_distance_A"), minimum, "result pair distance", 1e-6)
                requested = manifest.get("pair_direction_indices")
                if requested is not None and manifest.get("pair_selection") == "fixed_direction":
                    if direction != _axis_label(np.asarray(requested, dtype=float)):
                        failed.append("removed-site direction differs from requested fixed direction")
                elif manifest.get("pair_selection") == "fixed_direction":
                    missing.append("fixed-direction indices unavailable")
        except Exception as exc:
            failed.append(f"initial geometry unreadable: {exc}")
    checked["status"] = "failed" if failed else "unconverged" if unconverged else "missing" if missing else "qualified"
    checked["qualified"] = checked["status"] == "qualified"
    checked["qualification_reasons"] = "; ".join(failed + unconverged + missing) or "energy, geometry and combined-filter convergence evidence verified"
    return checked


def latest_dftpy_root(repo: Path) -> Path:
    override = os.environ.get("AL_DEFECTS_DIVACANCY_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    candidates = [Path(repo).resolve().parent]
    if os.environ.get("AL_DEFECTS_DATA_ROOT"):
        data = Path(os.environ["AL_DEFECTS_DATA_ROOT"]).expanduser()
        candidates.extend((data, data.parent))
    for base in candidates:
        root = base / "DFTPY_DIVACANCY_D110_L0p9_M0p1_RERUN_20260831"
        if (root / "pair_scan").is_dir():
            return root
    raise FileNotFoundError("Set AL_DEFECTS_DIVACANCY_ROOT to the corrected 20260831 data package")


def compatible_comparison(dftpy_rows: list[dict], qe_rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Conservative cross-method join; distance alone is never a join key.

    QE rows must carry independently audited evidence and numerical-convergence
    provenance. Different basis/cutoff and pseudopotentials must be documented
    by that audit, not assumed equal to an OF-DFT grid or recpot.
    """
    comparisons, reasons = [], []
    required = ("status", "pair_direction_verified", "initial_geometry_sha256",
                "initial_pristine_geometry_sha256", "energy_definition", "distance_convention",
                "xc", "relaxation_mode", "target_pressure_GPa", "N_pristine", "N_vacancy",
                "vacancy_count", "pair_distance_A", "Ef_vac_eV", "reference_consistency_verified")
    if not qe_rows:
        return [], ["No independently qualified QE divacancy data supplied; comparison unavailable."]
    for qe in qe_rows:
        label = str(qe.get("setting", "unknown QE case"))
        absent = [k for k in required if k not in qe or qe[k] in (None, "", "unknown")]
        if absent:
            reasons.append(f"{label}: missing comparison evidence: {', '.join(absent)}")
            continue
        if qe["status"] != "qualified" or str(qe["reference_consistency_verified"]).lower() != "true":
            reasons.append(f"{label}: QE qualification/reference verification absent")
            continue
        if not qe.get("numerical_convergence_evidence") or not qe.get("pseudopotential_validation_evidence"):
            reasons.append(f"{label}: QE convergence and pseudopotential-validation evidence required")
            continue
        matches = []
        for of in dftpy_rows:
            if of.get("status") != "qualified":
                continue
            strings = ("pair_direction_verified", "initial_geometry_sha256", "initial_pristine_geometry_sha256",
                       "energy_definition", "distance_convention", "xc", "relaxation_mode")
            numbers = ("N_pristine", "N_vacancy", "vacancy_count", "target_pressure_GPa", "pair_distance_A")
            if all(str(of.get(k, "")).upper() == str(qe[k]).upper() for k in strings) and all(
                math.isclose(as_float(of.get(k)), as_float(qe[k]), rel_tol=0, abs_tol=1e-7) for k in numbers):
                if math.isfinite(as_float(qe["Ef_vac_eV"])):
                    matches.append(of)
        if len(matches) != 1:
            reasons.append(f"{label}: expected one matching qualified geometry/method definition; found {len(matches)}")
            continue
        of = matches[0]
        comparisons.append({"r_A": of["pair_distance_A"], "direction": of["pair_direction_verified"],
                            "DFTpy_E_2vac_eV": of["Ef_vac_eV"], "QE_E_2vac_eV": qe["Ef_vac_eV"],
                            "DFTpy_setting": of["setting"], "QE_setting": label})
    return comparisons, reasons


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Read-only qualification of one DFTpy vacancy case")
    parser.add_argument("case", type=Path)
    result = qualify_case(parser.parse_args().case)
    # JSON has no NaN literal. Null communicates unavailable evidence.
    compact = {key: None if isinstance(value, float) and not math.isfinite(value) else value
               for key, value in result.items()}
    print(json.dumps(compact, sort_keys=True))
    sys.exit(0 if result["qualified"] else 1)
