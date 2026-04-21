from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from ase.io import read


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _latest_result_dir(case_dir: Path) -> Path:
    candidates = sorted(
        [p for p in (case_dir / "results").glob("*") if (p / "summary.csv").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No result folder with summary.csv found under {case_dir / 'results'}")
    return candidates[0]


def _case_dir_from_results(result_dir: Path) -> Path | None:
    try:
        if result_dir.parent.name == "results":
            return result_dir.parent.parent
    except IndexError:
        return None
    return None


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(value, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _sigma_cell_wire(row: dict[str, str]) -> float:
    return (
        float(row["sigma_cell_zz_GPa"])
        * float(row["cell_area_xy_A2"])
        / float(row["wire_area_ref_ellipse_A2"])
    )


def _d111_from_case(case_dir: Path | None, fallback_a0: float | None) -> float:
    if fallback_a0 is not None:
        return float(fallback_a0) / math.sqrt(3.0)
    if case_dir is not None:
        manifest = _read_json(case_dir / "inputs" / "grip_vacancy_manifest.json")
        a0 = manifest.get("geometry", {}).get("a0_input_A")
        if a0 is not None:
            return float(a0) / math.sqrt(3.0)
    raise ValueError("Could not infer a0. Pass --a0 explicitly.")


def _free_indices(n_atoms: int, fixed_indices: np.ndarray) -> np.ndarray:
    all_idx = np.arange(n_atoms, dtype=int)
    if fixed_indices.size == 0:
        return all_idx
    valid_fixed = fixed_indices[(fixed_indices >= 0) & (fixed_indices < n_atoms)]
    return np.setdiff1d(all_idx, valid_fixed, assume_unique=False)


def _bond_set(positions: np.ndarray, indices: np.ndarray, cutoff: float) -> set[tuple[int, int]]:
    bonds: set[tuple[int, int]] = set()
    idx = np.asarray(indices, dtype=int)
    for local_i, atom_i in enumerate(idx[:-1]):
        delta = positions[idx[local_i + 1 :]] - positions[atom_i]
        dist = np.linalg.norm(delta, axis=1)
        for atom_j in idx[local_i + 1 :][dist <= cutoff]:
            i = int(atom_i)
            j = int(atom_j)
            bonds.add((i, j) if i < j else (j, i))
    return bonds


def _d2min_like(
    prev_pos: np.ndarray,
    cur_pos: np.ndarray,
    indices: np.ndarray,
    cutoff: float,
) -> tuple[float, float, float]:
    values: list[float] = []
    idx = np.asarray(indices, dtype=int)
    for atom_i in idx:
        neighbors = idx[idx != atom_i]
        if neighbors.size < 4:
            continue
        r0 = prev_pos[neighbors] - prev_pos[atom_i]
        rc = cur_pos[neighbors] - cur_pos[atom_i]
        mask = np.linalg.norm(r0, axis=1) <= cutoff
        if int(mask.sum()) < 4:
            continue
        r0 = r0[mask]
        rc = rc[mask]
        # Best local affine map F where r0 @ F ~= rc.
        fmat, *_ = np.linalg.lstsq(r0, rc, rcond=None)
        residual = rc - r0 @ fmat
        values.append(float(np.mean(np.sum(residual * residual, axis=1))))
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    return float(np.nanmedian(arr)), float(np.nanpercentile(arr, 90)), float(np.nanmax(arr))


def _min_slice_area(positions: np.ndarray, free_idx: np.ndarray, n_slices: int) -> float:
    if free_idx.size < 4:
        return float("nan")
    free_pos = positions[free_idx]
    z = free_pos[:, 2]
    zmin = float(np.min(z))
    zmax = float(np.max(z))
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        return float("nan")
    edges = np.linspace(zmin, zmax, int(n_slices) + 1)
    areas: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == edges[-1]:
            mask = (z >= lo) & (z <= hi)
        else:
            mask = (z >= lo) & (z < hi)
        slab = free_pos[mask]
        if len(slab) < 3:
            continue
        span_x = float(np.max(slab[:, 0]) - np.min(slab[:, 0]))
        span_y = float(np.max(slab[:, 1]) - np.min(slab[:, 1]))
        areas.append(math.pi * max(span_x, 0.0) * max(span_y, 0.0) / 4.0)
    return float(np.min(areas)) if areas else float("nan")


def _fracture_by_components(row: dict[str, str], *, min_second_cluster: int) -> bool:
    n_components = int(_safe_float(row.get("n_components"), 1.0))
    second = int(_safe_float(row.get("second_component_atoms"), 0.0))
    return n_components > 1 and second >= int(min_second_cluster)


def _adaptive_threshold(values: np.ndarray, floor: float, baseline_count: int = 4) -> float:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return float(floor)
    base = clean[: min(baseline_count, clean.size)]
    median = float(np.nanmedian(base))
    mad = float(np.nanmedian(np.abs(base - median)))
    return float(max(floor, median + 6.0 * max(mad, 1.0e-12)))


def _first_true_cycle(cycles: np.ndarray, mask: np.ndarray) -> int | None:
    matches = cycles[mask]
    return int(matches[0]) if matches.size else None


def _find_strength_candidate(
    cycles: np.ndarray,
    stress: np.ndarray,
    structural_event: np.ndarray,
    *,
    abs_drop_min: float,
    rel_drop_min: float,
    future_window: int,
    recovery_tol_frac: float,
    recovery_tol_abs: float,
) -> tuple[int | None, str]:
    n = len(stress)
    for i in range(1, max(1, n - 1)):
        if i + 1 >= n:
            continue
        if not (stress[i] > stress[i - 1] and stress[i] >= stress[i + 1]):
            continue
        end = min(n, i + 1 + int(future_window))
        future = stress[i + 1 : end]
        if future.size == 0:
            continue
        drop = float(stress[i] - np.nanmin(future))
        drop_threshold = max(float(abs_drop_min), float(rel_drop_min) * max(abs(float(stress[i])), 1.0))
        if drop < drop_threshold:
            continue
        allowed_recovery = float(stress[i]) * (1.0 + float(recovery_tol_frac)) + float(recovery_tol_abs)
        if float(np.nanmax(future)) > allowed_recovery:
            continue
        if bool(np.any(structural_event[i:end])):
            return int(cycles[i]), "stress_peak+sustained_drop+structural_event"
        return int(cycles[i]), "stress_peak+sustained_drop; structural evidence not yet triggered"
    return None, "not_found"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Three-layer finite-grip tensile event analysis.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--case-dir", help="Case directory; latest results/*/summary.csv is used.")
    src.add_argument("--results-dir", help="Specific results directory containing summary.csv.")
    ap.add_argument("--a0", type=float, default=None, help="Lattice constant in Angstrom. Defaults to manifest a0.")
    ap.add_argument("--gap-factor", type=float, default=3.0)
    ap.add_argument("--bond-cutoff-factor", type=float, default=1.28)
    ap.add_argument("--bond-change-floor", type=float, default=0.08)
    ap.add_argument("--d2min-p90-floor", type=float, default=0.08)
    ap.add_argument("--diagnostic-abs-min", type=float, default=3.0)
    ap.add_argument("--diagnostic-rel-max", type=float, default=0.25)
    ap.add_argument("--drop-abs-min", type=float, default=2.0)
    ap.add_argument("--drop-rel-min", type=float, default=0.15)
    ap.add_argument("--future-window", type=int, default=3)
    ap.add_argument("--recovery-tol-frac", type=float, default=0.05)
    ap.add_argument("--recovery-tol-abs", type=float, default=0.5)
    ap.add_argument("--min-second-cluster", type=int, default=2)
    ap.add_argument("--slices", type=int, default=8)
    ap.add_argument("--out", default="", help="Output event CSV. Default: tensile_events.csv in result dir.")
    ap.add_argument("--summary-out", default="", help="Output event summary JSON. Default: tensile_event_summary.json.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.case_dir:
        case_dir = _resolve(args.case_dir)
        result_dir = _latest_result_dir(case_dir)
    else:
        result_dir = _resolve(args.results_dir)
        case_dir = _case_dir_from_results(result_dir)

    summary_csv = result_dir / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_csv}")

    rows = _read_csv(summary_csv)
    if not rows:
        raise ValueError(f"No rows in {summary_csv}")

    metadata = _read_json(case_dir / "inputs" / "grip_metadata.json") if case_dir else {}
    fixed_idx = np.asarray(metadata.get("fixed_indices", []), dtype=int)
    d111 = _d111_from_case(case_dir, args.a0)
    nn_dist = d111 * math.sqrt(3.0 / 2.0)
    bond_cutoff = float(args.bond_cutoff_factor) * nn_dist
    gap_threshold = float(args.gap_factor) * d111

    fracture_rows = _read_csv(result_dir / "fracture_status.csv") if (result_dir / "fracture_status.csv").exists() else []
    fracture_by_cycle = {int(row["cycle"]): row for row in fracture_rows if row.get("cycle") not in (None, "")}

    cycles = np.asarray([int(row["cycle"]) for row in rows], dtype=int)
    strain_pct = np.asarray([float(row["strain"]) * 100.0 for row in rows], dtype=float)
    grip_raw = np.asarray([float(row["grip_stress_avg_GPa"]) for row in rows], dtype=float)
    primary = grip_raw - float(grip_raw[0])
    cell_wire = np.asarray([_sigma_cell_wire(row) for row in rows], dtype=float)
    cell_wire_offset = cell_wire - float(cell_wire[0])
    primary_step = np.diff(primary, prepend=primary[0])
    cell_wire_step = np.diff(cell_wire, prepend=cell_wire[0])
    # Diagnostic anomalies are reserved for isolated or step-wise conflicts,
    # not for a persistent scale offset between cell stress and grip reaction.
    diagnostic_delta = np.abs(cell_wire_step - primary_step)
    diagnostic_threshold = max(float(args.diagnostic_abs_min), float(args.diagnostic_rel_max) * float(np.nanmax(np.abs(primary))))
    residual_anomaly = np.zeros(len(cycles), dtype=bool)
    residual_anomaly[0] = abs(float(cell_wire[0])) > diagnostic_threshold
    sign_anomaly = (primary > diagnostic_threshold) & (cell_wire < -float(args.diagnostic_abs_min))
    diagnostic_anomaly = residual_anomaly | sign_anomaly | (diagnostic_delta > diagnostic_threshold)

    atom_positions: dict[int, np.ndarray] = {}
    free_indices_by_cycle: dict[int, np.ndarray] = {}
    bond_sets: dict[int, set[tuple[int, int]]] = {}
    min_slice_area_by_cycle: dict[int, float] = {}
    for cycle in cycles:
        xyz = result_dir / f"cycle_{int(cycle):03d}_relaxed.xyz"
        if not xyz.exists():
            continue
        atoms = read(str(xyz))
        pos = np.asarray(atoms.get_positions(), dtype=float)
        free_idx = _free_indices(len(atoms), fixed_idx)
        atom_positions[int(cycle)] = pos
        free_indices_by_cycle[int(cycle)] = free_idx
        bond_sets[int(cycle)] = _bond_set(pos, free_idx, bond_cutoff)
        min_slice_area_by_cycle[int(cycle)] = _min_slice_area(pos, free_idx, int(args.slices))

    bond_count = np.full(len(cycles), np.nan)
    bond_lost = np.zeros(len(cycles), dtype=float)
    bond_gained = np.zeros(len(cycles), dtype=float)
    bond_change_fraction = np.zeros(len(cycles), dtype=float)
    d2min_median = np.full(len(cycles), np.nan)
    d2min_p90 = np.full(len(cycles), np.nan)
    d2min_max = np.full(len(cycles), np.nan)
    min_slice_area = np.full(len(cycles), np.nan)

    for i, cycle in enumerate(cycles):
        cycle_int = int(cycle)
        bonds = bond_sets.get(cycle_int, set())
        bond_count[i] = float(len(bonds))
        min_slice_area[i] = min_slice_area_by_cycle.get(cycle_int, float("nan"))
        if i == 0:
            continue
        prev_cycle = int(cycles[i - 1])
        prev_bonds = bond_sets.get(prev_cycle, set())
        if bonds or prev_bonds:
            lost = prev_bonds - bonds
            gained = bonds - prev_bonds
            bond_lost[i] = float(len(lost))
            bond_gained[i] = float(len(gained))
            bond_change_fraction[i] = (float(len(lost) + len(gained)) / max(float(len(prev_bonds)), 1.0))
        if prev_cycle in atom_positions and cycle_int in atom_positions:
            prev_pos = atom_positions[prev_cycle]
            cur_pos = atom_positions[cycle_int]
            free_prev = free_indices_by_cycle.get(prev_cycle, np.arange(len(prev_pos)))
            free_cur = free_indices_by_cycle.get(cycle_int, np.arange(len(cur_pos)))
            free_idx = np.intersect1d(free_prev, free_cur)
            med, p90, maxv = _d2min_like(prev_pos, cur_pos, free_idx, bond_cutoff)
            d2min_median[i] = med
            d2min_p90[i] = p90
            d2min_max[i] = maxv

    bond_threshold = _adaptive_threshold(bond_change_fraction[1:], float(args.bond_change_floor))
    d2min_threshold = _adaptive_threshold(d2min_p90[1:], float(args.d2min_p90_floor))
    structural_event = (bond_change_fraction >= bond_threshold) | (d2min_p90 >= d2min_threshold)

    complete_fracture = np.zeros(len(cycles), dtype=bool)
    max_gap_all = np.full(len(cycles), np.nan)
    max_gap_free = np.full(len(cycles), np.nan)
    n_components = np.full(len(cycles), np.nan)
    second_component = np.full(len(cycles), np.nan)
    for i, cycle in enumerate(cycles):
        frow = fracture_by_cycle.get(int(cycle), {})
        max_gap_all[i] = _safe_float(frow.get("max_gap_all_A"))
        max_gap_free[i] = _safe_float(frow.get("max_gap_free_A"))
        n_components[i] = _safe_float(frow.get("n_components"))
        second_component[i] = _safe_float(frow.get("second_component_atoms"))
        gap_fractured = bool(np.isfinite(max_gap_all[i]) and max_gap_all[i] > gap_threshold)
        component_fractured = _fracture_by_components(frow, min_second_cluster=int(args.min_second_cluster)) if frow else False
        complete_fracture[i] = gap_fractured or component_fractured or str(frow.get("fractured", "")).lower() == "true"

    plastic_onset_cycle = _first_true_cycle(cycles, structural_event & (cycles > cycles[0]))
    elastic_limit_cycle = None
    if plastic_onset_cycle is not None:
        matches = np.where(cycles == plastic_onset_cycle)[0]
        if matches.size and matches[0] > 0:
            elastic_limit_cycle = int(cycles[matches[0] - 1])

    strength_cycle, strength_basis = _find_strength_candidate(
        cycles,
        primary,
        structural_event,
        abs_drop_min=float(args.drop_abs_min),
        rel_drop_min=float(args.drop_rel_min),
        future_window=int(args.future_window),
        recovery_tol_frac=float(args.recovery_tol_frac),
        recovery_tol_abs=float(args.recovery_tol_abs),
    )
    complete_fracture_cycle = _first_true_cycle(cycles, complete_fracture)

    out_csv = _resolve(args.out) if args.out else result_dir / "tensile_events.csv"
    out_json = _resolve(args.summary_out) if args.summary_out else result_dir / "tensile_event_summary.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "cycle",
        "strain_pct",
        "sigma_primary_grip_offset_GPa",
        "sigma_grip_raw_GPa",
        "sigma_cell_wire_GPa",
        "sigma_cell_wire_offset_GPa",
        "diagnostic_delta_abs_GPa",
        "diagnostic_threshold_GPa",
        "diagnostic_anomaly",
        "bond_count",
        "bond_lost_from_prev",
        "bond_gained_from_prev",
        "bond_change_fraction",
        "bond_change_threshold",
        "d2min_median_A2",
        "d2min_p90_A2",
        "d2min_max_A2",
        "d2min_p90_threshold_A2",
        "structural_event",
        "min_slice_area_A2",
        "min_slice_area_ratio_to_cycle0",
        "max_gap_all_A",
        "max_gap_free_A",
        "gap_threshold_A",
        "n_components",
        "second_component_atoms",
        "complete_fracture",
        "event_tags",
        "regime",
    ]
    area0 = float(min_slice_area[0]) if np.isfinite(min_slice_area[0]) and min_slice_area[0] > 0 else float("nan")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, cycle in enumerate(cycles):
            tags: list[str] = []
            if bool(diagnostic_anomaly[i]):
                tags.append("diagnostic_anomaly")
            if plastic_onset_cycle is not None and int(cycle) == int(plastic_onset_cycle):
                tags.append("plastic_onset")
            if elastic_limit_cycle is not None and int(cycle) == int(elastic_limit_cycle):
                tags.append("elastic_limit")
            if strength_cycle is not None and int(cycle) == int(strength_cycle):
                tags.append("yield_strength_candidate")
            if complete_fracture_cycle is not None and int(cycle) == int(complete_fracture_cycle):
                tags.append("complete_fracture")

            if complete_fracture_cycle is not None and int(cycle) >= int(complete_fracture_cycle):
                regime = "complete_fracture"
            elif strength_cycle is not None and int(cycle) > int(strength_cycle):
                regime = "post_yield_load_bearing"
            elif plastic_onset_cycle is not None and int(cycle) >= int(plastic_onset_cycle):
                regime = "plastic_onset_region"
            else:
                regime = "elastic_loading_candidate"

            area_ratio = min_slice_area[i] / area0 if np.isfinite(min_slice_area[i]) and np.isfinite(area0) else float("nan")
            writer.writerow(
                {
                    "cycle": int(cycle),
                    "strain_pct": f"{strain_pct[i]:.10f}",
                    "sigma_primary_grip_offset_GPa": f"{primary[i]:.10f}",
                    "sigma_grip_raw_GPa": f"{grip_raw[i]:.10f}",
                    "sigma_cell_wire_GPa": f"{cell_wire[i]:.10f}",
                    "sigma_cell_wire_offset_GPa": f"{cell_wire_offset[i]:.10f}",
                    "diagnostic_delta_abs_GPa": f"{diagnostic_delta[i]:.10f}",
                    "diagnostic_threshold_GPa": f"{diagnostic_threshold:.10f}",
                    "diagnostic_anomaly": str(bool(diagnostic_anomaly[i])),
                    "bond_count": "" if not np.isfinite(bond_count[i]) else f"{bond_count[i]:.0f}",
                    "bond_lost_from_prev": f"{bond_lost[i]:.0f}",
                    "bond_gained_from_prev": f"{bond_gained[i]:.0f}",
                    "bond_change_fraction": f"{bond_change_fraction[i]:.10f}",
                    "bond_change_threshold": f"{bond_threshold:.10f}",
                    "d2min_median_A2": "" if not np.isfinite(d2min_median[i]) else f"{d2min_median[i]:.10f}",
                    "d2min_p90_A2": "" if not np.isfinite(d2min_p90[i]) else f"{d2min_p90[i]:.10f}",
                    "d2min_max_A2": "" if not np.isfinite(d2min_max[i]) else f"{d2min_max[i]:.10f}",
                    "d2min_p90_threshold_A2": f"{d2min_threshold:.10f}",
                    "structural_event": str(bool(structural_event[i])),
                    "min_slice_area_A2": "" if not np.isfinite(min_slice_area[i]) else f"{min_slice_area[i]:.10f}",
                    "min_slice_area_ratio_to_cycle0": "" if not np.isfinite(area_ratio) else f"{area_ratio:.10f}",
                    "max_gap_all_A": "" if not np.isfinite(max_gap_all[i]) else f"{max_gap_all[i]:.10f}",
                    "max_gap_free_A": "" if not np.isfinite(max_gap_free[i]) else f"{max_gap_free[i]:.10f}",
                    "gap_threshold_A": f"{gap_threshold:.10f}",
                    "n_components": "" if not np.isfinite(n_components[i]) else f"{n_components[i]:.0f}",
                    "second_component_atoms": "" if not np.isfinite(second_component[i]) else f"{second_component[i]:.0f}",
                    "complete_fracture": str(bool(complete_fracture[i])),
                    "event_tags": ";".join(tags),
                    "regime": regime,
                }
            )

    def _stress_at(cycle_value: int | None) -> float | None:
        if cycle_value is None:
            return None
        matches = np.where(cycles == int(cycle_value))[0]
        return float(primary[int(matches[0])]) if matches.size else None

    def _strain_at(cycle_value: int | None) -> float | None:
        if cycle_value is None:
            return None
        matches = np.where(cycles == int(cycle_value))[0]
        return float(strain_pct[int(matches[0])]) if matches.size else None

    summary = {
        "result_dir": str(result_dir),
        "case_dir": str(case_dir) if case_dir else "",
        "primary_stress_definition": "grip_stress_avg_GPa - grip_stress_avg_GPa(cycle 0)",
        "diagnostic_stress_definition": "sigma_cell_zz_GPa * A_cell/A_wire",
        "bond_cutoff_A": bond_cutoff,
        "d111_A": d111,
        "gap_threshold_A": gap_threshold,
        "diagnostic_threshold_GPa": diagnostic_threshold,
        "diagnostic_anomaly_definition": (
            "large cell-vs-primary step mismatch, large cycle-0 residual, "
            "or opposite-sign cell-wire stress while primary load is tensile"
        ),
        "bond_change_threshold": bond_threshold,
        "d2min_p90_threshold_A2": d2min_threshold,
        "elastic_limit_cycle": elastic_limit_cycle,
        "elastic_limit_strain_pct": _strain_at(elastic_limit_cycle),
        "elastic_limit_primary_stress_GPa": _stress_at(elastic_limit_cycle),
        "plastic_onset_cycle": plastic_onset_cycle,
        "plastic_onset_strain_pct": _strain_at(plastic_onset_cycle),
        "plastic_onset_primary_stress_GPa": _stress_at(plastic_onset_cycle),
        "yield_strength_candidate_cycle": strength_cycle,
        "yield_strength_candidate_basis": strength_basis,
        "yield_strength_candidate_strain_pct": _strain_at(strength_cycle),
        "yield_strength_candidate_primary_stress_GPa": _stress_at(strength_cycle),
        "complete_fracture_cycle": complete_fracture_cycle,
        "complete_fracture_strain_pct": _strain_at(complete_fracture_cycle),
        "complete_fracture_primary_stress_GPa": _stress_at(complete_fracture_cycle),
        "latest_cycle": int(cycles[-1]),
        "latest_strain_pct": float(strain_pct[-1]),
        "latest_primary_stress_GPa": float(primary[-1]),
        "latest_complete_fracture": bool(complete_fracture[-1]),
        "diagnostic_anomaly_cycles": [int(c) for c in cycles[diagnostic_anomaly]],
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[events] summary       : {summary_csv}")
    print(f"[events] output csv    : {out_csv}")
    print(f"[events] output json   : {out_json}")
    print(f"[events] primary stress: {summary['primary_stress_definition']}")
    if strength_cycle is None:
        print(f"[events] yield/strength: not found ({strength_basis})")
    else:
        print(
            "[events] yield/strength: "
            f"cycle {strength_cycle}, strain={_strain_at(strength_cycle):.6f}%, "
            f"sigma_primary={_stress_at(strength_cycle):.6f} GPa, basis={strength_basis}"
        )
    if plastic_onset_cycle is None:
        print("[events] plastic onset : not detected by structural metrics")
    else:
        print(
            "[events] plastic onset : "
            f"cycle {plastic_onset_cycle}, strain={_strain_at(plastic_onset_cycle):.6f}%"
        )
    if complete_fracture_cycle is None:
        print("[events] fracture      : no complete fracture marker")
    else:
        print(f"[events] fracture      : cycle {complete_fracture_cycle}")


if __name__ == "__main__":
    main()
