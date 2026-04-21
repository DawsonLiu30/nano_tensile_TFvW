from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_grip_vacancy_wire import (
    _delete_atom,
    _ellipse_area_xy,
    _remap_after_delete,
    _write_structure_pair,
    build_finite_grip_wire,
    choose_free_surface_vacancy,
    select_grip_indices,
)


def _parse_float_list(text: str) -> list[float]:
    values = []
    for chunk in str(text).split(","):
        token = chunk.strip()
        if token:
            values.append(float(token))
    if not values:
        raise ValueError("No diameters were provided.")
    return values


def _case_name(diameter_nm: float) -> str:
    return f"finite_grip_111_{float(diameter_nm):.1f}nm_vacancy_tfvw"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Create geometry-only preview inputs for finite-grip vacancy nanowires."
    )
    ap.add_argument("--diameters", required=True, help="Comma-separated diameter/r labels, e.g. 1,2,3,4,5,6")
    ap.add_argument("--orientation", default="111", choices=["111", "100", "110"])
    ap.add_argument("--a0", type=float, default=4.118877004246)
    ap.add_argument("--wire-length", type=float, default=21.0)
    ap.add_argument("--min-wire-span", type=float, default=10.0)
    ap.add_argument("--xy-vacuum", type=float, default=10.0)
    ap.add_argument("--z-vacuum", type=float, default=10.0)
    ap.add_argument("--grip-thickness", type=float, default=3.0)
    ap.add_argument("--vacancy-z-window-fraction", type=float, default=0.35)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    for diameter_nm in _parse_float_list(args.diameters):
        case_name = _case_name(diameter_nm)
        outdir = ROOT / "cases" / case_name / "inputs_preview"
        outdir.mkdir(parents=True, exist_ok=True)

        pristine = build_finite_grip_wire(
            a0=float(args.a0),
            diameter_nm=float(diameter_nm),
            wire_length=float(args.wire_length),
            xy_vacuum=float(args.xy_vacuum),
            z_vacuum=float(args.z_vacuum),
            orientation=str(args.orientation),
        )
        pos = pristine.get_positions()
        wire_span = float(pos[:, 2].max() - pos[:, 2].min())
        if wire_span < float(args.min_wire_span):
            raise RuntimeError(
                f"{case_name}: wire z span {wire_span:.6f} A is below {float(args.min_wire_span):.6f} A"
            )

        bottom, top = select_grip_indices(pristine, grip_thickness=float(args.grip_thickness))
        vacancy_idx, vacancy_site = choose_free_surface_vacancy(
            pristine,
            bottom_idx=bottom,
            top_idx=top,
            z_window_fraction=float(args.vacancy_z_window_fraction),
        )
        vacancy = _delete_atom(pristine, vacancy_idx)
        bottom_vac = _remap_after_delete(bottom, vacancy_idx)
        top_vac = _remap_after_delete(top, vacancy_idx)
        fixed_vac = np.unique(np.concatenate([bottom_vac, top_vac])).astype(int)

        _write_structure_pair(outdir / "pristine_raw_preview", pristine)
        _write_structure_pair(outdir / "vacancy_start_preview", vacancy)

        vpos = vacancy.get_positions()
        bottom_center = float(vpos[bottom_vac, 2].mean())
        top_center = float(vpos[top_vac, 2].mean())
        preview_metadata = {
            "case": case_name,
            "preview_only": True,
            "note": "Geometry preview only. Run prepare_grip_vacancy_wire.py to create relaxed production inputs.",
            "boundary_condition": "finite_grip_displacement_control",
            "bottom_grip_indices": [int(i) for i in bottom_vac],
            "top_grip_indices": [int(i) for i in top_vac],
            "fixed_indices": [int(i) for i in fixed_vac],
            "grip_thickness_A": float(args.grip_thickness),
            "bottom_grip_center_z_A": bottom_center,
            "top_grip_center_z_A": top_center,
            "grip_distance_preview_A": float(top_center - bottom_center),
            "wire_span_preview_A": float(vpos[:, 2].max() - vpos[:, 2].min()),
            "area_preview_ellipse_A2": _ellipse_area_xy(vacancy),
            "cell_lengths_A": [float(x) for x in vacancy.get_cell().lengths()],
        }
        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "case": case_name,
            "preview_only": True,
            "geometry": {
                "orientation": str(args.orientation),
                "diameter_nm": float(diameter_nm),
                "a0_input_A": float(args.a0),
                "wire_length_target_A": float(args.wire_length),
                "wire_span_preview_A": preview_metadata["wire_span_preview_A"],
                "xy_vacuum_A": float(args.xy_vacuum),
                "z_vacuum_A": float(args.z_vacuum),
                "min_wire_span_requirement_A": float(args.min_wire_span),
            },
            "grips": {
                "bottom_count": int(bottom_vac.size),
                "top_count": int(top_vac.size),
                "free_atom_count": int(len(vacancy) - fixed_vac.size),
            },
            "vacancy": {
                "selected_site": vacancy_site,
                "selection_rule": "outermost atom inside the central free region, excluding fixed grip atoms",
            },
            "artifacts": {
                "pristine_raw_preview": str(outdir / "pristine_raw_preview.vasp"),
                "vacancy_start_preview": str(outdir / "vacancy_start_preview.vasp"),
                "preview_metadata": str(outdir / "grip_metadata_preview.json"),
            },
        }
        (outdir / "grip_metadata_preview.json").write_text(json.dumps(preview_metadata, indent=2), encoding="utf-8")
        (outdir / "grip_vacancy_preview_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(
            f"[preview] {case_name}: atoms={len(vacancy)} span={preview_metadata['wire_span_preview_A']:.6f} A "
            f"bottom/top={bottom_vac.size}/{top_vac.size} vacancy_z={vacancy_site['z_A']:.6f} A"
        )


if __name__ == "__main__":
    main()
