from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
import re
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
from ase.io import read, write


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dft_engine import relax_atoms_and_cell


def repeat_label(manifest: dict[str, object]) -> str:
    repeat = manifest.get("conventional_repeat", ["?", "?", "?"])
    if isinstance(repeat, list) and len(repeat) == 3:
        return f"conv_{int(repeat[0]):02d}x{int(repeat[1]):02d}x{int(repeat[2]):02d}"
    return str(manifest.get("conventional_repeat_label", "unknown"))


def resolve_case(rootdir: Path, setting: str, scan: str) -> Path:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', setting) or '..' in setting:
        raise ValueError('Setting must be a single safe directory name')
    candidates: list[Path] = []
    if scan in {"auto", "spacing"}:
        candidates.append(rootdir / "spacing_scan" / setting)
    if scan in {"auto", "size"}:
        candidates.append(rootdir / "size_scan" / setting)
    if scan in {"auto", "weight"}:
        candidates.append(rootdir / "weight_scan" / setting)
    if scan in {"auto", "pair"}:
        candidates.append(rootdir / "pair_scan" / setting)
    for candidate in candidates:
        if candidate.exists():
            resolved = candidate.resolve()
            if resolved.parent != candidate.parent.resolve() or rootdir.resolve() not in resolved.parents:
                raise ValueError('Case path escapes calculation root')
            return resolved
    raise FileNotFoundError("Could not find setting. Tried:\n" + "\n".join(str(x) for x in candidates))


def write_structure_pair(base: Path, atoms) -> None:
    write(str(base.with_suffix(".xyz")), atoms)
    write(str(base.with_suffix(".vasp")), atoms, direct=True, vasp5=True)


def write_calculator_provenance(
    path: Path,
    *,
    structure_file: str,
    pp_file: Path,
    spacing: float,
    kedf: str,
    xc: str,
    kedf_x: float,
    kedf_y: float,
    fmax: float,
    steps: int,
    pressure_gpa: float,
    ase_optimizer: str,
) -> None:
    data = {
        "structure_file": structure_file,
        "dftpy_calculator": {
            "JOB": {"calctype": "Energy Force Stress"},
            "PATH": {"pppath": str(pp_file.parent)},
            "PP": {"Al": pp_file.name},
            "GRID": {"spacing": spacing},
            "EXC": {"xc": xc},
            "KEDF": {"kedf": kedf, "x": kedf_x, "y": kedf_y},
            "OPT": {"method": "LBFGS"},
        },
        "ase_full_relaxation": {
            "cell_filter": "FrechetCellFilter",
            "optimizer": ase_optimizer,
            "fmax_eV_A": fmax,
            "max_steps": steps,
            "scalar_pressure_GPa": pressure_gpa,
        },
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run DFTpy full atom+cell relaxation for one vacancy case.")
    ap.add_argument("--rootdir", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--scan", choices=["auto", "spacing", "size", "weight", "pair"], default="auto")
    ap.add_argument("--pressure-gpa", type=float, default=0.0)
    ap.add_argument('--restart', action='store_true', help='Archive the entire prior attempt before rerunning from input structures')
    ap.add_argument(
        "--ase-optimizer",
        choices=["BFGS", "LBFGS", "BFGSLineSearch", "SciPyFminBFGS", "SciPyFminCG", "MDMin"],
        default="BFGS",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    rootdir = Path(args.rootdir).expanduser().resolve()
    case_dir = resolve_case(rootdir, str(args.setting), str(args.scan))
    # Linux/WSL execution: refuse simultaneous writes to the same case.
    import fcntl
    case_lock = (case_dir / '.case.lock').open('a')
    fcntl.flock(case_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((case_dir / "point_manifest.json").read_text(encoding="utf-8"))
    if manifest.get('setting') != args.setting:
        raise ValueError('Manifest setting differs from directory')
    if not math.isfinite(args.pressure_gpa) or args.pressure_gpa != 0:
        raise ValueError('This formation-energy workflow is defined at zero external pressure')

    pp_declared = Path(str(manifest["pp_file"])).expanduser()
    pp_candidates = [
        pp_declared if pp_declared.is_absolute() else case_dir / pp_declared,
        case_dir / pp_declared.name,
        rootdir / pp_declared.name,
    ]
    pp_file = next((candidate.resolve() for candidate in pp_candidates if candidate.exists()), None)
    if pp_file is None:
        raise FileNotFoundError(
            "Missing DFTpy pseudopotential. Tried:\n" + "\n".join(str(path) for path in pp_candidates)
        )

    spacing = float(manifest["spacing_A"])
    kedf = str(manifest["kedf"])
    xc = str(manifest.get("xc", "PBE")).strip().upper()
    kedf_x = float(manifest.get("kedf_x", 1.0))
    kedf_y = float(manifest.get("kedf_y", 1.0))
    fmax = float(manifest["fmax_eV_per_A"])
    steps = int(manifest["relax_steps"])

    is_pair = str(manifest.get("scan_type", "")) == "pair"
    defect_label = "divacancy" if is_pair else "vacancy"
    defect_start = case_dir / f"{defect_label}_start.vasp"
    if not defect_start.exists() and is_pair:
        # Backward compatibility for packages prepared before the explicit
        # divacancy naming fix.
        defect_start = case_dir / "vacancy_start.vasp"

    pristine = read(str(case_dir / "pristine_raw.vasp"))
    defect = read(str(defect_start))
    if len(pristine) != int(manifest['pristine_n_atoms']) or len(defect) != int(manifest['vacancy_n_atoms']):
        raise ValueError('Input structures and manifest atom counts disagree')
    if is_pair and len(pristine) - len(defect) != 2:
        raise ValueError('Divacancy calculation must remove exactly two atoms')
    if not np.allclose(pristine.cell.array, defect.cell.array, atol=1e-7, rtol=0):
        raise ValueError('Initial pristine/defect cells differ')
    if not all(math.isfinite(x) for x in (spacing, fmax, kedf_x, kedf_y)) or spacing <= 0 or fmax <= 0 or steps <= 0:
        raise ValueError('Invalid numerical settings')
    if 'pp_sha256' in manifest and hashlib.sha256(pp_file.read_bytes()).hexdigest() != manifest['pp_sha256']:
        raise ValueError('Pseudopotential differs from prepared hash')
    artifacts = [p for p in case_dir.iterdir() if p.is_file() and (
        p.name in ('result.json', 'run_failure.json', 'run_provenance.json', 'local_runner.log')
        or p.name.endswith(('_relax.log', '_relax.traj', '_dftpy.out', '_relaxed.vasp', '_relaxed.xyz')))]
    if artifacts:
        if not args.restart:
            raise FileExistsError('Prior attempt exists; --restart explicitly archives it before rerunning')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        archive = rootdir / 'audit' / f'{args.setting}_{stamp}'
        shutil.copytree(case_dir, archive)
        print(f'[ARCHIVED] {archive}')
        # Only stale completion markers are removed; all copies are preserved.
        for name in ('result.json', 'run_failure.json'):
            (case_dir / name).unlink(missing_ok=True)
    provenance = {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version,
        'packages': {name: version(name) for name in ('dftpy', 'ase', 'numpy', 'scipy')},
        'sha256': {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name:
                   hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   (Path(__file__).resolve(), ROOT / 'app/dft_engine.py', pp_file,
                    case_dir / 'point_manifest.json', case_dir / 'pristine_raw.vasp', defect_start)},
        'electronic_convergence_status': 'not_exposed_by_dftpy_ase_api',
        'console_record': 'local_runner.log',
    }
    (case_dir / 'run_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')

    write_calculator_provenance(
        case_dir / "dftpy_pristine_calculator_config.json",
        structure_file="pristine_raw.vasp",
        pp_file=pp_file,
        spacing=spacing,
        kedf=kedf,
        xc=xc,
        kedf_x=kedf_x,
        kedf_y=kedf_y,
        fmax=fmax,
        steps=steps,
        pressure_gpa=float(args.pressure_gpa),
        ase_optimizer=str(args.ase_optimizer),
    )
    write_calculator_provenance(
        case_dir / f"dftpy_{defect_label}_calculator_config.json",
        structure_file=defect_start.name,
        pp_file=pp_file,
        spacing=spacing,
        kedf=kedf,
        xc=xc,
        kedf_x=kedf_x,
        kedf_y=kedf_y,
        fmax=fmax,
        steps=steps,
        pressure_gpa=float(args.pressure_gpa),
        ase_optimizer=str(args.ase_optimizer),
    )

    # Capture density-iteration console output as well as ASE optimizer logs.
    def recorded_relax(*values, **kwargs):
        with (case_dir / 'local_runner.log').open('a', encoding='utf-8') as console:
            # DFTpy retains a reference to stdout at import time. Redirecting
            # Python's sys.stdout alone loses that transcript; capture FDs too.
            sys.stdout.flush()
            sys.stderr.flush()
            saved_out, saved_err = os.dup(1), os.dup(2)
            try:
                os.dup2(console.fileno(), 1)
                os.dup2(console.fileno(), 2)
                with redirect_stdout(console), redirect_stderr(console):
                    try:
                        return relax_atoms_and_cell(*values, **kwargs)
                    except Exception as exc:
                        traceback.print_exc()
                        (case_dir / 'run_failure.json').write_text(json.dumps({
                            'status': 'failed', 'error': str(exc), 'logfile': kwargs.get('logfile'),
                            'time_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
                        raise
            finally:
                console.flush()
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(saved_out, 1)
                os.dup2(saved_err, 2)
                os.close(saved_out)
                os.close(saved_err)
    (case_dir / 'local_runner.log').write_text('', encoding='utf-8')
    pristine_relaxed, pristine_energy, pristine_stress = recorded_relax(
        pristine,
        pp_file=pp_file,
        spacing=spacing,
        kedf=kedf,
        xc=xc,
        kedf_x=kedf_x,
        kedf_y=kedf_y,
        fmax=fmax,
        steps=steps,
        logfile=str(case_dir / "pristine_relax.log"),
        trajfile=str(case_dir / "pristine_relax.traj"),
        dftpy_outfile=str(case_dir / "pristine_dftpy.out"),
        scalar_pressure_gpa=float(args.pressure_gpa),
        ase_optimizer=str(args.ase_optimizer),
    )
    defect_relaxed, defect_energy, defect_stress = recorded_relax(
        defect,
        pp_file=pp_file,
        spacing=spacing,
        kedf=kedf,
        xc=xc,
        kedf_x=kedf_x,
        kedf_y=kedf_y,
        fmax=fmax,
        steps=steps,
        logfile=str(case_dir / f"{defect_label}_relax.log"),
        trajfile=str(case_dir / f"{defect_label}_relax.traj"),
        dftpy_outfile=str(case_dir / f"{defect_label}_dftpy.out"),
        scalar_pressure_gpa=float(args.pressure_gpa),
        ase_optimizer=str(args.ase_optimizer),
    )

    write_structure_pair(case_dir / "pristine_vc_relaxed", pristine_relaxed)
    write_structure_pair(case_dir / f"{defect_label}_vc_relaxed", defect_relaxed)
    write_structure_pair(case_dir / "pristine_relaxed", pristine_relaxed)
    write_structure_pair(case_dir / f"{defect_label}_relaxed", defect_relaxed)

    n_pristine = int(manifest["pristine_n_atoms"])
    n_vacancy = int(manifest["vacancy_n_atoms"])
    vacancy_count = n_pristine - n_vacancy
    ef_vac = float(defect_energy - (n_vacancy / n_pristine) * pristine_energy)
    pristine_fmax = float(np.linalg.norm(pristine_relaxed.get_forces(), axis=1).max())
    defect_fmax = float(np.linalg.norm(defect_relaxed.get_forces(), axis=1).max())

    result = {
        "setting": str(manifest["setting"]),
        "scan_type": str(manifest.get("scan_type", "unknown")),
        "relaxation_mode": "full_atom_and_cell_relaxation_vc_relax_equivalent",
        "cell_basis": str(manifest["cell_basis"]),
        "conventional_repeat_label": repeat_label(manifest),
        "conventional_repeat": manifest.get("conventional_repeat", []),
        "pristine_n_atoms": n_pristine,
        "vacancy_n_atoms": n_vacancy,
        "vacancy_count": vacancy_count,
        "defect_label": defect_label,
        "vacancy_concentration_fraction": float(vacancy_count) / float(n_pristine),
        "vacancy_concentration_percent": 100.0 * float(vacancy_count) / float(n_pristine),
        "pair_distance_A": manifest.get("pair_distance_A"),
        "spacing_A": spacing,
        "ecut_analogue_eV": float(manifest.get("ecut_analogue_eV", 0.0)),
        "kedf": kedf,
        "kedf_x": kedf_x,
        "kedf_y": kedf_y,
        "xc": xc,
        "fmax_eV_per_A": fmax,
        "target_pressure_GPa": float(args.pressure_gpa),
        "ase_optimizer": str(args.ase_optimizer),
        'pristine_relaxation_evidence': pristine_relaxed.info.get('relaxation_evidence', {}),
        'defect_relaxation_evidence': defect_relaxed.info.get('relaxation_evidence', {}),
        'electronic_convergence_status': 'not_exposed_by_dftpy_ase_api',
        'thesis_acceptance': 'not_assessed',
        "pristine_energy_eV": float(pristine_energy),
        "vacancy_energy_eV": float(defect_energy),
        f"{defect_label}_energy_eV": float(defect_energy),
        "vacancy_formation_energy_eV": ef_vac,
        "pristine_stress_GPa": pristine_stress.tolist(),
        "vacancy_stress_GPa": defect_stress.tolist(),
        f"{defect_label}_stress_GPa": defect_stress.tolist(),
        "pristine_final_fmax_eV_A": pristine_fmax,
        "vacancy_final_fmax_eV_A": defect_fmax,
        f"{defect_label}_final_fmax_eV_A": defect_fmax,
        "pristine_cell_lengths_A": [float(x) for x in pristine_relaxed.cell.lengths()],
        "vacancy_cell_lengths_A": [float(x) for x in defect_relaxed.cell.lengths()],
        f"{defect_label}_cell_lengths_A": [float(x) for x in defect_relaxed.cell.lengths()],
        "pristine_cell_angles_deg": [float(x) for x in pristine_relaxed.cell.angles()],
        "vacancy_cell_angles_deg": [float(x) for x in defect_relaxed.cell.angles()],
        f"{defect_label}_cell_angles_deg": [float(x) for x in defect_relaxed.cell.angles()],
        "formula": "E_f^defect = E_full-relax_defect^(N-nvac) - ((N-nvac)/N) E_full-relax_pristine^N",
    }
    (case_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    from divacancy_analysis_checks import qualify_case
    qualification = qualify_case(case_dir)
    result['status'] = qualification['status']
    result['qualification_reasons'] = qualification['qualification_reasons']
    (case_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'case': str(case_dir), 'status': result['status'],
                      'formation_energy_eV': ef_vac, 'reasons': result['qualification_reasons']}, indent=2))
    return 0 if qualification['qualified'] else 1


if __name__ == "__main__":
    sys.exit(main())
