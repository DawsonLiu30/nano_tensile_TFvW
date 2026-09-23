from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read, write
from dftpy.mpi import utils as dftpy_mpi_utils


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dft_engine import evaluate_atoms_with_energy_components, relax_atoms_and_cell


def default_source_root() -> Path:
    return (
        ROOT
        / "iservice_packages"
        / "results"
        / "Al_defects"
        / "01_calibration"
        / "single_vacancy"
        / "dftpy_tfvw_lambda_mu"
        / "coarse_10x10_vacancy_formation"
    )


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def capture_dftpy_stdout(path: Path):
    capture = io.StringIO()
    original = dftpy_mpi_utils.environ.get("STDOUT")

    @contextlib.contextmanager
    def manager():
        dftpy_mpi_utils.environ["STDOUT"] = capture
        try:
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                yield
        finally:
            dftpy_mpi_utils.environ["STDOUT"] = original
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(capture.getvalue(), encoding="utf-8", errors="replace")

    return manager()


def parse_settings(source_root: Path) -> list[tuple[str, float, float]]:
    settings = source_root / "01_settings" / "settings_lambda_mu_10x10.txt"
    parsed: list[tuple[str, float, float]] = []
    if settings.exists():
        for line in settings.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 3 and not parts[0].startswith("#"):
                parsed.append((parts[0], float(parts[1]), float(parts[2])))
    else:
        for case_dir in sorted((source_root / "03_runs").glob("tfvw_lam*_mu*")):
            manifest = load_json(case_dir / "point_manifest.json")
            parsed.append((case_dir.name, float(manifest["kedf_x"]), float(manifest["kedf_y"])))
    if not parsed:
        raise FileNotFoundError(f"No lambda/mu settings found under {source_root}")
    return parsed


def copy_case_inputs(source_case: Path, case_dir: Path, *, refresh_inputs: bool) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    wanted = [
        "point_manifest.json",
        "CASE_README.md",
        "al.lda.recpot",
        "dftpy_pristine_input.ini",
        "dftpy_vacancy_input.ini",
        "pristine_raw.vasp",
        "vacancy_start.vasp",
        "pristine_raw.xyz",
        "vacancy_start.xyz",
    ]
    for name in wanted:
        src = source_case / name
        dst = case_dir / name
        if src.exists() and (refresh_inputs or not dst.exists()):
            shutil.copy2(src, dst)


def resolve_pp(case_dir: Path, manifest: dict[str, Any]) -> Path:
    pp_declared = Path(str(manifest.get("pp_file", "al.lda.recpot"))).expanduser()
    candidates = [
        pp_declared if pp_declared.is_absolute() else case_dir / pp_declared,
        case_dir / pp_declared.name,
        case_dir / "al.lda.recpot",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("Missing pseudopotential. Tried:\n" + "\n".join(str(x) for x in candidates))


def resolve_spacing(case_dir: Path, manifest: dict[str, Any]) -> float:
    for key in ("spacing_A", "spacing_A_derived_from_ecut", "spacing"):
        value = manifest.get(key)
        if value is not None:
            return float(value)

    ini = case_dir / "dftpy_pristine_input.ini"
    if ini.exists():
        for line in ini.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(r"\s*spacing\s*=\s*([-+0-9.eE]+)", line)
            if match:
                return float(match.group(1))

    raise KeyError("Could not resolve grid spacing from manifest or dftpy_pristine_input.ini")


def write_structure_pair(base: Path, atoms) -> None:
    write(str(base.with_suffix(".vasp")), atoms, direct=True, vasp5=True)
    write(str(base.with_suffix(".xyz")), atoms)


def write_summary_out(path: Path, *, total_energy: float, stress_gpa: np.ndarray, terms: dict[str, float]) -> None:
    """Write a grep-friendly compact output with the three professor-facing raw values."""
    total_component = sum(value for value in terms.values() if math.isfinite(float(value)))
    lines = [
        "# Compact DFTpy final single-point summary after local relaxation",
        "# Values below are written explicitly so they can be extracted by grep/table scripts.",
        f"total energy (eV) : {total_energy:.12f}",
    ]
    for key in ("KEDF-TF", "KEDF-VW", "KEDF", "XC", "HARTREE", "PSEUDO", "II"):
        if key in terms and math.isfinite(float(terms[key])):
            lines.append(f"{key}: {float(terms[key]):.12f}")
    lines.append(f"TOTAL: {total_component:.12f}")
    lines.append("TOTAL stress (GPa):")
    lines.extend(
        [
            f"{stress_gpa[0,0]:14.6f} {stress_gpa[0,1]:14.6f} {stress_gpa[0,2]:14.6f}",
            f"{stress_gpa[1,0]:14.6f} {stress_gpa[1,1]:14.6f} {stress_gpa[1,2]:14.6f}",
            f"{stress_gpa[2,0]:14.6f} {stress_gpa[2,1]:14.6f} {stress_gpa[2,2]:14.6f}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean_conventional_a0(atoms, repeat: list[int] | tuple[int, int, int]) -> float:
    lengths = np.asarray(atoms.cell.lengths(), dtype=float)
    reps = np.asarray(repeat, dtype=float)
    if reps.shape != (3,) or np.any(reps <= 0):
        reps = np.asarray([3.0, 3.0, 3.0])
    return float(np.mean(lengths / reps))


def run_one_structure(
    *,
    label: str,
    start_file: Path,
    case_dir: Path,
    pp_file: Path,
    manifest: dict[str, Any],
    ase_optimizer: str,
    force: bool,
    abort_fmax: float | None,
    abort_after_steps: int,
) -> dict[str, Any]:
    done_path = case_dir / f"local_{label}_result.json"
    summary_path = case_dir / f"{label}_dftpy.out"
    final_vasp = case_dir / f"{label}_vc_relaxed.vasp"
    if done_path.exists() and summary_path.exists() and final_vasp.exists() and not force:
        return {**load_json(done_path), "skipped_existing": True}

    atoms = read(str(start_file))
    spacing = resolve_spacing(case_dir, manifest)
    kedf = str(manifest.get("kedf", "TFVW"))
    xc = str(manifest.get("xc", "LDA")).upper()
    kedf_x = float(manifest.get("kedf_x", manifest.get("lambda", 1.0)))
    kedf_y = float(manifest.get("kedf_y", manifest.get("mu", 1.0)))
    fmax = float(manifest.get("fmax_eV_per_A", manifest.get("fmax", 0.01)))
    steps = int(manifest.get("relax_steps", 5000))

    start_time = time.time()
    with capture_dftpy_stdout(case_dir / f"{label}_relax_raw_stdout.log"):
        relaxed, relax_energy, relax_stress = relax_atoms_and_cell(
            atoms,
            pp_file=pp_file,
            spacing=spacing,
            kedf=kedf,
            xc=xc,
            kedf_x=kedf_x,
            kedf_y=kedf_y,
            fmax=fmax,
            steps=steps,
            logfile=str(case_dir / f"{label}_relax.log"),
            trajfile=str(case_dir / f"{label}_relax.traj"),
            dftpy_outfile=str(case_dir / f"{label}_relax_compact.out"),
            scalar_pressure_gpa=0.0,
            ase_optimizer=ase_optimizer,
            abort_fmax_eV_A=abort_fmax,
            abort_after_steps=abort_after_steps,
        )

    write_structure_pair(case_dir / f"{label}_vc_relaxed", relaxed)
    # Backward-compatible names expected by existing collectors.
    if label == "pristine":
        write_structure_pair(case_dir / "pristine_relaxed", relaxed)
    elif label == "vacancy":
        write_structure_pair(case_dir / "vacancy_relaxed", relaxed)

    with capture_dftpy_stdout(case_dir / f"{label}_final_scf_raw_stdout.log"):
        relaxed, total_energy, stress_gpa, terms = evaluate_atoms_with_energy_components(
            relaxed,
            pp_file=pp_file,
            spacing=spacing,
            kedf=kedf,
            xc=xc,
            kedf_x=kedf_x,
            kedf_y=kedf_y,
            dftpy_outfile=str(case_dir / f"{label}_final_scf_compact.out"),
        )
    write_summary_out(summary_path, total_energy=total_energy, stress_gpa=stress_gpa, terms=terms)

    forces = relaxed.get_forces()
    repeat = manifest.get("conventional_repeat", [3, 3, 3])
    result = {
        "label": label,
        "start_file": str(start_file),
        "final_vasp": str(final_vasp),
        "done": True,
        "started_at": timestamp(),
        "runtime_seconds": time.time() - start_time,
        "total_energy_eV": float(total_energy),
        "relax_energy_eV": float(relax_energy),
        "kedf_energy_eV": float(terms.get("KEDF", math.nan)),
        "kedf_tf_energy_eV": float(terms.get("KEDF-TF", math.nan)),
        "kedf_vw_energy_eV": float(terms.get("KEDF-VW", math.nan)),
        "xc_energy_eV": float(terms.get("XC", math.nan)),
        "hartree_energy_eV": float(terms.get("HARTREE", math.nan)),
        "pseudo_energy_eV": float(terms.get("PSEUDO", math.nan)),
        "stress_GPa": stress_gpa.tolist(),
        "max_abs_stress_GPa": float(np.max(np.abs(stress_gpa))),
        "final_fmax_eV_A": float(np.linalg.norm(forces, axis=1).max()),
        "cell_lengths_A": [float(x) for x in relaxed.cell.lengths()],
        "cell_angles_deg": [float(x) for x in relaxed.cell.angles()],
        "lattice_constant_A": mean_conventional_a0(relaxed, repeat),
        "n_atoms": len(relaxed),
    }
    done_path.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    return result


def assert_physical_pristine_lattice(result: dict[str, Any], *, min_a0: float, max_a0: float) -> None:
    a0 = float(result.get("lattice_constant_A", math.nan))
    if not math.isfinite(a0) or a0 < float(min_a0) or a0 > float(max_a0):
        raise RuntimeError(
            f"Pathological pristine lattice constant: a0={a0:.6g} A "
            f"outside [{float(min_a0):.6g}, {float(max_a0):.6g}] A; "
            "skipping vacancy calculation for this lambda-mu point"
        )


def maybe_collect(root: Path) -> None:
    script = ROOT / "scripts" / "make_professor_raw_log_table_lambda_mu.py"
    subprocess.run(
        [sys.executable, str(script), "--rootdir", str(root)],
        cwd=str(ROOT),
        text=True,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local resumable DFTpy TFvW lambda/mu rerun for professor raw tables."
    )
    parser.add_argument("--source-root", type=Path, default=default_source_root())
    parser.add_argument(
        "--outroot",
        type=Path,
        default=Path.home() / "Desktop" / "LOCAL_DFTPY_VACLM_10X10_FULL_RERUN_20260626",
    )
    parser.add_argument("--mode", choices=["pristine", "full"], default="full")
    parser.add_argument(
        "--ase-optimizer",
        choices=["BFGS", "LBFGS", "BFGSLineSearch", "SciPyFminBFGS", "SciPyFminCG", "MDMin"],
        default="MDMin",
    )
    parser.add_argument("--force", action="store_true", help="Rerun cases even if local result files already exist.")
    parser.add_argument("--refresh-inputs", action="store_true", help="Refresh copied input files from source.")
    parser.add_argument("--collect-every", type=int, default=5)
    parser.add_argument(
        "--settings",
        default="",
        help="Optional comma-separated setting names to run instead of the full matrix.",
    )
    parser.add_argument(
        "--abort-fmax",
        type=float,
        default=1.0e5,
        help="Abort and mark a structure failed if cell-filter fmax exceeds this value after the warmup.",
    )
    parser.add_argument("--abort-after-steps", type=int, default=80)
    parser.add_argument("--min-pristine-a0", type=float, default=3.5)
    parser.add_argument("--max-pristine-a0", type=float, default=4.5)
    parser.add_argument("--max-cases", type=int, default=0, help="Debug/testing limit; 0 means all cases.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    outroot = args.outroot.expanduser().resolve()
    runs_out = outroot / "03_runs"
    runs_src = source_root / "03_runs"
    outroot.mkdir(parents=True, exist_ok=True)
    runs_out.mkdir(parents=True, exist_ok=True)

    settings = parse_settings(source_root)
    if args.settings.strip():
        wanted = {item.strip() for item in args.settings.split(",") if item.strip()}
        by_name = {name: (name, lam, mu) for name, lam, mu in settings}
        missing = sorted(wanted.difference(by_name))
        if missing:
            raise KeyError("Requested settings not found: " + ", ".join(missing))
        settings = [by_name[name] for name in sorted(wanted)]
    if args.max_cases:
        settings = settings[: args.max_cases]

    (outroot / "LOCAL_RERUN_README.md").write_text(
        "# Local DFTpy lambda-mu rerun\n\n"
        f"- Source root: `{source_root}`\n"
        f"- Output root: `{outroot}`\n"
        f"- Mode: `{args.mode}`\n"
        f"- ASE optimizer: `{args.ase_optimizer}`\n"
        "- Raw professor table is generated from `pristine_dftpy.out` and `pristine_vc_relaxed.vasp`.\n"
        "- Vacancy outputs are produced when mode is `full`, for later Gillan-style formation-energy checks.\n",
        encoding="utf-8",
    )
    settings_dir = outroot / "01_settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / "settings_lambda_mu_10x10.txt").write_text(
        "\n".join(f"{name} {lam:.1f} {mu:.1f}" for name, lam, mu in settings) + "\n",
        encoding="utf-8",
    )

    summary: list[dict[str, Any]] = []
    for idx, (setting, lam, mu) in enumerate(settings, start=1):
        source_case = runs_src / setting
        case_dir = runs_out / setting
        print(f"[{timestamp()}] [{idx:03d}/{len(settings):03d}] {setting} lambda={lam:.1f} mu={mu:.1f}", flush=True)
        copy_case_inputs(source_case, case_dir, refresh_inputs=args.refresh_inputs)
        manifest = load_json(case_dir / "point_manifest.json")
        pp_file = resolve_pp(case_dir, manifest)
        case_record: dict[str, Any] = {
            "setting": setting,
            "lambda": lam,
            "mu": mu,
            "case_dir": str(case_dir),
            "mode": args.mode,
            "status": "started",
        }
        failed_marker = case_dir / "LOCAL_RERUN_FAILED.txt"
        if failed_marker.exists() and not args.force:
            message = failed_marker.read_text(encoding="utf-8", errors="replace").strip()
            case_record["status"] = "FAILED_PREVIOUSLY"
            case_record["error"] = message
            print(f"[{timestamp()}] [SKIP_FAILED] {setting}: {message}", flush=True)
            summary.append(case_record)
            (case_dir / "local_case_status.json").write_text(
                json.dumps(case_record, indent=2, allow_nan=True) + "\n",
                encoding="utf-8",
            )
            (outroot / "local_rerun_progress.json").write_text(
                json.dumps(
                    {
                        "updated_at": timestamp(),
                        "completed_or_attempted": idx,
                        "total": len(settings),
                        "ok": sum(1 for row in summary if row.get("status") == "OK"),
                        "failed": sum(1 for row in summary if str(row.get("status", "")).startswith("FAILED")),
                        "records": summary,
                    },
                    indent=2,
                    allow_nan=True,
                )
                + "\n",
                encoding="utf-8",
            )
            continue
        try:
            pristine = run_one_structure(
                label="pristine",
                start_file=case_dir / "pristine_raw.vasp",
                case_dir=case_dir,
                pp_file=pp_file,
                manifest=manifest,
                ase_optimizer=args.ase_optimizer,
                force=args.force,
                abort_fmax=args.abort_fmax,
                abort_after_steps=args.abort_after_steps,
            )
            assert_physical_pristine_lattice(
                pristine,
                min_a0=args.min_pristine_a0,
                max_a0=args.max_pristine_a0,
            )
            case_record["pristine"] = pristine
            if args.mode == "full":
                vacancy = run_one_structure(
                    label="vacancy",
                    start_file=case_dir / "vacancy_start.vasp",
                    case_dir=case_dir,
                    pp_file=pp_file,
                    manifest=manifest,
                    ase_optimizer=args.ase_optimizer,
                    force=args.force,
                    abort_fmax=args.abort_fmax,
                    abort_after_steps=args.abort_after_steps,
                )
                n_pristine = int(manifest.get("pristine_n_atoms", 108))
                n_vacancy = int(manifest.get("vacancy_n_atoms", 107))
                ef = vacancy["total_energy_eV"] - (n_vacancy / n_pristine) * pristine["total_energy_eV"]
                kedf_ef = vacancy["kedf_energy_eV"] - (n_vacancy / n_pristine) * pristine["kedf_energy_eV"]
                case_record["vacancy"] = vacancy
                case_record["vacancy_formation_energy_eV"] = float(ef)
                case_record["kedf_vacancy_formation_energy_eV"] = float(kedf_ef)
                merged = {
                    "setting": setting,
                    "scan_type": "weight",
                    "relaxation_mode": "local_full_atom_and_cell_relaxation_vc_relax_equivalent",
                    "cell_basis": manifest.get("cell_basis", "conventional cubic fcc"),
                    "conventional_repeat": manifest.get("conventional_repeat", [3, 3, 3]),
                    "conventional_repeat_label": manifest.get("conventional_repeat_label", "conv_03x03x03"),
                    "pristine_n_atoms": n_pristine,
                    "vacancy_n_atoms": n_vacancy,
                    "vacancy_count": n_pristine - n_vacancy,
                    "spacing_A": resolve_spacing(case_dir, manifest),
                    "kedf": manifest.get("kedf", "TFVW"),
                    "kedf_x": float(manifest.get("kedf_x", lam)),
                    "kedf_y": float(manifest.get("kedf_y", mu)),
                    "xc": manifest.get("xc", "LDA"),
                    "pristine_energy_eV": pristine["total_energy_eV"],
                    "vacancy_energy_eV": vacancy["total_energy_eV"],
                    "vacancy_formation_energy_eV": float(ef),
                    "kedf_vacancy_formation_energy_eV": float(kedf_ef),
                    "pristine_final_fmax_eV_A": pristine["final_fmax_eV_A"],
                    "vacancy_final_fmax_eV_A": vacancy["final_fmax_eV_A"],
                    "pristine_stress_GPa": pristine["stress_GPa"],
                    "vacancy_stress_GPa": vacancy["stress_GPa"],
                    "pristine_cell_lengths_A": pristine["cell_lengths_A"],
                    "vacancy_cell_lengths_A": vacancy["cell_lengths_A"],
                    "pristine_cell_angles_deg": pristine["cell_angles_deg"],
                    "vacancy_cell_angles_deg": vacancy["cell_angles_deg"],
                    "source_dir": str(case_dir),
                    "formula": "E_f^vac = E_vac(Al107) - (107/108) E_pristine(Al108)",
                }
                (case_dir / "result.json").write_text(
                    json.dumps(merged, indent=2, allow_nan=True) + "\n",
                    encoding="utf-8",
                )
            case_record["status"] = "OK"
        except Exception as exc:
            case_record["status"] = "FAILED"
            case_record["error"] = repr(exc)
            (case_dir / "LOCAL_RERUN_FAILED.txt").write_text(repr(exc) + "\n", encoding="utf-8")
            print(f"[{timestamp()}] [FAILED] {setting}: {exc!r}", flush=True)
        finally:
            (case_dir / "local_case_status.json").write_text(
                json.dumps(case_record, indent=2, allow_nan=True) + "\n",
                encoding="utf-8",
            )
            summary.append(case_record)
            (outroot / "local_rerun_progress.json").write_text(
                json.dumps(
                    {
                        "updated_at": timestamp(),
                        "completed_or_attempted": idx,
                        "total": len(settings),
                        "ok": sum(1 for row in summary if row.get("status") == "OK"),
                        "failed": sum(1 for row in summary if row.get("status") == "FAILED"),
                        "records": summary,
                    },
                    indent=2,
                    allow_nan=True,
                )
                + "\n",
                encoding="utf-8",
            )
        if args.collect_every > 0 and idx % args.collect_every == 0:
            maybe_collect(outroot)

    maybe_collect(outroot)
    print(f"[{timestamp()}] [DONE] outroot={outroot}", flush=True)


if __name__ == "__main__":
    main()
