from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import zipfile
from pathlib import Path


GITHUB_URL = "https://github.com/DawsonLiu30/nano_tensile_TFvW"


def copy_file(source: Path, destination: Path) -> bool:
    if not source.exists() or not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def git_value(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def write_provenance_ini(path: Path, manifest: dict[str, object], structure: str) -> None:
    pp_name = Path(str(manifest.get("pp_file", "al.lda.recpot"))).name
    path.write_text(
        f"""# Human-readable equivalent of the programmatic DftpyCalculator input.
# Full ionic/cell relaxation uses ASE FrechetCellFilter + BFGS.

[JOB]
task = Optdensity
calctype = Energy Force Stress

[PATH]
pppath = ../../
cellpath = ./

[PP]
Al = {pp_name}

[CELL]
cellfile = {structure}
format = vasp

[GRID]
spacing = {float(manifest.get('spacing_A', 0.2)):.8f}

[EXC]
xc = {str(manifest.get('xc', 'LDA')).upper()}

[KEDF]
kedf = {manifest.get('kedf', 'TFVW')}
x = {float(manifest.get('kedf_x', 1.0)):.8f}
y = {float(manifest.get('kedf_y', 0.13)):.8f}

[OPT]
method = LBFGS
""",
        encoding="utf-8",
    )


def write_case_readme(path: Path, case: str, manifest: dict[str, object]) -> None:
    path.write_text(
        f"""DFTpy divacancy case: {case}

Initial minimum-image distance:
  {float(manifest.get('pair_distance_A', float('nan'))):.8f} A

Input:
  pristine_raw.vasp
  divacancy_start.vasp
  dftpy_pristine_input.ini
  dftpy_divacancy_input.ini
  dftpy_pristine_calculator_config.json (new pipeline runs)
  dftpy_divacancy_calculator_config.json (new pipeline runs)

Output:
  pristine_dftpy.out
  divacancy_dftpy.out
  pristine_relax.log
  divacancy_relax.log
  pristine_vc_relaxed.vasp
  divacancy_vc_relaxed.vasp
  result.json

Method:
  DFTpy, XC={manifest.get('xc')}, KEDF={manifest.get('kedf')},
  lambda/x={manifest.get('kedf_x')}, mu/y={manifest.get('kedf_y')},
  spacing={manifest.get('spacing_A')} A, full atom+cell relaxation.

The .ini files document the DftpyCalculator input. The full relaxation is
driven through ASE FrechetCellFilter and the optimizer recorded in result.json.
""",
        encoding="utf-8",
    )


def copy_case(source: Path, destination: Path) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = source / "point_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    direct_names = [
        "point_manifest.json",
        "result.json",
        "pristine_raw.vasp",
        "pristine_raw.xyz",
        "pristine_dftpy.out",
        "pristine_relax.log",
        "pristine_relax.traj",
        "pristine_vc_relaxed.vasp",
        "pristine_vc_relaxed.xyz",
        "pristine_relaxed.vasp",
        "pristine_relaxed.xyz",
        "divacancy_start.vasp",
        "divacancy_start.xyz",
        "divacancy_dftpy.out",
        "divacancy_relax.log",
        "divacancy_relax.traj",
        "divacancy_vc_relaxed.vasp",
        "divacancy_vc_relaxed.xyz",
        "divacancy_relaxed.vasp",
        "divacancy_relaxed.xyz",
        "dftpy_pristine_input.ini",
        "dftpy_divacancy_input.ini",
        "dftpy_pristine_calculator_config.json",
        "dftpy_divacancy_calculator_config.json",
    ]
    for name in direct_names:
        copy_file(source / name, destination / name)

    # Normalize legacy generic vacancy filenames used by the 2026-06-16 pilot.
    legacy_map = {
        "vacancy_dftpy.out": "divacancy_dftpy.out",
        "vacancy_relax.log": "divacancy_relax.log",
        "vacancy_relax.traj": "divacancy_relax.traj",
        "vacancy_vc_relaxed.vasp": "divacancy_vc_relaxed.vasp",
        "vacancy_vc_relaxed.xyz": "divacancy_vc_relaxed.xyz",
        "vacancy_relaxed.vasp": "divacancy_relaxed.vasp",
        "vacancy_relaxed.xyz": "divacancy_relaxed.xyz",
    }
    for old, new in legacy_map.items():
        if not (destination / new).exists():
            copy_file(source / old, destination / new)

    if not (destination / "divacancy_start.vasp").exists():
        copy_file(source / "vacancy_start.vasp", destination / "divacancy_start.vasp")
    if not (destination / "divacancy_start.xyz").exists():
        copy_file(source / "vacancy_start.xyz", destination / "divacancy_start.xyz")

    if not (destination / "dftpy_pristine_input.ini").exists():
        write_provenance_ini(destination / "dftpy_pristine_input.ini", manifest, "pristine_raw.vasp")
    if not (destination / "dftpy_divacancy_input.ini").exists():
        write_provenance_ini(destination / "dftpy_divacancy_input.ini", manifest, "divacancy_start.vasp")
    write_case_readme(destination / "README_CASE.txt", source.name, manifest)

    expected = [
        "README_CASE.txt",
        "point_manifest.json",
        "result.json",
        "pristine_raw.vasp",
        "divacancy_start.vasp",
        "dftpy_pristine_input.ini",
        "dftpy_divacancy_input.ini",
        "pristine_dftpy.out",
        "divacancy_dftpy.out",
        "pristine_relax.log",
        "divacancy_relax.log",
        "pristine_vc_relaxed.vasp",
        "divacancy_vc_relaxed.vasp",
    ]
    missing = [name for name in expected if not (destination / name).exists()]
    return {
        "case": source.name,
        "source_dir": str(source),
        "package_dir": str(destination),
        "pair_distance_A": manifest.get("pair_distance_A"),
        "lambda": manifest.get("kedf_x"),
        "mu": manifest.get("kedf_y"),
        "missing_required_files": ";".join(missing),
        "complete": not missing,
    }


def zip_directory(source: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(source.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a professor-facing divacancy package.")
    parser.add_argument("--dftpy-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--analysis", default="")
    parser.add_argument("--scheduler-logs", default="")
    args = parser.parse_args()

    dftpy_root = Path(args.dftpy_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    repo = Path(args.repo).expanduser().resolve()
    analysis = Path(args.analysis).expanduser().resolve() if args.analysis else None
    scheduler_logs = Path(args.scheduler_logs).expanduser().resolve() if args.scheduler_logs else None

    if output.exists():
        shutil.rmtree(output)
    start = output / "00_START_HERE"
    tables = output / "01_TABLES"
    figures = output / "02_FIGURES"
    calculations = output / "03_CALCULATIONS" / "DFTpy" / "pair_scan"
    notebooks = output / "04_NOTEBOOKS"
    code = output / "05_CODE_POINTERS"
    logs = output / "06_SCHEDULER_LOGS"
    for folder in (start, tables, figures, calculations, notebooks, code, logs):
        folder.mkdir(parents=True, exist_ok=True)

    rows = []
    for case_dir in sorted((dftpy_root / "pair_scan").glob("pair_*")):
        if case_dir.is_dir() and (case_dir / "point_manifest.json").exists():
            rows.append(copy_case(case_dir, calculations / case_dir.name))

    with (tables / "case_file_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    for pattern in ("*.csv", "manifest.json", "settings_pair_scan.txt"):
        for source in dftpy_root.glob(pattern):
            copy_file(source, tables / source.name)

    if analysis and analysis.exists():
        for source in analysis.iterdir():
            target = figures / source.name if source.suffix.lower() in {".png", ".pdf"} else tables / source.name
            copy_file(source, target)

    for source in (repo / "notebooks").glob("*.ipynb"):
        copy_file(source, notebooks / source.name)
    copy_file(repo / "notebooks" / "README.md", notebooks / "README.md")
    copy_file(repo / "DIVACANCY_ENERGY_AND_PBC_DEFINITIONS.md", start / "ENERGY_AND_PBC_DEFINITIONS.md")

    branch = git_value(repo, "branch", "--show-current")
    commit = git_value(repo, "rev-parse", "HEAD")
    (code / "GITHUB_AND_CODE_POINTERS.txt").write_text(
        f"""GitHub repository:
  {GITHUB_URL}

Branch:
  {branch}

Commit:
  {commit}

Authoritative production scripts:
  scripts/prepare_dftpy_divacancy_rscan_20260616.py
  scripts/run_dftpy_vcrelax_vacancy_one.py
  scripts/collect_dftpy_conventional_vacancy.py
  scripts/analyze_divacancy_geometry_strain.py

The notebooks in 04_NOTEBOOKS demonstrate generation, output parsing, and
formation-energy calculation. The GitHub commit is the authoritative code
delivery; this package does not duplicate the complete repository.
""",
        encoding="utf-8",
    )

    if scheduler_logs and scheduler_logs.exists():
        for source in scheduler_logs.iterdir():
            copy_file(source, logs / source.name)

    complete_count = sum(bool(row["complete"]) for row in rows)
    (start / "README.txt").write_text(
        f"""DFTpy divacancy pilot package

Status:
  Pilot calculation generated before final lambda/mu + QE calibration.
  Do not treat this package as the final production conclusion.

Cases:
  {complete_count}/{len(rows)} case folders contain the required direct input/output files.

Start here:
  00_START_HERE/ENERGY_AND_PBC_DEFINITIONS.md
  01_TABLES/case_file_audit.csv
  03_CALCULATIONS/DFTpy/pair_scan/<case>/README_CASE.txt
  04_NOTEBOOKS/
  05_CODE_POINTERS/GITHUB_AND_CODE_POINTERS.txt

Main figure definition:
  E_2vac(r) = E_defect(N-2,r) - ((N-2)/N) E_pristine(N)
  Per-vacancy energy is intentionally not duplicated in the main figures.
""",
        encoding="utf-8",
    )

    zip_path = output.with_suffix(".zip")
    zip_directory(output, zip_path)
    print(f"cases={len(rows)} complete={complete_count}")
    print(f"package={output}")
    print(f"zip={zip_path}")


if __name__ == "__main__":
    main()

