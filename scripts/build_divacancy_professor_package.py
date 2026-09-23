from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from divacancy_analysis_checks import qualify_case


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
            ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        ).strip()
    except Exception:
        return "unknown"


def write_provenance_ini(path: Path, manifest: dict[str, object], structure: str) -> None:
    pp_name = Path(str(manifest.get("pp_file", "al.lda.recpot"))).name
    path.write_text(
        f"""# Human-readable equivalent of the programmatic DftpyCalculator input.
# Reconstructed documentation, not an original input file.
# Full ionic/cell relaxation uses ASE FrechetCellFilter; see result.json for optimizer.

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
x = {float(manifest.get('kedf_x', float('nan'))):.8f}
y = {float(manifest.get('kedf_y', float('nan'))):.8f}

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
    qualification = qualify_case(source)
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
        **qualification,
    }


def zip_directory(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite saved archive: {destination}")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
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

    if not (dftpy_root / "pair_scan").is_dir():
        raise FileNotFoundError(f"Missing pair_scan: {dftpy_root}")
    if output.exists() or output.with_suffix(".zip").exists():
        raise FileExistsError(f"Choose a new output path; preserving existing package: {output}")
    if output == dftpy_root or dftpy_root in output.parents:
        raise ValueError("Package output must be outside the source dataset")
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
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)

    for pattern in ("*.csv", "manifest.json", "settings_pair_scan.txt"):
        for source in dftpy_root.glob(pattern):
            # Imported tables are historical provenance, not fresh acceptance.
            copy_file(source, tables / "SOURCE_TABLES_UNREASSESSED" / source.name)

    for source in (dftpy_root / "pseudopotentials").glob("*"):
        copy_file(source, output / "03_CALCULATIONS" / "DFTpy" / "pseudopotentials" / source.name)

    if analysis and analysis.exists():
        for source in analysis.iterdir():
            target = figures / source.name if source.suffix.lower() in {".png", ".pdf"} else tables / source.name
            copy_file(source, target)

    for source in (repo / "notebooks").glob("*.ipynb"):
        copy_file(source, notebooks / source.name)
    copy_file(repo / "notebooks" / "README.md", notebooks / "README.md")
    copy_file(repo / "DIVACANCY_ENERGY_AND_PBC_DEFINITIONS.md", start / "ENERGY_AND_PBC_DEFINITIONS.md")

    commit = git_value(repo, "rev-parse", "HEAD")
    # No status, diff, ls-files or other index-touching Git operation. SHA-256
    # of the actual included files is the code-delivery identity.
    (code / "WORKTREE_STATUS.txt").write_text(
        "working_tree_not_committed; source SHA manifest authoritative\n", encoding="utf-8")
    source_tree = code / "source_tree"
    source_files = []
    for name in ("app", "scripts", "tests", "notebooks", "profess_input_examples"):
        source_files.extend(path for path in (repo / name).rglob("*") if path.is_file()
                            and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"})
    source_files.extend(path for path in repo.iterdir() if path.is_file() and path.suffix.lower() in
                        {".py", ".sh", ".ps1", ".sbatch", ".md", ".json", ".yml", ".yaml", ".toml", ".ini", ".txt", ".recpot", ".upf"})
    source_hashes = {}
    for source in sorted(set(source_files)):
        relative = source.relative_to(repo)
        copy_file(source, source_tree / relative)
        source_hashes[str(relative)] = hashlib.sha256((source_tree / relative).read_bytes()).hexdigest()
    (code / "SOURCE_SHA256.json").write_text(json.dumps(source_hashes, indent=2) + "\n", encoding="utf-8")
    for source in (dftpy_root / "scripts_used").glob("*"):
        copy_file(source, code / "PRODUCTION_SCRIPTS_USED" / source.name)
    (code / "GITHUB_AND_CODE_POINTERS.txt").write_text(
        f"""GitHub repository:
  {GITHUB_URL}

Commit:
  {commit}

Authoritative production scripts:
  scripts/prepare_dftpy_divacancy_rscan_20260616.py
  scripts/run_dftpy_vcrelax_vacancy_one.py
  scripts/collect_dftpy_conventional_vacancy.py
  scripts/analyze_divacancy_geometry_strain.py

The Git commit identifies the base revision only. The worktree is not committed.
The current app/scripts/tests/notebooks and root text configuration/documentation
files are included in source_tree. SOURCE_SHA256.json identifies their bytes.
Available original production scripts are copied into PRODUCTION_SCRIPTS_USED.
The notebooks require AL_DEFECTS_REPO pointing to a complete working repository.
""",
        encoding="utf-8",
    )

    if scheduler_logs and scheduler_logs.exists():
        for source in scheduler_logs.iterdir():
            copy_file(source, logs / source.name)

    complete_count = sum(bool(row["complete"]) for row in rows)
    counts = {status: sum(row["status"] == status for row in rows)
              for status in ("missing", "failed", "unconverged", "qualified")}
    state = "All cases numerically qualified" if rows and counts["qualified"] == len(rows) else "Incomplete or mixed qualification"
    (start / "README.txt").write_text(
        f"""DFTpy divacancy evidence package

Status:
  {state}.
  {json.dumps(counts, sort_keys=True)}
  Qualification is recomputed from the source inputs, outputs, trajectory,
  reference settings and combined atom/cell optimizer logs at packaging time.
  Numerical qualification does not establish thesis acceptance, finite-size
  convergence, electronic-density convergence, or an independently calibrated
  QE comparison. The *_dftpy.out files are saved calculator energy/stress
  summaries, not full electronic-density iteration traces.
  Source: {dftpy_root}

Cases:
  {complete_count}/{len(rows)} case folders contain the required direct input/output files.

Start here:
  00_START_HERE/ENERGY_AND_PBC_DEFINITIONS.md
  01_TABLES/case_file_audit.csv
  01_TABLES/SOURCE_TABLES_UNREASSESSED/ (historical source tables)
  03_CALCULATIONS/DFTpy/pair_scan/<case>/README_CASE.txt
  04_NOTEBOOKS/
  05_CODE_POINTERS/GITHUB_AND_CODE_POINTERS.txt

Main figure definition:
  E_2vac(r) = E_defect(N-2,r) - ((N-2)/N) E_pristine(N)
  Per-vacancy energy is intentionally not duplicated in the main figures.
  r is the initial minimum-image distance under PBC, not a final relaxed distance.

Comparison limitations:
  Never connect mixed directions or join QE by distance alone. QE comparisons
  require separately verified matching initial geometry, XC, relaxation and
  energy/reference definitions, plus convergence and pseudopotential evidence.
  Do not derive a formal binding energy from the historical monovacancy
  600 eV / 0.250343 A scan and the 0.20 A divacancy scan: the grids differ.
  Binding needs matching code, PP hash, XC/KEDF/weights, grid, cell, pressure,
  energy reference and convergence tolerance before combining mono/divacancy.
""",
        encoding="utf-8",
    )

    package_hashes = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted(output.rglob("*")) if path.is_file()}
    (output / "SHA256SUMS.json").write_text(json.dumps(package_hashes, indent=2) + "\n", encoding="utf-8")
    zip_path = output.with_suffix(".zip")
    zip_directory(output, zip_path)
    print(f"cases={len(rows)} complete={complete_count}")
    print(f"package={output}")
    print(f"zip={zip_path}")


if __name__ == "__main__":
    main()
