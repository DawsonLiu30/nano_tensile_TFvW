from __future__ import annotations

import json
import shutil
from pathlib import Path


DESKTOP = Path.home() / "Desktop"
REPO = Path(__file__).resolve().parents[1]
LOCAL_QE_PSEUDO = DESKTOP / "LATEST_VACANCY_BENCHMARK_20260601" / "raw" / "QE" / "psp" / "Al_PAW_PBE.UPF"

SRC = DESKTOP / "DFTPY_VACLM_PROF_DELIVERY_20260629_ISERVICE_ONLY"
STAGE = SRC / "_REMOTE_PROVENANCE_STAGE_20260629"
STAGE_TFVW = STAGE / "TFvW_test"
STAGE_CODE = STAGE / "code"

NAS = DESKTOP / "DFTPY_VACLM_ISERVICE_NAS_20260630"
QE = DESKTOP / "QE_VACLM_REFERENCE_20260630"


def remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def copytree(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def count_files(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.rglob(pattern)) if root.exists() else 0


def count_dirs(root: Path) -> int:
    return sum(1 for p in root.iterdir() if p.is_dir()) if root.exists() else 0


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def build_nas_package() -> None:
    remove_dir(NAS)
    NAS.mkdir(parents=True)

    # Professor-facing tables and simple figures.
    copytree(SRC / "01_RAW_TABLE", NAS / "01_RAW_TABLE")
    copytree(SRC / "02_SIMPLE_MAPS", NAS / "02_SIMPLE_MAPS")

    # Use the remote provenance stage as the primary raw-case source because it
    # contains the trajectory files that were not merged into the old delivery.
    copytree(STAGE_TFVW / "03_runs", NAS / "03_RAW_CASES" / "03_runs")

    # Metadata from the previous package plus a fresh audit.
    copytree(SRC / "04_RUN_METADATA", NAS / "04_RUN_METADATA")
    copytree(SRC / "06_REFERENCE_NOTES", NAS / "06_REFERENCE_NOTES")
    copytree(SRC / "07_EVALUATION_SLIDES", NAS / "07_EVALUATION_SLIDES")

    # Reproducible code provenance.
    copytree(STAGE_CODE / "scripts", NAS / "05_SCRIPTS_USED" / "scripts")
    copytree(STAGE_CODE / "app", NAS / "05_SCRIPTS_USED" / "app")
    copy_file(REPO / "scripts" / "pull_iservice_vaclm_full_provenance_20260629.ps1", NAS / "05_SCRIPTS_USED" / "pull_iservice_vaclm_full_provenance_20260629.ps1")
    copy_file(REPO / "scripts" / "package_iservice_vaclm_nas_and_qe_20260630.py", NAS / "05_SCRIPTS_USED" / "package_iservice_vaclm_nas_and_qe_20260630.py")
    copy_file(REPO / "scripts" / "push_qe_vaclm_reference_to_iservice_20260630.ps1", NAS / "05_SCRIPTS_USED" / "push_qe_vaclm_reference_to_iservice_20260630.ps1")

    audit = {
        "package_root": str(NAS),
        "source_delivery": str(SRC),
        "remote_stage": str(STAGE),
        "raw_case_dirs": count_dirs(NAS / "03_RAW_CASES" / "03_runs"),
        "result_json": count_files(NAS / "03_RAW_CASES" / "03_runs", "result.json"),
        "traj_files": count_files(NAS / "03_RAW_CASES" / "03_runs", "*.traj"),
        "relax_logs": count_files(NAS / "03_RAW_CASES" / "03_runs", "*relax*.log"),
        "dftpy_ini_inputs": count_files(NAS / "03_RAW_CASES" / "03_runs", "*.ini"),
        "submit_script_snapshots": count_files(NAS / "03_RAW_CASES" / "03_runs", "submit_script_used*.sh"),
        "runner_exists": (NAS / "05_SCRIPTS_USED" / "scripts" / "run_dftpy_vcrelax_vacancy_matrix_one.py").exists(),
        "engine_exists": (NAS / "05_SCRIPTS_USED" / "app" / "dft_engine.py").exists(),
    }
    write_text(NAS / "04_RUN_METADATA" / "ISERVICE_NAS_PACKAGE_AUDIT.json", json.dumps(audit, indent=2))

    write_text(
        NAS / "README_PACKAGE.md",
        f"""# DFTpy VACLM iService NAS Package

This package is the cleaned iService production-provenance delivery for the
DFTpy TFvW lambda-mu single-vacancy scan.

## Recommended Use

Use this package as the main NAS upload for Professor Luder.

It is preferred over the local rerun package because it preserves the original
iService production calculation folders, submission-script snapshots, stdout
logs, result files, and trajectory files.

## Layout

- `01_RAW_TABLE`: professor-facing Total / KEDF / lattice table and raw CSV files.
- `02_SIMPLE_MAPS`: simple maps made from the table data.
- `03_RAW_CASES/03_runs`: one folder per lambda-mu point.
- `04_RUN_METADATA`: audit and provenance notes.
- `05_SCRIPTS_USED`: Python runner, DFT engine, and pull/package scripts.
- `06_REFERENCE_NOTES`: notes about reference values and method interpretation.
- `07_EVALUATION_SLIDES`: concise evaluation slides, if present.

## Current Audit

```json
{json.dumps(audit, indent=2)}
```

## Important Interpretation

The DFTpy input `.ini` files contain `task = Optdensity` because that is the
inner density-optimization task used by the DFTpy calculator. The production
workflow is still a relaxation workflow: the Python runner wraps the DFTpy
calculator with ASE cell/atom relaxation and writes `*_relax.log`, final
structures, and `.traj` files where available.
""",
    )

    write_text(
        NAS / "04_RUN_METADATA" / "ISERVICE_VS_LOCAL_RERUN_DECISION.md",
        """# iService Production Package vs Local Official-Style Rerun

## Decision

Use this iService package as the main NAS delivery.

## Reason

The advisor asked for source directories, input files, output files, and
critical computational details. This package preserves the original iService
production folders and submission-script snapshots, so it is the correct
provenance package.

## Role of the Local Rerun

The local official-style rerun is useful as a verification/cross-check because
it follows the DFTpy relaxation tutorial style explicitly and writes trajectory
files clearly. However, it should not replace the iService production data in
the main advisor delivery.

## Practical Summary

- Main package for NAS: `DFTPY_VACLM_ISERVICE_NAS_20260630`
- Supporting verification only: `DFTPY_VACLM_OFFICIAL_RELAX_NAS_20260630`
- QE reference workflow: `QE_VACLM_REFERENCE_20260630`
""",
    )


def read_vasp(path: Path) -> tuple[list[list[float]], list[str], list[list[float]]]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    scale = float(lines[1])
    cell = [[float(x) * scale for x in lines[i].split()[:3]] for i in range(2, 5)]
    symbols = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    coord_mode = lines[7].lower()
    coords = [[float(x) for x in lines[8 + i].split()[:3]] for i in range(sum(counts))]
    atoms: list[str] = []
    for sym, n in zip(symbols, counts):
        atoms.extend([sym] * n)
    if coord_mode.startswith("d"):
        cart: list[list[float]] = []
        for frac in coords:
            cart.append([
                frac[0] * cell[0][j] + frac[1] * cell[1][j] + frac[2] * cell[2][j]
                for j in range(3)
            ])
        coords = cart
    return cell, atoms, coords


def write_qe_input(path: Path, vasp: Path, calculation: str = "vc-relax") -> None:
    cell, atoms, coords = read_vasp(vasp)
    nat = len(atoms)
    positions = "\n".join(
        f"{sym:2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}" for sym, xyz in zip(atoms, coords)
    )
    cell_txt = "\n".join(f"{v[0]: .10f} {v[1]: .10f} {v[2]: .10f}" for v in cell)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""&CONTROL
  calculation = '{calculation}'
  prefix = '{path.stem}'
  outdir = './tmp'
  pseudo_dir = './pseudo'
  tstress = .true.
  tprnfor = .true.
  verbosity = 'high'
/
&SYSTEM
  ibrav = 0
  nat = {nat}
  ntyp = 1
  ecutwfc = 58.8
  ecutrho = 470.4
  occupations = 'smearing'
  smearing = 'mv'
  degauss = 0.02
/
&ELECTRONS
  conv_thr = 1.0d-8
  mixing_beta = 0.7
/
&IONS
/
&CELL
  press = 0.0
/
ATOMIC_SPECIES
Al 26.9815385 Al_PAW_PBE.UPF

CELL_PARAMETERS angstrom
{cell_txt}

ATOMIC_POSITIONS angstrom
{positions}

K_POINTS automatic
3 3 3 0 0 0
""",
        encoding="utf-8",
        newline="\n",
    )


def build_qe_reference() -> None:
    remove_dir(QE)
    QE.mkdir(parents=True)
    source_case = NAS / "03_RAW_CASES" / "03_runs" / "tfvw_lam0p5_mu0p5"
    if not source_case.exists():
        source_case = next((NAS / "03_RAW_CASES" / "03_runs").glob("tfvw_lam*_mu*"))

    pristine = source_case / "pristine_raw.vasp"
    vacancy = source_case / "vacancy_start.vasp"
    copy_file(pristine, QE / "structures" / "pristine_raw.vasp")
    copy_file(vacancy, QE / "structures" / "vacancy_start.vasp")
    copy_file(LOCAL_QE_PSEUDO, QE / "pseudo" / "Al_PAW_PBE.UPF")
    write_qe_input(QE / "pristine_vcrelax" / "pw.in", pristine)
    write_qe_input(QE / "vacancy_vcrelax" / "pw.in", vacancy)

    write_text(
        QE / "submit_qe_vacancy_reference_array.sh",
        """#!/bin/bash
#SBATCH --job-name QEVACREF
#SBATCH --output logs_submit/%x_%A_%a.out
#SBATCH --error  logs_submit/%x_%A_%a.err
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=128G
#SBATCH --partition=ct56
#SBATCH --no-requeue
#SBATCH --account=MST114175
#SBATCH --array=0-1%2

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd -P)}"
cd "$ROOT"
mkdir -p logs_submit

module purge || true
module load gcc/10.5.0 || true

PWX="${PWX:-/work/dawson666/q-e-qe-7.3.1/PW/src/pw.x}"
if [ ! -x "$PWX" ]; then
  echo "[ERROR] pw.x not executable: $PWX"
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0) CASE_DIR="pristine_vcrelax" ;;
  1) CASE_DIR="vacancy_vcrelax" ;;
  *) echo "[ERROR] unknown task ${SLURM_ARRAY_TASK_ID}"; exit 3 ;;
esac

cd "$CASE_DIR"
mkdir -p tmp pseudo

if [ -n "${PSEUDO_SRC:-}" ]; then
  CANDIDATES=("$PSEUDO_SRC")
else
  CANDIDATES=(
    "$ROOT/pseudo/Al_PAW_PBE.UPF"
    "../pseudo/Al_PAW_PBE.UPF"
    "/work/dawson666/qe_cases/qe_runs/psp/Al_PAW_PBE.UPF"
    "/work/dawson666/qe_cases/qe_runs/psp/Al.pbe-n-kjpaw_psl.1.0.0.UPF"
    "/work/dawson666/qe_cases/qe_runs/psp/al_pbe_v1.uspp.F.UPF"
    "/work/dawson666/qe_cases/qe_runs/psp/Al.pbe-spn-kjpaw_psl.1.0.0.UPF"
  )
fi
PSEUDO_SRC=""
for cand in "${CANDIDATES[@]}"; do
  if [ -s "$cand" ]; then
    PSEUDO_SRC="$cand"
    break
  fi
done
if [ ! -s "$PSEUDO_SRC" ]; then
  echo "[ERROR] set PSEUDO_SRC to a valid Al PBE PAW UPF"
  echo "[ERROR] tried candidates:"
  printf '  %s\\n' "${CANDIDATES[@]}"
  exit 4
fi
cp -f "$PSEUDO_SRC" pseudo/Al_PAW_PBE.UPF

echo "[INFO] CASE_DIR=$CASE_DIR"
echo "[INFO] PWX=$PWX"
echo "[INFO] PSEUDO_SRC=$PSEUDO_SRC"
echo "[INFO] OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-28}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-28}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-28}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-28}"
"$PWX" -in pw.in > pw.out
""",
    )

    write_text(
        QE / "README_RUN_ON_ISERVICE.md",
        """# QE single-vacancy reference from VACLM starting geometry

This folder prepares a KSDFT/QE reference for the same pristine and single-vacancy
starting structures used in the DFTpy lambda-mu scan.

It intentionally runs only one pristine and one vacancy vc-relax calculation.
QE does not depend on the DFTpy lambda/mu parameters, so running 100 QE jobs for
the 100 lambda-mu points would be redundant unless we explicitly want to test
DFTpy-relaxed geometries one by one.

## Upload

From local PowerShell:

```powershell
scp -O -r C:\\Users\\dawso\\Desktop\\QE_VACLM_REFERENCE_20260630 dawson666@twnia3.nchc.org.tw:/work/dawson666/qe_cases/qe_runs/
```

## Submit on iService

```bash
cd /work/dawson666/qe_cases/qe_runs/QE_VACLM_REFERENCE_20260630
sbatch submit_qe_vacancy_reference_array.sh
```

If the pseudopotential path differs, submit with:

```bash
PSEUDO_SRC=/path/to/Al_PAW_PBE.UPF sbatch submit_qe_vacancy_reference_array.sh
```
""",
    )


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing source package: {SRC}")
    if not STAGE_TFVW.exists():
        raise SystemExit(f"Missing remote provenance stage: {STAGE_TFVW}")
    build_nas_package()
    build_qe_reference()
    print(f"[NAS] {NAS}")
    print((NAS / "04_RUN_METADATA" / "ISERVICE_NAS_PACKAGE_AUDIT.json").read_text())
    print(f"[QE ] {QE}")


if __name__ == "__main__":
    main()
