from __future__ import annotations

import argparse
import csv
import shutil
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path.home() / "Desktop"
DEFAULT_LOCAL_PULL = DESKTOP / "latest_professor_pull_20260511"
DEFAULT_NANO_PULL = DESKTOP / "vacancy_qe_ofdft_results_2026-04-25"
DEFAULT_OUT = DESKTOP / f"professor_raw_data_browser_{date.today():%Y%m%d}"


WORK_ITEMS = [
    {
        "folder": "01_QE_bulk_B_EOS_convergence_FULL_RAW",
        "title": "QE bulk EOS and B0 convergence",
        "source": DEFAULT_LOCAL_PULL / "qe_bulk_b_convergence_20260506",
        "remote": "/gpfs-work/dawson666/qe_cases/qe_runs/qe_bulk_b_convergence_20260506",
        "look": [
            "ecut_scan/*/a0_*/scf.in and scf.out",
            "kmesh_scan/*/a0_*/scf.in and scf.out",
            "final_reference/high_kmesh_e0600/*/a0_*/scf.in and scf.out",
            "processed_bulk_B_convergence/*.csv and *.png",
            "parse_qe_bulk_b_convergence.py, analyze_qe_bulk_high_kmesh.py",
        ],
    },
    {
        "folder": "02_QE_vacancy_formation_convergence_FULL_RAW",
        "title": "QE vacancy formation energy convergence",
        "source": DEFAULT_LOCAL_PULL / "qe_vacancy_convergence_20260506",
        "remote": "/gpfs-work/dawson666/qe_cases/qe_runs/qe_vacancy_convergence_20260506",
        "look": [
            "ecut_scan/*/pristine_scf/scf.in and scf.out",
            "ecut_scan/*/vacancy_relax/relax.in and relax.out",
            "kmesh_scan/*/pristine_scf/scf.in and scf.out",
            "kmesh_scan/*/vacancy_relax/relax.in and relax.out",
            "dense_k05_ecut_scan/*/pristine_scf/scf.in and scf.out",
            "dense_k05_ecut_scan/*/vacancy_relax/relax.in and relax.out",
            "processed_vacancy_convergence/*.csv",
        ],
    },
    {
        "folder": "03_DFTpy_vacancy_spacing_fixed_QE_a0_FULL_RAW",
        "title": "DFTpy vacancy spacing convergence, fixed QE a0",
        "source": DEFAULT_LOCAL_PULL / "dftpy_vacancy_convergence_primitive4_qe_a0_20260508",
        "remote": "/gpfs-work/dawson666/dftpy_project/relax/dftpy45/results/dftpy_vacancy_convergence_primitive4_qe_a0_20260508",
        "look": [
            "spacing_scan/*/point_manifest.json",
            "spacing_scan/*/result.json",
            "spacing_scan/*/pristine_dftpy.out",
            "spacing_scan/*/vacancy_dftpy.out",
            "spacing_scan/*/vacancy_relax.log",
            "summary.csv, summary.txt, ef_vac_vs_spacing.png",
        ],
    },
    {
        "folder": "04_DFTpy_vacancy_spacing_DFTpy_own_a0_FULL_RAW",
        "title": "DFTpy vacancy spacing convergence, DFTpy own a0",
        "source": DEFAULT_LOCAL_PULL / "dftpy_vacancy_convergence_primitive4_20260508",
        "remote": "/gpfs-work/dawson666/dftpy_project/relax/dftpy45/results/dftpy_vacancy_convergence_primitive4_20260508",
        "look": [
            "spacing_scan/*/point_manifest.json",
            "spacing_scan/*/result.json",
            "spacing_scan/*/pristine_dftpy.out",
            "spacing_scan/*/vacancy_dftpy.out",
            "spacing_scan/*/vacancy_relax.log",
            "summary.csv, summary.txt, ef_vac_vs_spacing.png",
        ],
    },
    {
        "folder": "05_DFTpy_vacancy_primitive_size_fixed_QE_a0_FULL_RAW",
        "title": "DFTpy vacancy primitive supercell-size convergence",
        "source": DEFAULT_LOCAL_PULL / "dftpy_vacancy_size_primitive_qe_a0_spacing020_20260511",
        "remote": "/gpfs-work/dawson666/dftpy_project/relax/dftpy45/results/dftpy_vacancy_size_primitive_qe_a0_spacing020_20260511",
        "look": [
            "size_scan/prim_*/point_manifest.json",
            "size_scan/prim_*/result.json",
            "size_scan/prim_*/pristine_dftpy.out",
            "size_scan/prim_*/vacancy_dftpy.out",
            "size_scan/prim_*/vacancy_relax.log",
            "dftpy_primitive_size_summary.csv",
            "dftpy_primitive_size_summary_with_fmax.csv if present",
        ],
    },
    {
        "folder": "06_QE_OFDFT_faceted_nanostructure_actual_points_RAW",
        "title": "QE/OFDFT vacancy faceted nanostructure actual-point archive",
        "source": DEFAULT_NANO_PULL,
        "remote": "local archive from earlier pull",
        "look": [
            "comparison_vacancy_ofdft_vs_qe.csv",
            "all_vacancy_energy_data.csv if present",
            "qe_vacancy_raw_data.csv if present",
            "ofdft_vacancy_raw_data.csv if present",
            "Fig*.png if present",
        ],
    },
]


def copytree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copyfile(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def has_job_done(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        return "JOB DONE" in path.read_text(errors="ignore")
    except Exception:
        return False


def classify_file(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name.endswith((".in", ".ini")) or name in {"poscar", "point_manifest.json", "manifest.json"}:
        return "input"
    if name.endswith((".out", ".log", ".err")) or name.startswith("slurm"):
        return "output_log"
    if suffix in {".py", ".sh", ".sbatch", ".ps1"}:
        return "script"
    if suffix in {".csv", ".json", ".txt", ".md"}:
        return "processed_or_metadata"
    if suffix in {".png", ".jpg", ".jpeg", ".pdf", ".pptx"}:
        return "figure_or_report"
    if suffix in {".vasp", ".xyz", ".xsf", ".traj"}:
        return "structure_or_density"
    return "other"


def write_work_item_readme(item: dict[str, object], dst: Path) -> None:
    title = str(item["title"])
    source = Path(item["source"])
    remote = str(item["remote"])
    look = "\n".join(f"- `{entry}`" for entry in item["look"])
    text = f"""# {title}

## What This Folder Contains

This is a full local copy of the raw work-item folder. It is intended for quick inspection during a meeting.

## Original Local Source

```text
{source}
```

## Remote Source

```text
{remote}
```

## Where To Look

{look}

## Notes

- QE input files are usually `*.in`; QE outputs are usually `*.out`.
- DFTpy case metadata is usually `point_manifest.json`; DFTpy raw outputs include `pristine_dftpy.out`, `vacancy_dftpy.out`, and `vacancy_relax.log`.
- Processed summaries are usually `summary.csv`, `*_summary.csv`, or files inside `processed_*` folders.
"""
    (dst / "README_WHERE_TO_LOOK.md").write_text(text, encoding="utf-8")


def write_index(out_dir: Path, copied: list[dict[str, str]], warnings: list[str]) -> None:
    index_dir = out_dir / "00_OPEN_THIS_FIRST"
    index_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": str(path.relative_to(out_dir)).replace("\\", "/"),
                    "category": classify_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    with (index_dir / "COMPLETE_FILE_INDEX.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "category", "size_bytes"])
        writer.writeheader()
        writer.writerows(rows)

    copied_lines = "\n".join(
        f"| {entry['folder']} | {entry['title']} | `{entry['source']}` |"
        for entry in copied
    )
    warning_text = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- No obvious missing critical folders were detected."

    readme = f"""# Professor Raw Data Browser

Date: {date.today():%Y-%m-%d}

This folder is organized for live inspection with the professor. It contains complete local copies of the currently pulled QE and DFTpy work-item folders, plus processed outputs and report files.

## Work Items

| Folder | Work item | Original local source |
|---|---|---|
{copied_lines}

## File Index

Open:

```text
00_OPEN_THIS_FIRST/COMPLETE_FILE_INDEX.csv
```

The `category` column labels files as `input`, `output_log`, `script`, `processed_or_metadata`, `figure_or_report`, `structure_or_density`, or `other`.

## Warnings / Things To Re-sync

{warning_text}

## Important

This browser folder is copied from local pulled data. If the remote iservice data changed after the latest pull, run the commands in:

```text
00_OPEN_THIS_FIRST/RESYNC_FROM_ISERVICE_COMMANDS.md
```
"""
    (index_dir / "README_OPEN_THIS_FIRST.md").write_text(readme, encoding="utf-8")


def write_rsync_commands(out_dir: Path) -> None:
    index_dir = out_dir / "00_OPEN_THIS_FIRST"
    index_dir.mkdir(parents=True, exist_ok=True)
    dest = str(DEFAULT_LOCAL_PULL).replace("\\", "/")
    nano_dest = str(DEFAULT_NANO_PULL).replace("\\", "/")
    commands = [
        "# Run these in WSL/Git Bash/terminal with interactive 2FA. Each command is intentionally one line.",
        "",
        f"mkdir -p {dest}",
        f"rsync -avhP --exclude '*/tmp/***' iservice:/gpfs-work/dawson666/qe_cases/qe_runs/qe_bulk_b_convergence_20260506/ {dest}/qe_bulk_b_convergence_20260506/",
        f"rsync -avhP --exclude '*/tmp/***' iservice:/gpfs-work/dawson666/qe_cases/qe_runs/qe_vacancy_convergence_20260506/ {dest}/qe_vacancy_convergence_20260506/",
        f"rsync -avhP iservice:/gpfs-work/dawson666/dftpy_project/relax/dftpy45/results/dftpy_vacancy_convergence_primitive4_qe_a0_20260508/ {dest}/dftpy_vacancy_convergence_primitive4_qe_a0_20260508/",
        f"rsync -avhP iservice:/gpfs-work/dawson666/dftpy_project/relax/dftpy45/results/dftpy_vacancy_convergence_primitive4_20260508/ {dest}/dftpy_vacancy_convergence_primitive4_20260508/",
        f"rsync -avhP iservice:/gpfs-work/dawson666/dftpy_project/relax/dftpy45/results/dftpy_vacancy_size_primitive_qe_a0_spacing020_20260511/ {dest}/dftpy_vacancy_size_primitive_qe_a0_spacing020_20260511/",
        "",
        "# Optional older nanostructure actual-point archive, only if it exists remotely under your chosen path.",
        f"# rsync -avhP iservice:/gpfs-work/dawson666/qe_cases/qe_runs/vacancy_nanocrystal_relax/ {nano_dest}/vacancy_nanocrystal_relax/",
        "",
        "# After re-sync, rebuild the local browser:",
        "python scripts/build_professor_raw_data_browser.py --zip",
    ]
    (index_dir / "RESYNC_FROM_ISERVICE_COMMANDS.md").write_text("\n".join(commands) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a professor-facing local raw data browser.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--no-clean", dest="clean", action="store_false")
    parser.set_defaults(clean=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and args.clean:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, str]] = []
    warnings: list[str] = []

    for item in WORK_ITEMS:
        source = Path(item["source"]).expanduser().resolve()
        dst = out_dir / str(item["folder"])
        if source.exists():
            copytree(source, dst)
            write_work_item_readme(item, dst)
            copied.append(
                {
                    "folder": str(item["folder"]),
                    "title": str(item["title"]),
                    "source": str(source),
                }
            )
            print(f"[COPY] {source} -> {dst}")
        else:
            warnings.append(f"Missing local source for {item['title']}: {source}")
            print(f"[WARN] missing {source}")

    processed_dst = out_dir / "07_processed_outputs_from_repo"
    copytree(ROOT / "outputs", processed_dst)
    print(f"[COPY] {ROOT / 'outputs'} -> {processed_dst}")

    scripts_dst = out_dir / "08_repo_scripts_and_workflow_snapshot"
    scripts_dst.mkdir(parents=True, exist_ok=True)
    copytree(ROOT / "scripts", scripts_dst / "scripts")
    copytree(ROOT / "app", scripts_dst / "app")
    for rel in [
        "WORKFLOW.md",
        "REPRODUCIBILITY_INDEX.md",
        "REPORT_SUMMARY_20260519.md",
        "run_bulk_vacancy_supercell_one_ct56.sbatch",
        "run_dftpy_primitive_size_one_ct56.sbatch",
        "run_dftpy_vacancy_convergence_one_ct56.sbatch",
        "submit_bulk_vacancy_supercell_ct56.sh",
    ]:
        copyfile(ROOT / rel, scripts_dst / rel)
    print(f"[COPY] repo scripts/workflow -> {scripts_dst}")

    reports_dst = out_dir / "09_reports_pptx_pdf"
    report_candidates = [
        Path.home() / "OneDrive" / "Documents" / "qe_bulk_vacancy.pptx",
        Path.home()
        / "OneDrive"
        / "Documents"
        / "qe_bulk_vacancy_concise_english_report_20260511_comment_response_finalsync.pptx",
        Path.home() / "Desktop" / "NUS_upload" / "reproducibility_package_20260519" / "05_reports" / "qe_bulk_vacancy.pdf",
        Path.home()
        / "Desktop"
        / "NUS_upload"
        / "reproducibility_package_20260519"
        / "05_reports"
        / "qe_bulk_vacancy_concise_english_report_20260511_comment_response_finalsync.pdf",
    ]
    for report in report_candidates:
        copyfile(report, reports_dst / report.name)

    k6_relax = out_dir / "02_QE_vacancy_formation_convergence_FULL_RAW" / "kmesh_scan" / "k_06x06x06" / "vacancy_relax" / "relax.out"
    if k6_relax.exists() and not has_job_done(k6_relax):
        warnings.append(
            "QE vacancy k_06x06x06/vacancy_relax/relax.out exists locally but does not contain JOB DONE. Re-sync this case from iservice before presenting k=6 as raw-output verified."
        )

    write_rsync_commands(out_dir)
    write_index(out_dir, copied, warnings)

    if args.zip:
        zip_path = out_dir.with_suffix(".zip")
        if zip_path.exists():
            zip_path.unlink()
        archive = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
        print(f"[ZIP] {archive}")

    print("============================================================")
    print("Professor raw data browser completed")
    print("============================================================")
    print(out_dir)
    print(out_dir / "00_OPEN_THIS_FIRST" / "README_OPEN_THIS_FIRST.md")


if __name__ == "__main__":
    main()
