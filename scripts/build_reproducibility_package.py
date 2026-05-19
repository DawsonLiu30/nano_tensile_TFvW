from __future__ import annotations

import argparse
import csv
import shutil
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path.home() / "Desktop" / "NUS_upload" / f"reproducibility_package_{date.today():%Y%m%d}"
DEFAULT_RAW_ROOT = Path.home() / "Desktop" / "latest_professor_pull_20260511"
DEFAULT_NANO_RAW = Path.home() / "Desktop" / "vacancy_qe_ofdft_results_2026-04-25"
DEFAULT_REPORTS = [
    Path.home() / "OneDrive" / "Documents" / "qe_bulk_vacancy.pptx",
    Path.home()
    / "OneDrive"
    / "Documents"
    / "qe_bulk_vacancy_concise_english_report_20260511_comment_response_finalsync.pptx",
]


IGNORE_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".ipynb_checkpoints",
    "tmp",
}


def ignore_runtime(dir_path: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name in IGNORE_DIR_NAMES:
            ignored.add(name)
    return ignored


def copy_path(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"[WARN] missing: {src}")
        return
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=ignore_runtime)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f"[COPY] {src} -> {dst}")


def write_manifest(package_dir: Path) -> None:
    rows = []
    for path in sorted(package_dir.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": str(path.relative_to(package_dir)).replace("\\", "/"),
                    "size_bytes": path.stat().st_size,
                }
            )
    manifest_csv = package_dir / "file_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes"])
        writer.writeheader()
        writer.writerows(rows)

    manifest_txt = package_dir / "PACKAGE_MANIFEST.txt"
    total_mb = sum(row["size_bytes"] for row in rows) / (1024 * 1024)
    manifest_txt.write_text(
        "\n".join(
            [
                "Al nanostructure reproducibility package",
                f"Date: {date.today():%Y-%m-%d}",
                f"Source repository: {ROOT}",
                f"Files: {len(rows)}",
                f"Total size: {total_mb:.2f} MB",
                "",
                "Open 00_REPRODUCIBILITY_INDEX.md first.",
                "Open 01_REPORT_SUMMARY_20260519.md for the concise result summary.",
                "Raw code input/output files are under 04_raw_data when included.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_package(args: argparse.Namespace) -> Path:
    package_dir = Path(args.out_dir).expanduser().resolve()
    if package_dir.exists() and args.clean:
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    copy_path(ROOT / "REPRODUCIBILITY_INDEX.md", package_dir / "00_REPRODUCIBILITY_INDEX.md")
    copy_path(ROOT / "REPORT_SUMMARY_20260519.md", package_dir / "01_REPORT_SUMMARY_20260519.md")

    code_root = package_dir / "02_code_snapshot"
    for rel in [
        ".gitignore",
        "WORKFLOW.md",
        "al.gga.recpot",
        "run_bulk_vacancy_supercell_one_ct56.sbatch",
        "run_dftpy_primitive_size_one_ct56.sbatch",
        "run_dftpy_vacancy_convergence_one_ct56.sbatch",
        "submit_bulk_vacancy_supercell_ct56.sh",
    ]:
        copy_path(ROOT / rel, code_root / rel)
    copy_path(ROOT / "app", code_root / "app")
    copy_path(ROOT / "scripts", code_root / "scripts")

    copy_path(ROOT / "outputs", package_dir / "03_processed_outputs")

    if not args.skip_raw:
        raw_root = Path(args.raw_root).expanduser().resolve()
        nano_raw = Path(args.nanostructure_raw).expanduser().resolve()
        copy_path(raw_root, package_dir / "04_raw_data" / raw_root.name)
        copy_path(nano_raw, package_dir / "04_raw_data" / nano_raw.name)

    reports_dir = package_dir / "05_reports"
    for report in args.report:
        copy_path(Path(report).expanduser().resolve(), reports_dir / Path(report).name)

    write_manifest(package_dir)

    if args.zip:
        archive_base = package_dir.with_suffix("")
        zip_path = shutil.make_archive(str(archive_base), "zip", root_dir=package_dir)
        print(f"[ZIP] {zip_path}")

    return package_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local reproducibility package for NUS/NAS upload."
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT), help="Output package directory.")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT), help="Latest pulled raw data root.")
    parser.add_argument(
        "--nanostructure-raw",
        default=str(DEFAULT_NANO_RAW),
        help="Pulled vacancy nanostructure QE/OFDFT comparison archive.",
    )
    parser.add_argument(
        "--report",
        action="append",
        default=[str(path) for path in DEFAULT_REPORTS],
        help="Report PPT/PDF file to copy. May be passed multiple times.",
    )
    parser.add_argument("--skip-raw", action="store_true", help="Only package code and processed outputs.")
    parser.add_argument("--no-clean", dest="clean", action="store_false", help="Do not remove existing package dir first.")
    parser.add_argument("--zip", action="store_true", help="Also create a .zip archive next to the package dir.")
    parser.set_defaults(clean=True)
    return parser.parse_args()


def main() -> None:
    package_dir = build_package(parse_args())
    print("============================================================")
    print("Reproducibility package completed")
    print("============================================================")
    print(f"Package directory: {package_dir}")
    print(f"Start with       : {package_dir / '00_REPRODUCIBILITY_INDEX.md'}")


if __name__ == "__main__":
    main()
