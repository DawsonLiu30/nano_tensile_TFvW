from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from collect_qe_vcrelax_vacancy import collect  # noqa: E402


PRIMARY_KMESH = "5x5x5"
PRIMARY_ECUT_EV = 800.0
HISTORICAL_REFERENCE_EV = 0.601167
ORIGINAL_REMOTE_SOURCE = (
    "/gpfs-work/dawson666/qe_cases/qe_runs/"
    "qe_vacancy_vcrelax_conv3x3x3_centered_20260528"
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a complete corrected QE single-vacancy reference dossier.")
    parser.add_argument("--source", required=True, help="Local raw QE 3x3x3 vc-relax directory")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--remote-source", default=ORIGINAL_REMOTE_SOURCE)
    parser.add_argument("--zip", action="store_true", dest="make_zip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    manifest_path = source / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    rows = collect(source)
    completed = [row for row in rows if row["pristine_done"] and row["vacancy_done"]]
    kmesh_rows = sorted(
        [row for row in completed if row["mode"] == "kmesh_scan"],
        key=lambda row: int(str(row["kmesh"]).split("x")[0]),
    )
    ecut_rows = sorted(
        [row for row in completed if row["mode"] == "ecut_scan"],
        key=lambda row: float(row["ecut_eV"]),
    )
    primary = next(
        row
        for row in kmesh_rows
        if row["kmesh"] == PRIMARY_KMESH and abs(float(row["ecut_eV"]) - PRIMARY_ECUT_EV) < 1.0
    )
    dense_rows = [row for row in kmesh_rows if int(str(row["kmesh"]).split("x")[0]) >= 3]
    dense_values = [float(row["Ef_vac_eV"]) for row in dense_rows]

    if outdir.exists():
        shutil.rmtree(outdir)
    summary_dir = outdir / "01_REFERENCE_SUMMARY"
    raw_dir = outdir / "02_COMPLETE_RAW_QE_3x3x3"
    provenance_dir = outdir / "03_PROVENANCE"
    summary_dir.mkdir(parents=True)
    provenance_dir.mkdir(parents=True)

    shutil.copytree(source, raw_dir)
    write_csv(summary_dir / "qe_vcrelax_all_completed.csv", completed)
    write_csv(summary_dir / "qe_vcrelax_kmesh_scan.csv", kmesh_rows)
    write_csv(summary_dir / "qe_vcrelax_ecut_scan.csv", ecut_rows)
    write_csv(
        summary_dir / "qe_reference_selection.csv",
        [
            {
                "reference_role": "primary_corrected_best_completed_dense_k",
                "cell": "conventional cubic fcc 3x3x3",
                "N_pristine": primary["N_pristine"],
                "N_vacancy": primary["N_vacancy"],
                "calculation": "vc-relax for pristine and vacancy",
                "xc": "PBE",
                "pseudo": "Al_PAW_PBE.UPF",
                "ecut_eV": primary["ecut_eV"],
                "kmesh": primary["kmesh"],
                "Ef_vac_eV": primary["Ef_vac_eV"],
                "pristine_final_atomic_fmax_eV_A": primary["pristine_final_atomic_fmax_eV_A"],
                "vacancy_final_atomic_fmax_eV_A": primary["vacancy_final_atomic_fmax_eV_A"],
                "dense_k_min_Ef_eV": min(dense_values),
                "dense_k_max_Ef_eV": max(dense_values),
                "raw_case_dir": primary["path"],
            }
        ],
    )

    (provenance_dir / "ORIGINAL_SOURCE_PATHS.txt").write_text(
        f"Original remote source:\n  {args.remote_source}\n\n"
        f"Local source used to build this package:\n  {source}\n",
        encoding="utf-8",
    )
    shutil.copy2(SCRIPT_DIR / "collect_qe_vcrelax_vacancy.py", provenance_dir)

    details = f"""# Corrected QE single-vacancy reference

## Primary value

- Cell: conventional cubic fcc `3x3x3`
- Atoms: `{manifest['pristine_n_atoms']} -> {manifest['vacancy_n_atoms']}`
- Vacancy concentration: `{manifest['vacancy_concentration_percent']:.6f}%`
- Vacancy site: centered
- Calculation: `vc-relax` for pristine and vacancy
- XC/pseudopotential: PBE / `Al_PAW_PBE.UPF`
- Plane-wave cutoff: `{float(primary['ecut_eV']):.0f} eV`
- k mesh: `{primary['kmesh']}`
- Formation-energy formula: `E_vac(107) - (107/108) E_pristine(108)`
- Formation energy: `{float(primary['Ef_vac_eV']):.9f} eV`
- Final pristine atomic fmax: `{float(primary['pristine_final_atomic_fmax_eV_A']):.8f} eV/A`
- Final vacancy atomic fmax: `{float(primary['vacancy_final_atomic_fmax_eV_A']):.8f} eV/A`

## Dense-k sensitivity

The completed `3x3x3` to `5x5x5` values span
`{min(dense_values):.9f}-{max(dense_values):.9f} eV`. The `5x5x5` point is the
best completed dense-k value, but the spread must remain visible in reporting.

## Historical value kept separate

`{HISTORICAL_REFERENCE_EV:.6f} eV` came from the older conventional `2x2x4`,
`64 -> 63` workflow. It is not the corrected `3x3x3` reference.

## Force columns

`Total force` in QE output is a system-wide aggregate. The `fmax` columns in
this package are independently parsed as the maximum norm among the final
per-atom force vectors.
"""
    (summary_dir / "README_REFERENCE.md").write_text(details, encoding="utf-8")

    inventory_rows: list[dict[str, object]] = []
    for path in sorted(outdir.rglob("*")):
        if path.is_file():
            inventory_rows.append(
                {
                    "relative_path": str(path.relative_to(outdir)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_csv(provenance_dir / "FILE_INVENTORY_SHA256.csv", inventory_rows)

    if args.make_zip:
        zip_path = shutil.make_archive(str(outdir), "zip", root_dir=outdir.parent, base_dir=outdir.name)
        print(f"ZIP: {zip_path}")

    print("============================================================")
    print("QE single-vacancy reference package built")
    print("============================================================")
    print(f"Output: {outdir}")
    print(f"Completed cases: {len(completed)}/{len(rows)}")
    print(f"Primary Ef: {float(primary['Ef_vac_eV']):.9f} eV")
    print(f"Dense-k range: {min(dense_values):.9f}-{max(dense_values):.9f} eV")


if __name__ == "__main__":
    main()
