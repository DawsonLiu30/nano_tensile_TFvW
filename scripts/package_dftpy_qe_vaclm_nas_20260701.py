from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path


DESKTOP = Path(r"C:\Users\dawso\Desktop")
DFTPY_SRC = DESKTOP / "DFTPY_VACLM_ISERVICE_NAS_20260630"
QE_SRC = DESKTOP / "QE_VACLM_REFERENCE_20260630_LOCAL_RESULTS"
OUT = DESKTOP / "DFTPY_QE_VACLM_NAS_20260701"

QE_RY_TO_EV = 13.605693122994
QE_REFERENCE_EF_EV = 0.6369464632544929


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source folder: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def parse_qe_output(path: Path) -> dict[str, float | int | None]:
    text = path.read_text(errors="ignore")
    totals = [float(x) for x in re.findall(r"!\s+total energy\s+=\s+([-0-9.]+)\s+Ry", text)]
    enthalpies = [float(x) for x in re.findall(r"Final enthalpy\s+=\s+([-0-9.]+)\s+Ry", text)]
    forces = [float(x) for x in re.findall(r"Total force\s+=\s+([-0-9.]+)", text)]
    return {
        "job_done_count": text.count("JOB DONE"),
        "final_total_energy_Ry": totals[-1] if totals else None,
        "final_total_energy_eV": totals[-1] * QE_RY_TO_EV if totals else None,
        "final_enthalpy_Ry": enthalpies[-1] if enthalpies else None,
        "final_enthalpy_eV": enthalpies[-1] * QE_RY_TO_EV if enthalpies else None,
        "last_total_force_Ry_bohr": forces[-1] if forces else None,
    }


def read_dftpy_candidates() -> list[dict[str, str]]:
    table = DFTPY_SRC / "02_SIMPLE_MAPS" / "fresh_iservice_resultjson_flat_results.csv"
    if not table.exists():
        raise FileNotFoundError(f"Missing DFTpy summary table: {table}")
    with table.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    dftpy_dst = OUT / "01_DFTPY_ISERVICE_VACLM"
    qe_dst = OUT / "02_QE_REFERENCE_VCRELAX"
    summary_dir = OUT / "03_COMPARISON_SUMMARY"
    slides_dir = OUT / "04_EVALUATION_SLIDES"
    summary_dir.mkdir(parents=True, exist_ok=True)
    slides_dir.mkdir(parents=True, exist_ok=True)

    copy_tree(DFTPY_SRC, dftpy_dst)
    copy_tree(QE_SRC, qe_dst)

    pristine = parse_qe_output(qe_dst / "pristine_vcrelax" / "pw.out")
    vacancy = parse_qe_output(qe_dst / "vacancy_vcrelax" / "pw.out")
    pristine_total = pristine["final_total_energy_Ry"]
    vacancy_total = vacancy["final_total_energy_Ry"]
    ef_ry = None
    ef_ev = None
    if pristine_total is not None and vacancy_total is not None:
        ef_ry = vacancy_total - (107.0 / 108.0) * pristine_total
        ef_ev = ef_ry * QE_RY_TO_EV

    qe_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": "QE 7.5 vc-relax reference",
        "cell": "conventional fcc Al 3x3x3",
        "N_pristine": 108,
        "N_vacancy": 107,
        "pseudo": "Al_PAW_PBE.UPF",
        "formula": "E_f^vac = E_vac(Al107) - (107/108) E_pristine(Al108)",
        "pristine": pristine,
        "vacancy": vacancy,
        "vacancy_formation_energy_Ry": ef_ry,
        "vacancy_formation_energy_eV": ef_ev,
    }
    (summary_dir / "qe_reference_summary.json").write_text(
        json.dumps(qe_summary, indent=2), encoding="utf-8"
    )

    write_csv(
        summary_dir / "qe_reference_summary.csv",
        [
            "case",
            "job_done_count",
            "final_total_energy_Ry",
            "final_total_energy_eV",
            "final_enthalpy_Ry",
            "final_enthalpy_eV",
            "last_total_force_Ry_bohr",
        ],
        [
            {"case": "pristine_vcrelax", **pristine},
            {"case": "vacancy_vcrelax", **vacancy},
            {
                "case": "vacancy_formation_energy",
                "final_total_energy_Ry": ef_ry,
                "final_total_energy_eV": ef_ev,
            },
        ],
    )

    dftpy_rows = read_dftpy_candidates()
    candidates = []
    for row in dftpy_rows:
        try:
            ef = float(row["vacancy_formation_energy_eV"])
        except Exception:
            continue
        status = row.get("status", "")
        qualified = str(row.get("qualified", "")).lower() in {"true", "1", "yes"}
        candidates.append(
            {
                "setting": row.get("setting", ""),
                "lambda": row.get("lambda", ""),
                "mu": row.get("mu", ""),
                "dftpy_Ef_vac_eV": ef,
                "qe_reference_Ef_vac_eV": ef_ev,
                "delta_vs_qe_eV": ef - float(ef_ev) if ef_ev is not None else "",
                "abs_delta_vs_qe_eV": abs(ef - float(ef_ev)) if ef_ev is not None else "",
                "status": status,
                "qualified": row.get("qualified", ""),
                "lattice_constant_A": row.get("lattice_constant_A", ""),
                "pristine_kedf_energy_eV": row.get("pristine_kedf_energy_eV", ""),
                "pristine_final_fmax_eV_A": row.get("pristine_final_fmax_eV_A", ""),
                "vacancy_final_fmax_eV_A": row.get("vacancy_final_fmax_eV_A", ""),
                "source_dir_iservice": row.get("source_dir_iservice", ""),
            }
        )

    candidates.sort(
        key=lambda r: (
            0 if str(r["qualified"]).lower() in {"true", "1", "yes"} else 1,
            float(r["abs_delta_vs_qe_eV"]) if r["abs_delta_vs_qe_eV"] != "" else 1e99,
        )
    )
    fields = [
        "setting",
        "lambda",
        "mu",
        "dftpy_Ef_vac_eV",
        "qe_reference_Ef_vac_eV",
        "delta_vs_qe_eV",
        "abs_delta_vs_qe_eV",
        "status",
        "qualified",
        "lattice_constant_A",
        "pristine_kedf_energy_eV",
        "pristine_final_fmax_eV_A",
        "vacancy_final_fmax_eV_A",
        "source_dir_iservice",
    ]
    write_csv(summary_dir / "dftpy_nearest_to_qe_reference.csv", fields, candidates[:20])
    write_csv(summary_dir / "dftpy_qe_all_points_comparison.csv", fields, candidates)

    best = candidates[0] if candidates else {}
    readme = f"""# DFTpy + QE VACLM NAS package

Generated: {datetime.now().isoformat(timespec="seconds")}

## Purpose

This folder combines the iService DFTpy TFvW lambda-mu vacancy calibration package
with the local QE 7.5 `vc-relax` reference calculation requested as the KSDFT/DFT
baseline.

## Folder map

- `01_DFTPY_ISERVICE_VACLM/`: iService DFTpy lambda-mu scan package, including raw cases, DFTpy inputs, outputs, trajectories, summary tables, scripts, and earlier evaluation slides.
- `02_QE_REFERENCE_VCRELAX/`: QE pristine and single-vacancy `vc-relax` inputs/outputs, pseudo, structures, and run scripts.
- `03_COMPARISON_SUMMARY/`: parsed QE reference and DFTpy-vs-QE comparison tables.
- `04_EVALUATION_SLIDES/`: updated concise slides with QE reference added.

## QE reference

- Formula: `E_f^vac = E_vac(Al107) - (107/108) E_pristine(Al108)`
- Pristine final total energy: `{pristine.get("final_total_energy_Ry")}` Ry
- Vacancy final total energy: `{vacancy.get("final_total_energy_Ry")}` Ry
- QE vacancy formation energy: `{ef_ev:.6f}` eV

## Nearest qualified DFTpy coarse-grid point

- Setting: `{best.get("setting", "")}`
- lambda: `{best.get("lambda", "")}`
- mu: `{best.get("mu", "")}`
- DFTpy vacancy formation energy: `{float(best.get("dftpy_Ef_vac_eV", 0.0)):.6f}` eV
- Difference from QE: `{float(best.get("delta_vs_qe_eV", 0.0)):+.6f}` eV

## Notes

The DFTpy scan is a coarse 0.1-grid lambda-mu scan. The QE value is used here
as the reference for selecting or refining lambda-mu values before moving to
divacancy or nanostructure calculations.
"""
    (OUT / "README_PACKAGE.md").write_text(readme, encoding="utf-8")

    print(json.dumps({
        "out": str(OUT),
        "dftpy_folder": str(dftpy_dst),
        "qe_folder": str(qe_dst),
        "summary_folder": str(summary_dir),
        "qe_Ef_eV": ef_ev,
        "best_dftpy_setting": best.get("setting", ""),
        "best_dftpy_Ef_eV": best.get("dftpy_Ef_vac_eV", ""),
        "best_delta_eV": best.get("delta_vs_qe_eV", ""),
    }, indent=2))


if __name__ == "__main__":
    main()
