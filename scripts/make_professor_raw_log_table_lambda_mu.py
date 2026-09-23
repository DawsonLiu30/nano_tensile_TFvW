#!/usr/bin/env python3
"""Create professor-facing lambda/mu raw-output tables.

This intentionally does *not* compute vacancy formation energy.  It extracts
the simple values requested from the final DFTpy output summaries:

  - pristine total energy from pristine_dftpy.out
  - pristine KEDF / kinetic energy from pristine_dftpy.out
  - pristine relaxed lattice constant from the final pristine VASP structure

The flat table keeps vacancy raw output fields only for audit/provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


MU_VALUES = [round(0.1 * i, 1) for i in range(1, 11)]
LAMBDA_VALUES = [round(0.1 * i, 1) for i in range(1, 11)]


def fmt_param(value: float) -> str:
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return f"{value:.1f}".rstrip("0").rstrip(".")


def parse_settings(root: Path) -> list[dict[str, Any]]:
    settings = root / "01_settings" / "settings_lambda_mu_10x10.txt"
    rows: list[dict[str, Any]] = []
    if settings.exists():
        for line in settings.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            rows.append({"setting": parts[0], "lambda": float(parts[1]), "mu": float(parts[2])})
        return rows

    for d in sorted((root / "03_runs").glob("tfvw_lam*_mu*")):
        m = re.search(r"lam([0-9p]+)_mu([0-9p]+)", d.name)
        if not m:
            continue
        lam = float(m.group(1).replace("p", "."))
        mu = float(m.group(2).replace("p", "."))
        rows.append({"setting": d.name, "lambda": lam, "mu": mu})
    return rows


def parse_dftpy_summary(path: Path) -> dict[str, float]:
    """Parse the compact *_dftpy.out summary written by our runner."""
    data: dict[str, float] = {}
    if not path.exists():
        return data

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        m = re.match(r"total energy \(eV\)\s*:\s*([-+0-9.eE]+)", line, flags=re.I)
        if m:
            data["total_energy_eV"] = float(m.group(1))
            continue

        m = re.match(r"(KEDF|KEDF-TF|KEDF-VW|TOTAL|XC|HARTREE|PSEUDO|II)\s*:\s*([-+0-9.eE]+)", line)
        if m:
            key = m.group(1).lower().replace("-", "_")
            if key == "total":
                # Keep this separate from the grep-style "total energy (eV)"
                # line above.  The two can differ slightly because the compact
                # summary records both the calculator total and the decomposed
                # component total.
                data["component_total_energy_eV"] = float(m.group(2))
            else:
                data[f"{key}_energy_eV"] = float(m.group(2))
            continue

    return data


def read_vasp_lattice_constant(path: Path) -> tuple[float | None, str]:
    if not path.exists():
        return None, "missing"

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        scale = float(lines[1].split()[0])
        vectors = []
        for i in range(2, 5):
            vec = [float(x) * scale for x in lines[i].split()[:3]]
            vectors.append(vec)
        lengths = [math.sqrt(sum(x * x for x in vec)) for vec in vectors]
        # These cases are conventional 3x3x3 fcc supercells.  The requested
        # lattice constant is the relaxed supercell length divided by 3.
        return sum(lengths) / len(lengths) / 3.0, f"mean(|a|,|b|,|c|)/3 from {path.name}"
    except Exception as exc:  # pragma: no cover - defensive for malformed VASP
        return None, f"parse_error: {exc}"


def choose_pristine_structure(case_dir: Path) -> Path:
    for name in ("pristine_vc_relaxed.vasp", "pristine_relaxed.vasp"):
        p = case_dir / name
        if p.exists():
            return p
    return case_dir / "pristine_vc_relaxed.vasp"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def read_failure_reason(case_dir: Path) -> str:
    for name in ("LOCAL_RERUN_FAILED.txt", "RESCUE_FAILED.txt"):
        p = case_dir / name
        if p.exists():
            text = p.read_text(encoding="utf-8", errors="replace").strip()
            return text or name

    status = read_json(case_dir / "local_case_status.json")
    if status.get("status") == "FAILED":
        return str(status.get("error") or "FAILED")

    return ""


def collect_rows(root: Path) -> list[dict[str, Any]]:
    runs = root / "03_runs"
    rows = []

    for item in parse_settings(root):
        setting = item["setting"]
        lam = item["lambda"]
        mu = item["mu"]
        case_dir = runs / setting

        pristine_out = case_dir / "pristine_dftpy.out"
        vacancy_out = case_dir / "vacancy_dftpy.out"
        pristine_ini = case_dir / "dftpy_pristine_input.ini"
        vacancy_ini = case_dir / "dftpy_vacancy_input.ini"
        result_json = case_dir / "result.json"
        result = read_json(result_json)
        failure_reason = read_failure_reason(case_dir)

        p_terms = parse_dftpy_summary(pristine_out)
        v_terms = parse_dftpy_summary(vacancy_out)

        lattice_path = choose_pristine_structure(case_dir)
        lattice_a, lattice_source = read_vasp_lattice_constant(lattice_path)

        status_bits = []
        if failure_reason:
            status_bits.append("FAILED")
        if not pristine_out.exists():
            status_bits.append("missing_pristine_output")
        if "total_energy_eV" not in p_terms:
            status_bits.append("missing_pristine_total")
        if "kedf_energy_eV" not in p_terms:
            status_bits.append("missing_pristine_kedf")
        if lattice_a is None:
            status_bits.append("missing_pristine_lattice")
        status = "OK" if not status_bits else ";".join(status_bits)

        source_dir = result.get("source_dir") or str(case_dir)
        row = {
            "setting": setting,
            "lambda": lam,
            "mu": mu,
            "raw_table_status": status,
            "failure_reason": failure_reason,
            "pristine_total_energy_eV": p_terms.get("total_energy_eV"),
            "pristine_kedf_energy_eV": p_terms.get("kedf_energy_eV"),
            "pristine_kedf_tf_energy_eV": p_terms.get("kedf_tf_energy_eV"),
            "pristine_kedf_vw_energy_eV": p_terms.get("kedf_vw_energy_eV"),
            "pristine_component_total_energy_eV": p_terms.get("component_total_energy_eV"),
            "pristine_lattice_constant_A": lattice_a,
            "lattice_source": lattice_source,
            "vacancy_total_energy_eV": v_terms.get("total_energy_eV"),
            "vacancy_kedf_energy_eV": v_terms.get("kedf_energy_eV"),
            "vacancy_kedf_tf_energy_eV": v_terms.get("kedf_tf_energy_eV"),
            "vacancy_kedf_vw_energy_eV": v_terms.get("kedf_vw_energy_eV"),
            "vacancy_component_total_energy_eV": v_terms.get("component_total_energy_eV"),
            "source_dir": source_dir,
            "input_file_pristine": str(pristine_ini),
            "input_file_vacancy": str(vacancy_ini),
            "output_file_pristine": str(pristine_out),
            "output_file_vacancy": str(vacancy_out),
            "relaxed_structure_pristine": str(lattice_path),
            "result_json": str(result_json),
            "result_json_exists": result_json.exists(),
            "calculation_details": "DFTpy LDA TFvW lambda-mu scan; conventional fcc 3x3x3 Al108 pristine; full atom+cell relaxation/vc-relax-equivalent; raw values parsed from final *_dftpy.out summaries; lattice constant calculated from relaxed pristine VASP cell.",
        }
        rows.append(row)

    return rows


def matrix(rows: list[dict[str, Any]], field: str) -> list[list[Any]]:
    by_pair = {(round(r["lambda"], 10), round(r["mu"], 10)): r for r in rows}
    out: list[list[Any]] = [["lambda/mu", *[fmt_param(mu) for mu in MU_VALUES]]]
    for lam in LAMBDA_VALUES:
        line: list[Any] = [fmt_param(lam)]
        for mu in MU_VALUES:
            r = by_pair.get((round(lam, 10), round(mu, 10)))
            if not r:
                line.append("MISSING")
            else:
                value = r.get(field)
                if value is not None:
                    line.append(value)
                elif str(r.get("raw_table_status", "")).startswith("FAILED"):
                    line.append("FAILED")
                else:
                    line.append("MISSING")
        out.append(line)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_matrix_csv(path: Path, matrices: list[tuple[str, list[list[Any]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        first = True
        for title, mat in matrices:
            if not first:
                writer.writerow([])
            first = False
            writer.writerow([title])
            writer.writerows(mat)


def col_letter(idx: int) -> str:
    result = ""
    while idx:
        idx, rem = divmod(idx - 1, 26)
        result = chr(65 + rem) + result
    return result


def xcell(row: int, col: int, value: Any) -> str:
    ref = f"{col_letter(col)}{row}"
    if value is None:
        return f'<c r="{ref}"/>'
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return f'<c r="{ref}" t="n"><v>{value:.12g}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def sheet_xml(rows: list[list[Any]]) -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        "<sheetData>",
    ]
    for r_idx, row in enumerate(rows, 1):
        parts.append(f'<row r="{r_idx}">')
        for c_idx, value in enumerate(row, 1):
            parts.append(xcell(r_idx, c_idx, value))
        parts.append("</row>")
    parts.extend(["</sheetData>", "</worksheet>"])
    return "".join(parts)


def write_xlsx(path: Path, sheets: list[tuple[str, list[list[Any]]]]) -> None:
    """Write a simple standards-compliant xlsx using only the stdlib."""
    path.parent.mkdir(parents=True, exist_ok=True)

    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
    ]
    for i in range(1, len(sheets) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    workbook_sheets = []
    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for i, (name, _rows) in enumerate(sheets, 1):
        safe_name = escape(name[:31])
        workbook_sheets.append(f'<sheet name="{safe_name}" sheetId="{i}" r:id="rId{i}"/>')
        workbook_rels.append(
            f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
        )
    workbook_rels.append("</Relationships>")

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(workbook_sheets)}</sheets></workbook>'
    )

    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "".join(content_types))
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        for i, (_name, rows) in enumerate(sheets, 1):
            zf.writestr(f"xl/worksheets/sheet{i}.xml", sheet_xml(rows))


def professor_raw_rows(rows: list[dict[str, Any]]) -> list[list[Any]]:
    total = matrix(rows, "pristine_total_energy_eV")
    kedf = matrix(rows, "pristine_kedf_energy_eV")
    lattice = matrix(rows, "pristine_lattice_constant_A")
    status = matrix(rows, "raw_table_status")

    # One wide sheet matching the professor's requested three-block layout.
    out: list[list[Any]] = []
    title_row = ["Total energy from pristine_dftpy.out (eV)"] + [""] * 11
    title_row += ["KEDF / kinetic energy from pristine_dftpy.out (eV)"] + [""] * 11
    title_row += ["Lattice constant from relaxed pristine VASP (A)"] + [""] * 10
    out.append(title_row)
    for r in range(len(total)):
        out.append(total[r] + [""] + kedf[r] + [""] + lattice[r])
    out.append([])
    out.append(["Status matrix for raw pristine table"] + [""] * 10)
    out.extend(status)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", required=True, help="coarse_10x10_vacancy_formation root")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    root = Path(args.rootdir).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else root / "10_professor_raw_log_table"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(root)
    fieldnames = [
        "setting",
        "lambda",
        "mu",
        "raw_table_status",
        "failure_reason",
        "pristine_total_energy_eV",
        "pristine_kedf_energy_eV",
        "pristine_kedf_tf_energy_eV",
        "pristine_kedf_vw_energy_eV",
        "pristine_component_total_energy_eV",
        "pristine_lattice_constant_A",
        "lattice_source",
        "vacancy_total_energy_eV",
        "vacancy_kedf_energy_eV",
        "vacancy_kedf_tf_energy_eV",
        "vacancy_kedf_vw_energy_eV",
        "vacancy_component_total_energy_eV",
        "source_dir",
        "input_file_pristine",
        "input_file_vacancy",
        "output_file_pristine",
        "output_file_vacancy",
        "relaxed_structure_pristine",
        "result_json",
        "result_json_exists",
        "calculation_details",
    ]

    write_csv(outdir / "professor_raw_log_flat_results.csv", rows, fieldnames)
    write_matrix_csv(
        outdir / "professor_raw_log_three_block_table.csv",
        [
            ("Total energy from pristine_dftpy.out (eV)", matrix(rows, "pristine_total_energy_eV")),
            ("KEDF / kinetic energy from pristine_dftpy.out (eV)", matrix(rows, "pristine_kedf_energy_eV")),
            ("Lattice constant from relaxed pristine VASP (A)", matrix(rows, "pristine_lattice_constant_A")),
            ("Status matrix", matrix(rows, "raw_table_status")),
        ],
    )
    with (outdir / "professor_raw_log_three_block_table_wide.csv").open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(professor_raw_rows(rows))

    notes = [
        ["Item", "Definition"],
        ["Total", "Final pristine total energy parsed directly from pristine_dftpy.out; no formation-energy subtraction."],
        ["KEDF", "Final pristine KEDF / kinetic-energy component parsed directly from pristine_dftpy.out."],
        ["Lattice constant", "Calculated from relaxed pristine VASP cell as mean(|a|, |b|, |c|) / 3 for the 3x3x3 conventional cell."],
        ["Flat_Raw_Outputs", "Audit table containing source_dir, input files, output files, relaxed structure path, and vacancy raw outputs for provenance."],
        ["Important", "This workbook intentionally does not compute vacancy formation energy. Formation energy belongs to the next analysis step."],
    ]

    flat_sheet = [fieldnames] + [[row.get(k) for k in fieldnames] for row in rows]
    write_xlsx(
        outdir / "professor_raw_log_total_kedf_lattice_table.xlsx",
        [
            ("Professor_Raw_Table", professor_raw_rows(rows)),
            ("Flat_Raw_Outputs", flat_sheet),
            ("Notes", notes),
        ],
    )

    readme = """# Professor raw-log lambda/mu table

This folder is intentionally separate from the vacancy-formation-energy analysis.

## Purpose

Extract the three first-pass quantities requested directly from the raw DFTpy
output files for the single-vacancy lambda/mu scan:

1. Total energy
2. KEDF / kinetic energy
3. Lattice constant

## Definitions

- Total energy: final pristine total energy parsed from `pristine_dftpy.out`.
- KEDF / kinetic energy: final pristine `KEDF` component parsed from `pristine_dftpy.out`.
- Lattice constant: calculated from the relaxed pristine VASP output as
  `mean(|a|, |b|, |c|) / 3` for the conventional `3x3x3` fcc supercell.
  The initial `pristine_raw.vasp` is not used as a fallback for failed
  calculations, because this table is intended to report relaxed values.

No vacancy formation energy is computed in the main table.

## Files

- `professor_raw_log_total_kedf_lattice_table.xlsx`
  - `Professor_Raw_Table`: professor-facing three-block matrix.
  - `Flat_Raw_Outputs`: source paths and raw values for provenance.
  - `Notes`: definitions.
- `professor_raw_log_three_block_table.csv`
  - CSV version of the three requested matrices.
- `professor_raw_log_three_block_table_wide.csv`
  - Side-by-side CSV version matching the Excel layout.
- `professor_raw_log_flat_results.csv`
  - Flat audit table with source folders, input files, output files, relaxed
    structures, and vacancy raw outputs for traceability.
- `RAW_LOG_TABLE_SUMMARY.json`
  - Counts and generated file paths.

## Missing values

The main three-block table contains raw pristine values whenever the pristine
output file exists, even if the later vacancy calculation failed.  The status
matrix and flat audit table should therefore be checked together with the raw
value table.

Cells marked `FAILED` mean the local rerun guard classified the case as
pathological or failed and no raw pristine value was available for that field.
Cells marked `MISSING` mean the corresponding `pristine_dftpy.out` was not
present and no explicit failure marker was found.  Values are not back-filled
from `result.json`, because this table is meant to be raw-output based.
"""
    (outdir / "README_RAW_LOG_TABLE.md").write_text(readme, encoding="utf-8")

    ok_count = sum(1 for row in rows if row["raw_table_status"] == "OK")
    failed_count = sum(1 for row in rows if str(row["raw_table_status"]).startswith("FAILED"))
    pristine_raw_available = sum(
        1
        for row in rows
        if row.get("pristine_total_energy_eV") is not None
        and row.get("pristine_kedf_energy_eV") is not None
        and row.get("pristine_lattice_constant_A") is not None
    )
    summary = {
        "rootdir": str(root),
        "outdir": str(outdir),
        "points_total": len(rows),
        "complete_case_status_ok": ok_count,
        "case_status_failed": failed_count,
        "pristine_raw_three_values_available": pristine_raw_available,
        "pristine_raw_three_values_missing": len(rows) - pristine_raw_available,
        "case_status_not_ok": len(rows) - ok_count,
        "main_xlsx": str(outdir / "professor_raw_log_total_kedf_lattice_table.xlsx"),
        "main_csv": str(outdir / "professor_raw_log_three_block_table.csv"),
        "wide_csv": str(outdir / "professor_raw_log_three_block_table_wide.csv"),
        "flat_csv": str(outdir / "professor_raw_log_flat_results.csv"),
        "readme": str(outdir / "README_RAW_LOG_TABLE.md"),
    }
    (outdir / "RAW_LOG_TABLE_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
