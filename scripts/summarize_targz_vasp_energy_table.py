#!/usr/bin/env python3
"""Summarize VASP files inside a DFTpy delivery tar.gz.

The table is intended for advisor review: file name, total energy, atom count,
and volume for every VASP file in the archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

from ase.io import read


BFGS_RE = re.compile(
    r"^\s*BFGS:\s+(?P<step>\d+)\s+\S+\s+(?P<energy>[-+0-9.]+)\s+(?P<fmax>[-+0-9.]+)"
)
TOTAL_E_RE = re.compile(r"total energy \(eV\)\s*:\s*(?P<energy>[-+0-9.eE]+)")


def safe_extract_tar(tar_path: Path, dest: Path) -> None:
    """Extract tar_path into dest while rejecting path traversal members."""
    dest_resolved = dest.resolve()
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(dest_resolved)):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tf.extractall(dest)


def parse_spacing(path: Path) -> float | None:
    for part in path.parts:
        m = re.fullmatch(r"spacing_(\d+)p(\d+)A", part)
        if m:
            return float(f"{m.group(1)}.{m.group(2)}")
    return None


def read_total_energy(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    m = TOTAL_E_RE.search(text)
    return float(m.group("energy")) if m else None


def read_bfgs_steps(path: Path) -> list[dict[str, float | int]]:
    if not path.exists():
        return []
    steps: list[dict[str, float | int]] = []
    for line in path.read_text(errors="ignore").splitlines():
        m = BFGS_RE.match(line)
        if not m:
            continue
        steps.append(
            {
                "step": int(m.group("step")),
                "energy": float(m.group("energy")),
                "fmax": float(m.group("fmax")),
            }
        )
    return steps


def file_sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def classify_vasp(path: Path) -> tuple[str, str]:
    name = path.name
    if name.startswith("pristine"):
        system = "pristine"
    elif name.startswith("vacancy"):
        system = "vacancy"
    else:
        system = "unknown"

    if name.endswith("_raw.vasp") or name.endswith("_start.vasp"):
        structure_state = "initial"
    elif "vc_relaxed" in name:
        structure_state = "final_vc_relaxed"
    elif "relaxed" in name:
        structure_state = "final_relaxed_alias"
    else:
        structure_state = "unknown"
    return system, structure_state


def energy_for_vasp(path: Path) -> tuple[float | None, str, str, float | None, int | None]:
    """Return energy, source, note, fmax, bfgs step for a VASP file."""
    system, structure_state = classify_vasp(path)
    case_dir = path.parent
    out_path = case_dir / f"{system}_dftpy.out"
    log_path = case_dir / f"{system}_relax.log"
    steps = read_bfgs_steps(log_path)

    if system not in {"pristine", "vacancy"}:
        return None, "", "Unknown VASP role; no energy assigned.", None, None

    if structure_state == "initial":
        if steps:
            step0 = steps[0]
            return (
                float(step0["energy"]),
                f"{log_path.name}: BFGS step 0 SCF energy",
                "Initial-structure SCF energy recorded inside relaxation log.",
                float(step0["fmax"]),
                int(step0["step"]),
            )
        return None, f"{log_path.name}", "No BFGS step 0 found; standalone SCF needed.", None, None

    if structure_state in {"final_vc_relaxed", "final_relaxed_alias"}:
        energy = read_total_energy(out_path)
        if energy is not None:
            fmax = float(steps[-1]["fmax"]) if steps else None
            step = int(steps[-1]["step"]) if steps else None
            return (
                energy,
                f"{out_path.name}: post-relax DFTpy SCF/evaluate output",
                "Final relaxed-structure total energy from DFTpy output.",
                fmax,
                step,
            )
        if steps:
            last = steps[-1]
            return (
                float(last["energy"]),
                f"{log_path.name}: final BFGS energy",
                "Final relaxed-structure energy from relaxation log; standalone SCF may be requested.",
                float(last["fmax"]),
                int(last["step"]),
            )
        return None, f"{out_path.name}", "No final energy found; standalone SCF needed.", None, None

    return None, "", "No energy rule for this VASP role; standalone SCF needed.", None, None


def load_case_result(case_dir: Path) -> dict:
    path = case_dir / "result.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(errors="ignore"))
    except Exception:
        return {}


def write_markdown(rows: list[dict[str, object]], out_path: Path) -> None:
    columns = [
        "file_name",
        "spacing_A",
        "system",
        "structure_state",
        "number_of_atoms",
        "volume_A3",
        "total_energy_eV",
        "energy_source",
    ]
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(columns) + " |\n")
        fh.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            vals = []
            for col in columns:
                val = row.get(col, "")
                if isinstance(val, float):
                    if col == "volume_A3":
                        vals.append(f"{val:.6f}")
                    elif col == "total_energy_eV":
                        vals.append(f"{val:.9f}")
                    else:
                        vals.append(f"{val:g}")
                else:
                    vals.append(str(val))
            fh.write("| " + " | ".join(vals) + " |\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", required=True, type=Path, help="Input tar.gz path.")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--keep-extracted",
        action="store_true",
        help="Keep extracted tar contents for review.",
    )
    args = parser.parse_args()

    tar_path = args.tar.resolve()
    outdir = args.outdir.resolve()
    if not tar_path.exists():
        raise FileNotFoundError(tar_path)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.keep_extracted:
        extract_dir = outdir / "extracted_tar_for_audit"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)
    else:
        extract_dir = Path(tempfile.mkdtemp(prefix="vasp_audit_"))
    safe_extract_tar(tar_path, extract_dir)

    vasp_files = sorted(extract_dir.rglob("*.vasp"))
    rows: list[dict[str, object]] = []
    for vasp in vasp_files:
        atoms = read(str(vasp), format="vasp")
        system, structure_state = classify_vasp(vasp)
        energy, source, note, fmax, bfgs_step = energy_for_vasp(vasp)
        result = load_case_result(vasp.parent)
        rel_path = vasp.relative_to(extract_dir)
        rows.append(
            {
                "file_name": str(rel_path).replace("\\", "/"),
                "spacing_A": parse_spacing(vasp),
                "system": system,
                "structure_state": structure_state,
                "number_of_atoms": len(atoms),
                "volume_A3": atoms.get_volume(),
                "cell_a_A": atoms.cell.lengths()[0],
                "cell_b_A": atoms.cell.lengths()[1],
                "cell_c_A": atoms.cell.lengths()[2],
                "alpha_deg": atoms.cell.angles()[0],
                "beta_deg": atoms.cell.angles()[1],
                "gamma_deg": atoms.cell.angles()[2],
                "total_energy_eV": energy,
                "energy_source": source,
                "scf_note": note,
                "final_fmax_eV_A_if_available": fmax,
                "bfgs_step_if_available": bfgs_step,
                "kedf": result.get("kedf", ""),
                "xc": result.get("xc", ""),
                "vacancy_formation_energy_eV_case": result.get("vacancy_formation_energy_eV", ""),
                "sha256_16": file_sha256_short(vasp),
            }
        )

    csv_path = outdir / "vasp_file_total_energy_table.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_markdown(rows, outdir / "vasp_file_total_energy_table.md")

    status_counts: dict[str, int] = {}
    for row in rows:
        key = str(row["scf_note"])
        status_counts[key] = status_counts.get(key, 0) + 1
    with (outdir / "energy_source_audit.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scf_note", "count"])
        for key, count in sorted(status_counts.items()):
            writer.writerow([key, count])

    readme = outdir / "README_professor_targz_vasp_table.md"
    readme.write_text(
        "\n".join(
            [
                "# VASP file total-energy table",
                "",
                "This folder summarizes every `.vasp` file in the previously sent tar.gz.",
                "",
                "Main table:",
                "- `vasp_file_total_energy_table.csv`",
                "- `vasp_file_total_energy_table.md`",
                "",
                "Energy-source convention:",
                "- Initial raw/start structures use the BFGS step-0 SCF energy recorded in the relaxation log.",
                "- Final relaxed structures use the `total energy (eV)` written in the corresponding DFTpy output after relaxation.",
                "- The `energy_source` and `scf_note` columns are included so the energy origin is explicit.",
                "",
                f"Input archive: `{tar_path.name}`",
                f"Number of VASP files summarized: {len(rows)}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    if not args.keep_extracted:
        shutil.rmtree(extract_dir)

    print(f"Wrote: {csv_path}")
    print(f"Rows : {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
