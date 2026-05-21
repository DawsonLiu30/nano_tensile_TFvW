from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import neighbor_list


ROOT = Path(__file__).resolve().parents[1]
BOHR_TO_ANG = 0.529177210903


STRUCTURE_NAMES = {"POSCAR", "CONTCAR"}
STRUCTURE_EXTS = {".vasp", ".cif", ".xyz"}
QE_EXTS = {".in"}
SKIP_PARTS = {
    ".git",
    "__pycache__",
    "tmp",
    ".ipynb_checkpoints",
    ".vscode",
}


@dataclass
class ExpectedInfo:
    expected_atoms: int | None = None
    expected_sites: int | None = None
    label: str = ""


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _default_roots() -> list[Path]:
    candidates = [
        Path(r"C:\Users\dawso\Desktop\latest_professor_pull_20260511"),
        ROOT / "cases",
        ROOT / "results" / "professor_review",
    ]
    return [path for path in candidates if path.exists()]


def _is_skipped(path: Path) -> bool:
    lowered = {part.lower() for part in path.parts}
    return bool(lowered & SKIP_PARTS)


def _is_structure_file(path: Path) -> bool:
    if path.name in STRUCTURE_NAMES:
        return True
    if path.suffix.lower() in STRUCTURE_EXTS:
        return True
    if path.suffix.lower() in QE_EXTS:
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            return False
        return "ATOMIC_POSITIONS" in text and ("CELL_PARAMETERS" in text or "ibrav" in text)
    return False


def _find_structure_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or _is_skipped(path):
                continue
            if _is_structure_file(path):
                files.append(path)
    return sorted(set(files), key=lambda p: str(p).lower())


def _parse_float_token(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e").strip().strip(","))


def _parse_qe_namelist_scalar(text: str, key: str) -> float | None:
    match = re.search(rf"\b{re.escape(key)}\s*=\s*([0-9.+\-EeDd]+)", text, flags=re.I)
    if not match:
        return None
    return _parse_float_token(match.group(1))


def _parse_qe_cell(text: str) -> tuple[np.ndarray | None, str]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.strip().upper().startswith("CELL_PARAMETERS"):
            continue
        unit = "angstrom"
        lower = line.lower()
        if "bohr" in lower:
            unit = "bohr"
        elif "alat" in lower:
            unit = "alat"
        vectors: list[list[float]] = []
        for vec_line in lines[i + 1 : i + 4]:
            parts = vec_line.split()
            if len(parts) < 3:
                return None, "bad CELL_PARAMETERS block"
            vectors.append([_parse_float_token(parts[j]) for j in range(3)])
        cell = np.asarray(vectors, dtype=float)
        if unit == "bohr":
            cell *= BOHR_TO_ANG
        elif unit == "alat":
            alat = _parse_qe_namelist_scalar(text, "celldm(1)")
            if alat is None:
                alat = _parse_qe_namelist_scalar(text, "A")
                if alat is None:
                    return None, "CELL_PARAMETERS alat without celldm(1) or A"
                cell *= alat
            else:
                cell *= alat * BOHR_TO_ANG
        return cell, f"CELL_PARAMETERS {unit}"

    ibrav = _parse_qe_namelist_scalar(text, "ibrav")
    if ibrav is not None and int(ibrav) != 0:
        ibrav_int = int(ibrav)
        alat_bohr = _parse_qe_namelist_scalar(text, "celldm(1)")
        alat_ang = _parse_qe_namelist_scalar(text, "A")
        if alat_bohr is not None:
            a = alat_bohr * BOHR_TO_ANG
        elif alat_ang is not None:
            a = alat_ang
        else:
            return None, f"ibrav={ibrav_int} without celldm(1) or A"
        if ibrav_int == 1:
            return np.diag([a, a, a]), "ibrav=1 cubic"
        if ibrav_int == 2:
            # Quantum ESPRESSO ibrav=2 is fcc with primitive vectors.
            cell = 0.5 * a * np.array(
                [
                    [-1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [-1.0, 1.0, 0.0],
                ],
                dtype=float,
            )
            return cell, "ibrav=2 fcc primitive"
        return None, f"ibrav={ibrav_int} without explicit parser"
    return None, "missing CELL_PARAMETERS"


def _parse_qe_positions(text: str, cell: np.ndarray) -> tuple[list[str], np.ndarray, str]:
    lines = text.splitlines()
    nat_value = _parse_qe_namelist_scalar(text, "nat")
    nat = int(nat_value) if nat_value is not None else None
    for i, line in enumerate(lines):
        if not line.strip().upper().startswith("ATOMIC_POSITIONS"):
            continue
        lower = line.lower()
        unit = "alat"
        if "crystal" in lower:
            unit = "crystal"
        elif "angstrom" in lower:
            unit = "angstrom"
        elif "bohr" in lower:
            unit = "bohr"
        symbols: list[str] = []
        coords: list[list[float]] = []
        for raw in lines[i + 1 :]:
            stripped = raw.strip()
            if not stripped or stripped.upper().startswith(
                (
                    "K_POINTS",
                    "CELL_PARAMETERS",
                    "ATOMIC_SPECIES",
                    "OCCUPATIONS",
                    "CONSTRAINTS",
                    "&",
                    "/",
                )
            ):
                break
            parts = stripped.split()
            if len(parts) < 4:
                break
            symbols.append(parts[0])
            coords.append([_parse_float_token(parts[j]) for j in range(1, 4)])
            if nat is not None and len(symbols) >= nat:
                break
        arr = np.asarray(coords, dtype=float)
        if unit == "crystal":
            arr = arr @ cell
        elif unit == "bohr":
            arr *= BOHR_TO_ANG
        elif unit == "alat":
            alat = _parse_qe_namelist_scalar(text, "celldm(1)")
            if alat is None:
                alat = _parse_qe_namelist_scalar(text, "A")
                if alat is None:
                    raise ValueError("ATOMIC_POSITIONS alat without celldm(1) or A")
                arr *= alat
            else:
                arr *= alat * BOHR_TO_ANG
        return symbols, arr, f"ATOMIC_POSITIONS {unit}"
    raise ValueError("missing ATOMIC_POSITIONS")


def _read_qe_input(path: Path) -> tuple[Atoms, str]:
    text = path.read_text(errors="ignore")
    cell, cell_note = _parse_qe_cell(text)
    if cell is None:
        raise ValueError(cell_note)
    symbols, positions, pos_note = _parse_qe_positions(text, cell)
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    return atoms, f"qe-input; {cell_note}; {pos_note}"


def _read_atoms(path: Path) -> tuple[Atoms, str]:
    if path.suffix.lower() == ".in":
        return _read_qe_input(path)
    if path.name in STRUCTURE_NAMES or path.suffix.lower() == ".vasp":
        return read(path, format="vasp"), "vasp"
    return read(path), path.suffix.lower().lstrip(".")


def _min_distance(atoms: Atoms) -> float:
    n_atoms = len(atoms)
    if n_atoms < 2:
        return math.nan
    try:
        distances = neighbor_list("d", atoms, cutoff=4.0, self_interaction=False)
        distances = distances[np.isfinite(distances) & (distances > 1e-8)]
        if len(distances) > 0:
            return float(np.min(distances))
    except Exception:
        pass

    positions = atoms.get_positions()
    if n_atoms > 5000:
        return math.nan
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    dist[dist < 1e-8] = np.nan
    return float(np.nanmin(dist))


def _z_gap(atoms: Atoms) -> float:
    if len(atoms) == 0:
        return math.nan
    cell = atoms.get_cell().array
    c_len = float(np.linalg.norm(cell[2]))
    if c_len <= 0.0 or not np.isfinite(c_len):
        return math.nan
    scaled_z = atoms.get_scaled_positions(wrap=True)[:, 2]
    scaled_z = np.sort(scaled_z % 1.0)
    if len(scaled_z) == 1:
        return c_len
    gaps = np.diff(np.r_[scaled_z, scaled_z[0] + 1.0])
    return float(np.max(gaps) * c_len)


def _cell_metrics(atoms: Atoms) -> dict[str, float]:
    cell = atoms.get_cell()
    lengths = cell.lengths()
    angles = cell.angles()
    return {
        "a_A": float(lengths[0]),
        "b_A": float(lengths[1]),
        "c_A": float(lengths[2]),
        "alpha_deg": float(angles[0]),
        "beta_deg": float(angles[1]),
        "gamma_deg": float(angles[2]),
        "volume_A3": float(abs(atoms.get_volume())),
    }


def _infer_expected(path: Path) -> ExpectedInfo:
    text = str(path).replace("\\", "/")
    lower = text.lower()
    info = ExpectedInfo()

    def is_pristine() -> bool:
        return any(key in lower for key in ["pristine", "perfect"])

    def is_vacancy() -> bool:
        return "vacancy" in lower or "/vac_" in lower or "_vac" in lower

    if "qe_vacancy_convergence_20260506" in lower:
        if is_pristine():
            return ExpectedInfo(expected_atoms=64, expected_sites=64, label="QE vacancy pristine primitive 4x4x4")
        if is_vacancy():
            return ExpectedInfo(expected_atoms=63, expected_sites=64, label="QE vacancy cell primitive 4x4x4")

    if "dftpy_vacancy_convergence_primitive4" in lower:
        if is_pristine():
            return ExpectedInfo(expected_atoms=64, expected_sites=64, label="DFTpy vacancy pristine primitive 4x4x4")
        if is_vacancy():
            return ExpectedInfo(expected_atoms=63, expected_sites=64, label="DFTpy vacancy cell primitive 4x4x4")

    match = re.search(r"prim_(\d+)x\d+x\d+", lower)
    if match:
        n = int(match.group(1))
        sites = n**3
        if is_pristine():
            return ExpectedInfo(expected_atoms=sites, expected_sites=sites, label=f"primitive {n}x{n}x{n} pristine")
        if is_vacancy():
            return ExpectedInfo(expected_atoms=sites - 1, expected_sites=sites, label=f"primitive {n}x{n}x{n} vacancy")
        return ExpectedInfo(expected_sites=sites, label=f"primitive {n}x{n}x{n}")

    return info


def _classify_work_item(path: Path) -> str:
    lower = str(path).replace("\\", "/").lower()
    if "qe_bulk_b_convergence_20260506" in lower:
        return "QE bulk B/EOS"
    if "qe_vacancy_convergence_20260506" in lower:
        return "QE vacancy convergence"
    if "dftpy_vacancy_convergence_primitive4_qe_a0" in lower:
        return "DFTpy vacancy spacing fixed QE-a0"
    if "dftpy_vacancy_convergence_primitive4_20260508" in lower:
        return "DFTpy vacancy spacing own-a0"
    if "dftpy_vacancy_size_primitive_qe_a0" in lower:
        return "DFTpy primitive-size fixed QE-a0"
    if "finite_grip" in lower:
        return "legacy finite-name folder"
    if "paper_periodic" in lower:
        return "periodic nanowire/nanoprism"
    return "other"


def _make_row(path: Path, root_paths: list[Path]) -> dict[str, object]:
    rel = str(path)
    for root in root_paths:
        try:
            rel = str(path.relative_to(root))
            break
        except ValueError:
            continue

    row: dict[str, object] = {
        "path": str(path),
        "relative_path": rel,
        "work_item": _classify_work_item(path),
        "file_name": path.name,
        "read_ok": False,
        "reader": "",
        "n_atoms": math.nan,
        "formula": "",
        "expected_atoms": "",
        "expected_sites": "",
        "expected_label": "",
        "volume_A3": math.nan,
        "volume_per_atom_A3": math.nan,
        "volume_per_expected_site_A3": math.nan,
        "a_A": math.nan,
        "b_A": math.nan,
        "c_A": math.nan,
        "alpha_deg": math.nan,
        "beta_deg": math.nan,
        "gamma_deg": math.nan,
        "min_distance_A": math.nan,
        "max_z_gap_A": math.nan,
        "flags": "",
        "error": "",
    }

    flags: list[str] = []
    expected = _infer_expected(path)
    row["expected_atoms"] = expected.expected_atoms if expected.expected_atoms is not None else ""
    row["expected_sites"] = expected.expected_sites if expected.expected_sites is not None else ""
    row["expected_label"] = expected.label

    try:
        atoms, reader = _read_atoms(path)
        row["reader"] = reader
        row["read_ok"] = True
        row["n_atoms"] = len(atoms)
        row["formula"] = atoms.get_chemical_formula()

        metrics = _cell_metrics(atoms)
        row.update(metrics)
        volume = float(metrics["volume_A3"])
        if len(atoms) > 0 and np.isfinite(volume):
            row["volume_per_atom_A3"] = volume / len(atoms)
        if expected.expected_sites and np.isfinite(volume):
            row["volume_per_expected_site_A3"] = volume / expected.expected_sites

        row["min_distance_A"] = _min_distance(atoms)
        row["max_z_gap_A"] = _z_gap(atoms)

        if any(symbol != "Al" for symbol in atoms.get_chemical_symbols()):
            flags.append("NON_AL_SPECIES")
        if expected.expected_atoms is not None and len(atoms) != expected.expected_atoms:
            flags.append("ATOM_COUNT_MISMATCH")

        primitive_match = re.search(r"prim_(\d+)x\d+x\d+", str(path).lower())
        if primitive_match:
            n = int(primitive_match.group(1))
            conventional_pristine = 4 * n**3
            conventional_vacancy = conventional_pristine - 1
            if len(atoms) in {conventional_pristine, conventional_vacancy}:
                flags.append("PRIMITIVE_LABEL_BUT_CONVENTIONAL_ATOM_COUNT")

        min_dist = float(row["min_distance_A"])
        if np.isfinite(min_dist):
            if min_dist < 1.5:
                flags.append("SEVERE_OVERLAP_MIN_DIST_LT_1P5A")
            elif min_dist < 2.2:
                flags.append("SHORT_AL_AL_DISTANCE_LT_2P2A")

        z_gap = float(row["max_z_gap_A"])
        if np.isfinite(z_gap) and z_gap > 5.0:
            flags.append("LARGE_Z_EMPTY_INTERVAL_GT_5A")

        vol_site = row["volume_per_expected_site_A3"]
        if isinstance(vol_site, float) and np.isfinite(vol_site):
            if vol_site < 12.0 or vol_site > 22.0:
                flags.append("SUSPICIOUS_VOLUME_PER_EXPECTED_SITE")

        if not np.isfinite(volume) or volume <= 0.0:
            flags.append("BAD_CELL_VOLUME")

    except Exception as exc:
        row["error"] = repr(exc)
        flags.append("READ_FAIL")

    row["flags"] = ";".join(flags) if flags else "OK"
    return row


def _safe_copy_name(path: Path) -> str:
    text = str(path).replace(":", "").replace("\\", "__").replace("/", "__")
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)
    return text[-180:]


def _copy_vesta_review_files(df: pd.DataFrame, outdir: Path, max_files: int) -> None:
    review_dir = outdir / "vesta_review_files"
    review_dir.mkdir(parents=True, exist_ok=True)
    flagged = df[(df["read_ok"] == True) & (df["flags"] != "OK")].copy()  # noqa: E712
    priority_flags = [
        "ATOM_COUNT_MISMATCH",
        "PRIMITIVE_LABEL_BUT_CONVENTIONAL_ATOM_COUNT",
        "SEVERE_OVERLAP_MIN_DIST_LT_1P5A",
        "SHORT_AL_AL_DISTANCE_LT_2P2A",
        "LARGE_Z_EMPTY_INTERVAL_GT_5A",
        "SUSPICIOUS_VOLUME_PER_EXPECTED_SITE",
    ]

    def score(flags: str) -> int:
        return sum((len(priority_flags) - i) for i, flag in enumerate(priority_flags) if flag in flags)

    flagged["review_score"] = flagged["flags"].map(score)
    selected = flagged.sort_values(["review_score", "work_item", "relative_path"], ascending=[False, True, True])

    copied: list[dict[str, str]] = []
    for _, row in selected.head(max_files).iterrows():
        src = Path(str(row["path"]))
        dst = review_dir / _safe_copy_name(src)
        try:
            if src.suffix.lower() == ".in":
                atoms, _ = _read_atoms(src)
                dst = dst.with_suffix(".vasp")
                write(dst, atoms, format="vasp", direct=True, vasp5=True)
            else:
                shutil.copy2(src, dst)
            copied.append(
                {
                    "copied_file": str(dst),
                    "source_file": str(src),
                    "work_item": str(row["work_item"]),
                    "flags": str(row["flags"]),
                }
            )
        except Exception as exc:
            copied.append(
                {
                    "copied_file": "",
                    "source_file": str(src),
                    "work_item": str(row["work_item"]),
                    "flags": f"COPY_FAIL {exc!r}",
                }
            )
    if copied:
        pd.DataFrame(copied).to_csv(review_dir / "VESTA_REVIEW_FILE_INDEX.csv", index=False)


def _write_markdown(df: pd.DataFrame, outpath: Path, roots: list[Path]) -> None:
    total = len(df)
    read_fail = int((df["read_ok"] == False).sum())  # noqa: E712
    flagged = df[df["flags"] != "OK"]
    lines: list[str] = []
    lines.append("# QE / DFTpy Structure Audit")
    lines.append("")
    lines.append("This audit checks structure files programmatically before VESTA inspection.")
    lines.append("")
    lines.append("## Roots")
    lines.append("")
    for root in roots:
        lines.append(f"- `{root}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total structure-like files scanned: `{total}`")
    lines.append(f"- Read failures: `{read_fail}`")
    lines.append(f"- Files with geometry/count flags: `{len(flagged)}`")
    lines.append("")
    lines.append("## By Work Item")
    lines.append("")
    by_item = (
        df.assign(flagged=df["flags"] != "OK")
        .groupby("work_item")
        .agg(files=("path", "count"), read_fail=("read_ok", lambda x: int((x == False).sum())), flagged=("flagged", "sum"))  # noqa: E712
        .reset_index()
        .sort_values("work_item")
    )
    lines.append(by_item.to_markdown(index=False))
    lines.append("")
    lines.append("## Flag Definitions")
    lines.append("")
    lines.append("- `ATOM_COUNT_MISMATCH`: atom count does not match the expected pristine/vacancy supercell.")
    lines.append("- `PRIMITIVE_LABEL_BUT_CONVENTIONAL_ATOM_COUNT`: folder says primitive but atom count matches conventional fcc replication.")
    lines.append("- `SHORT_AL_AL_DISTANCE_LT_2P2A`: shortest Al-Al distance is suspiciously short.")
    lines.append("- `SEVERE_OVERLAP_MIN_DIST_LT_1P5A`: atoms are almost certainly overlapping.")
    lines.append("- `LARGE_Z_EMPTY_INTERVAL_GT_5A`: possible finite-length/vacuum gap along z.")
    lines.append("- `SUSPICIOUS_VOLUME_PER_EXPECTED_SITE`: volume per expected fcc site is outside 12-22 A^3.")
    lines.append("- `READ_FAIL`: parser could not read the structure.")
    lines.append("")
    lines.append("## Highest-Priority Flagged Files")
    lines.append("")
    if flagged.empty:
        lines.append("No flagged files.")
    else:
        cols = [
            "work_item",
            "relative_path",
            "n_atoms",
            "expected_atoms",
            "volume_per_expected_site_A3",
            "min_distance_A",
            "max_z_gap_A",
            "flags",
            "error",
        ]
        sample = flagged[cols].head(80).copy()
        lines.append(sample.to_markdown(index=False))
    lines.append("")
    lines.append("## VESTA Review Files")
    lines.append("")
    lines.append("Representative flagged structures were copied to `vesta_review_files/`.")
    lines.append("Open those files in VESTA before showing any structure to the professor.")
    outpath.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit QE/DFTpy structure files for geometry/count mistakes.")
    parser.add_argument("--root", action="append", default=[], help="Root folder to scan. Can be repeated.")
    parser.add_argument("--outdir", default="", help="Output directory.")
    parser.add_argument("--max-vesta-files", type=int, default=80, help="Max flagged files copied for VESTA review.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [_resolve(path) for path in args.root] if args.root else _default_roots()
    roots = [path for path in roots if path.exists()]
    if not roots:
        raise SystemExit("No valid roots to scan.")

    outdir = _resolve(args.outdir) if args.outdir else ROOT / "outputs" / "structure_audit_20260521"
    outdir.mkdir(parents=True, exist_ok=True)

    files = _find_structure_files(roots)
    rows = [_make_row(path, roots) for path in files]
    df = pd.DataFrame(rows)

    csv_path = outdir / "structure_audit_all_files.csv"
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

    flagged_path = outdir / "structure_audit_flagged_files.csv"
    df[df["flags"] != "OK"].to_csv(flagged_path, index=False)

    _copy_vesta_review_files(df, outdir, max_files=args.max_vesta_files)
    _write_markdown(df, outdir / "STRUCTURE_AUDIT_REPORT.md", roots)

    print("============================================================")
    print("QE / DFTpy structure audit completed")
    print("============================================================")
    print(f"Scanned files : {len(df)}")
    print(f"Flagged files : {int((df['flags'] != 'OK').sum())}")
    print(f"Read failures : {int((df['read_ok'] == False).sum())}")  # noqa: E712
    print(f"Output folder : {outdir}")
    print(f"All CSV       : {csv_path}")
    print(f"Flagged CSV   : {flagged_path}")
    print(f"Report        : {outdir / 'STRUCTURE_AUDIT_REPORT.md'}")


if __name__ == "__main__":
    main()
