from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add case-local pseudo and README files to an existing DFTpy vacancy series."
    )
    parser.add_argument("--rootdir", required=True)
    parser.add_argument("--pp", required=True)
    args = parser.parse_args()

    root = Path(args.rootdir).expanduser().resolve()
    pp_source = Path(args.pp).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not pp_source.is_file():
        raise FileNotFoundError(pp_source)

    rows: list[dict[str, object]] = []
    for manifest_path in sorted(root.rglob("point_manifest.json")):
        case = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        case_pp = case / pp_source.name
        shutil.copy2(pp_source, case_pp)
        previous_pp = str(manifest.get("pp_file", ""))
        manifest["pp_source_file"] = str(manifest.get("pp_source_file", previous_pp))
        manifest["pp_file"] = case_pp.name
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        scan_type = str(manifest.get("scan_type", ""))
        defect_label = "divacancy" if scan_type == "pair" else "vacancy"
        repeat = manifest.get("conventional_repeat", [])
        readme = f"""DFTpy {defect_label} calculation case

Setting: {manifest.get('setting', case.name)}
Cell repeat: {repeat}
Atoms: {manifest.get('pristine_n_atoms')} -> {manifest.get('vacancy_n_atoms')}
XC: {manifest.get('xc')}
KEDF: {manifest.get('kedf')}
lambda/x: {manifest.get('kedf_x')}
mu/y: {manifest.get('kedf_y')}
Grid spacing: {manifest.get('spacing_A')} A
Target fmax: {manifest.get('fmax_eV_per_A')} eV/A
Pseudopotential: {case_pp.name}

The INI files define DFTpy density optimization and Energy/Force/Stress
evaluation. Full atom-and-cell relaxation is performed by
scripts/run_dftpy_vcrelax_vacancy_one.py using ASE FrechetCellFilter plus BFGS.

Formation energy:
E_f = E_defect(N-n) - ((N-n)/N) E_pristine(N)
"""
        (case / "README_CASE.txt").write_text(readme, encoding="utf-8")
        rows.append(
            {
                "case": str(case.relative_to(root)),
                "pseudo": case_pp.name,
                "pseudo_size_bytes": case_pp.stat().st_size,
                "readme": "README_CASE.txt",
                "result_present": (case / "result.json").exists(),
            }
        )

    audit = {
        "rootdir": str(root),
        "case_count": len(rows),
        "pseudo_source": str(pp_source),
        "cases": rows,
    }
    (root / "CASE_REPRODUCIBILITY_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Materialized {len(rows)} DFTpy case folders under {root}")


if __name__ == "__main__":
    main()
