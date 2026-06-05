from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ase.io import read


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit representative structures in the final NUS vacancy database.")
    parser.add_argument("--rootdir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.rootdir).expanduser().resolve()
    paths = [
        ("QE", "pristine_start", root / "02_QE_PBE_VCRELAX_3x3x3_RAW" / "pristine_start.vasp"),
        ("QE", "vacancy_start", root / "02_QE_PBE_VCRELAX_3x3x3_RAW" / "vacancy_start.vasp"),
    ]

    for label, folder in [
        ("DFTpy_TFVW", "03_DFTPY_LDA_TFVW_RAW"),
        ("DFTpy_WT", "04_DFTPY_LDA_WT_RAW"),
        ("DFTpy_SM", "05_DFTPY_LDA_SM_RAW"),
    ]:
        matches = list((root / folder).glob("*/spacing_scan/spacing_0p20A"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one spacing_0p20A case under {folder}, found {len(matches)}")
        case = matches[0]
        for name in ["pristine_raw", "vacancy_start", "pristine_vc_relaxed", "vacancy_vc_relaxed"]:
            paths.append((label, name, case / f"{name}.vasp"))

    rows: list[dict[str, object]] = []
    for method, structure, path in paths:
        atoms = read(path)
        lengths = atoms.cell.lengths()
        angles = atoms.cell.angles()
        distances = atoms.get_all_distances(mic=True)
        positive_distances = distances[distances > 1e-10]
        max_angle_deviation = max(abs(value - 90.0) for value in angles)
        rows.append(
            {
                "method": method,
                "structure": structure,
                "atoms": len(atoms),
                "a_A": f"{lengths[0]:.6f}",
                "b_A": f"{lengths[1]:.6f}",
                "c_A": f"{lengths[2]:.6f}",
                "alpha_deg": f"{angles[0]:.6f}",
                "beta_deg": f"{angles[1]:.6f}",
                "gamma_deg": f"{angles[2]:.6f}",
                "max_angle_deviation_deg": f"{max_angle_deviation:.9f}",
                "volume_A3": f"{atoms.get_volume():.6f}",
                "minimum_distance_A": f"{positive_distances.min():.6f}",
                "orthogonal_within_0p001deg": max_angle_deviation < 1e-3,
                "file": str(path.relative_to(root)),
            }
        )

    output = root / "00_START_HERE" / "STRUCTURE_AUDIT.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {output}")
    print(f"Rows : {len(rows)}")
    print(f"All orthogonal within 0.001 deg: {all(row['orthogonal_within_0p001deg'] for row in rows)}")
    print(f"Minimum atomic distance: {min(float(row['minimum_distance_A']) for row in rows):.6f} A")


if __name__ == "__main__":
    main()
