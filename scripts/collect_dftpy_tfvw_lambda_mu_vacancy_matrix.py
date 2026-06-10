from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MATRIX_FIELDS = {
    "vacancy_formation_energy_eV": "Vacancy formation energy (eV)",
    "pristine_energy_eV": "Pristine total energy (eV)",
    "vacancy_energy_eV": "Vacancy total energy (eV)",
    "pristine_final_fmax_eV_A": "Pristine final fmax (eV/A)",
    "vacancy_final_fmax_eV_A": "Vacancy final fmax (eV/A)",
    "pristine_volume_A3": "Pristine final volume (A^3)",
    "vacancy_volume_A3": "Vacancy final volume (A^3)",
}


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def final_fmax(path: Path) -> float:
    if not path.exists():
        return math.nan
    value = math.nan
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            value = float(parts[-1])
        except ValueError:
            continue
    return value


def product(values: list[float]) -> float:
    result = 1.0
    for value in values:
        result *= float(value)
    return result


def cell_volume(result: dict[str, object], prefix: str) -> float:
    direct = result.get(f"{prefix}_volume_A3")
    if direct is not None:
        return float(direct)
    lengths = result.get(f"{prefix}_cell_lengths_A")
    angles = result.get(f"{prefix}_cell_angles_deg")
    if isinstance(lengths, list) and len(lengths) == 3:
        if not isinstance(angles, list) or len(angles) != 3:
            return product([float(value) for value in lengths])
        alpha, beta, gamma = np.radians(np.asarray(angles, dtype=float))
        factor = math.sqrt(
            max(
                0.0,
                1.0
                + 2.0 * math.cos(alpha) * math.cos(beta) * math.cos(gamma)
                - math.cos(alpha) ** 2
                - math.cos(beta) ** 2
                - math.cos(gamma) ** 2,
            )
        )
        return product([float(value) for value in lengths]) * factor
    return math.nan


def settings(rootdir: Path) -> list[str]:
    setting_file = rootdir / "settings_weight_scan.txt"
    if setting_file.exists():
        output = []
        for line in setting_file.read_text(encoding="utf-8").splitlines():
            tokens = line.split()
            if tokens:
                output.append(tokens[-1])
        return output
    return sorted(path.name for path in (rootdir / "weight_scan").iterdir() if path.is_dir())


def status_for(case_dir: Path, result_path: Path) -> str:
    if result_path.exists() and result_path.stat().st_size:
        return "DONE"
    if list(case_dir.glob("RESCUE_TIMEOUT_*.txt")):
        return "TIMEOUT"
    if (case_dir / "RESCUE_FAILED.txt").exists():
        return "FAILED"
    return "MISSING"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matrix_rows(
    rows: list[dict[str, object]],
    lambdas: list[float],
    mus: list[float],
    field: str,
) -> list[dict[str, object]]:
    lookup = {
        (float(row["lambda_tf"]), float(row["mu_vw"])): row[field]
        for row in rows
        if row["status"] == "DONE"
    }
    output = []
    for lambda_tf in lambdas:
        row: dict[str, object] = {"lambda/mu": lambda_tf}
        for mu_vw in mus:
            row[f"{mu_vw:.1f}"] = lookup.get((lambda_tf, mu_vw), math.nan)
        output.append(row)
    return output


def plot_heatmap(
    path: Path,
    rows: list[dict[str, object]],
    lambdas: list[float],
    mus: list[float],
    field: str,
    title: str,
) -> None:
    lookup = {
        (float(row["lambda_tf"]), float(row["mu_vw"])): float(row[field])
        for row in rows
        if row["status"] == "DONE"
    }
    matrix = np.asarray(
        [[lookup.get((lambda_tf, mu_vw), math.nan) for mu_vw in mus] for lambda_tf in lambdas]
    )
    fig, ax = plt.subplots(figsize=(8.2, 6.6), constrained_layout=True)
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(mus)), [f"{value:.1f}" for value in mus])
    ax.set_yticks(range(len(lambdas)), [f"{value:.1f}" for value in lambdas])
    ax.set_xlabel("mu_vW")
    ax.set_ylabel("lambda_TF")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect a DFTpy TFvW lambda-mu vacancy vc-relax matrix."
    )
    parser.add_argument("--rootdir", required=True)
    args = parser.parse_args()
    rootdir = Path(args.rootdir).expanduser().resolve()
    scan_root = rootdir / "weight_scan"
    if not scan_root.exists():
        raise FileNotFoundError(scan_root)

    rows: list[dict[str, object]] = []
    for index, setting in enumerate(settings(rootdir)):
        case_dir = scan_root / setting
        manifest_path = case_dir / "point_manifest.json"
        result_path = case_dir / "result.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        result = read_json(result_path) if result_path.exists() and result_path.stat().st_size else {}
        timeout_markers = sorted(path.name for path in case_dir.glob("RESCUE_TIMEOUT_*.txt"))
        rows.append(
            {
                "index": index,
                "setting": setting,
                "status": status_for(case_dir, result_path),
                "lambda_tf": float(result.get("kedf_x", manifest.get("kedf_x", math.nan))),
                "mu_vw": float(result.get("kedf_y", manifest.get("kedf_y", math.nan))),
                "xc": result.get("xc", manifest.get("xc", "")),
                "kedf": result.get("kedf", manifest.get("kedf", "")),
                "N_pristine": result.get(
                    "pristine_n_atoms", manifest.get("pristine_n_atoms", math.nan)
                ),
                "N_vacancy": result.get(
                    "vacancy_n_atoms", manifest.get("vacancy_n_atoms", math.nan)
                ),
                "vacancy_concentration_percent": result.get(
                    "vacancy_concentration_percent", math.nan
                ),
                "spacing_A": result.get("spacing_A", manifest.get("spacing_A", math.nan)),
                "target_fmax_eV_A": result.get(
                    "fmax_eV_per_A", manifest.get("fmax_eV_per_A", math.nan)
                ),
                "pristine_energy_eV": result.get("pristine_energy_eV", math.nan),
                "vacancy_energy_eV": result.get("vacancy_energy_eV", math.nan),
                "vacancy_formation_energy_eV": result.get(
                    "vacancy_formation_energy_eV", math.nan
                ),
                "pristine_final_fmax_eV_A": final_fmax(case_dir / "pristine_relax.log"),
                "vacancy_final_fmax_eV_A": final_fmax(case_dir / "vacancy_relax.log"),
                "pristine_volume_A3": cell_volume(result, "pristine"),
                "vacancy_volume_A3": cell_volume(result, "vacancy"),
                "pristine_cell_lengths_A": json.dumps(
                    result.get("pristine_cell_lengths_A", [])
                ),
                "vacancy_cell_lengths_A": json.dumps(
                    result.get("vacancy_cell_lengths_A", [])
                ),
                "timeout_marker_count": len(timeout_markers),
                "timeout_markers": ";".join(timeout_markers),
                "failed_marker": (case_dir / "RESCUE_FAILED.txt").exists(),
                "case_dir": str(case_dir),
            }
        )

    rows.sort(key=lambda row: int(row["index"]))
    lambdas = sorted(
        {float(row["lambda_tf"]) for row in rows if math.isfinite(float(row["lambda_tf"]))}
    )
    mus = sorted({float(row["mu_vw"]) for row in rows if math.isfinite(float(row["mu_vw"]))})
    analysis = rootdir / "analysis"
    tables_dir = analysis / "tables"
    figures_dir = analysis / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    write_csv(tables_dir / "lambda_mu_vacancy_long_summary.csv", rows)

    for field, title in MATRIX_FIELDS.items():
        matrix = matrix_rows(rows, lambdas, mus, field)
        write_csv(tables_dir / f"matrix_{field}.csv", matrix)
        plot_heatmap(
            figures_dir / f"heatmap_{field}.png",
            rows,
            lambdas,
            mus,
            field,
            title,
        )

    counts = Counter(str(row["status"]) for row in rows)
    missing_rows = [row for row in rows if row["status"] != "DONE"]
    write_csv(tables_dir / "incomplete_cases.csv", missing_rows)
    note = [
        "# DFTpy TFvW Lambda-Mu Vacancy Matrix Audit",
        "",
        f"- Expected settings: {len(rows)}",
        f"- DONE: {counts['DONE']}",
        f"- TIMEOUT: {counts['TIMEOUT']}",
        f"- FAILED: {counts['FAILED']}",
        f"- MISSING: {counts['MISSING']}",
        f"- Lambda values: {', '.join(f'{value:g}' for value in lambdas)}",
        f"- Mu values: {', '.join(f'{value:g}' for value in mus)}",
        "",
        "Incomplete settings:",
        "",
    ]
    note.extend(
        f"- index {int(row['index']):02d}: {row['setting']} ({row['status']})"
        for row in missing_rows
    )
    (analysis / "COMPLETION_AUDIT.md").write_text("\n".join(note) + "\n", encoding="utf-8")
    print("============================================================")
    print("DFTpy TFvW lambda-mu vacancy matrix collected")
    print("============================================================")
    print(f"Root     : {rootdir}")
    print(f"Expected : {len(rows)}")
    for status in ("DONE", "TIMEOUT", "FAILED", "MISSING"):
        print(f"{status:8s} : {counts[status]}")
    print(f"Summary  : {tables_dir / 'lambda_mu_vacancy_long_summary.csv'}")


if __name__ == "__main__":
    main()
