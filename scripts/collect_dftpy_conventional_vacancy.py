from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

from divacancy_analysis_checks import as_float, qualify_case


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def collect_scan(rootdir: Path, scan_name: str) -> list[dict[str, object]]:
    scan_root = rootdir / scan_name
    rows: list[dict[str, object]] = []
    if not scan_root.exists():
        return rows

    case_dirs = {path for path in scan_root.iterdir() if path.is_dir()}
    settings = rootdir / f"settings_{scan_name}.txt"
    if settings.is_file():
        for line in settings.read_text(encoding="utf-8-sig").splitlines():
            setting = line.strip()
            if setting and not setting.startswith("#") and Path(setting).name == setting:
                case_dirs.add(scan_root / setting)
    for case_dir in sorted(case_dirs):
        manifest_path = case_dir / "point_manifest.json"
        result_path = case_dir / "result.json"
        if not manifest_path.exists():
            rows.append({"setting": case_dir.name, "case_dir": str(case_dir), "done": False,
                         "result_exists": result_path.exists(), **qualify_case(case_dir), "Ef_vac_eV": math.nan,
                         "pair_distance_A": math.nan})
            continue
        try:
            manifest = read_json(manifest_path)
            result = read_json(result_path) if result_path.exists() else {}
        except (OSError, ValueError) as exc:
            rows.append({"setting": case_dir.name, "case_dir": str(case_dir), "done": False,
                         "result_exists": result_path.exists(), "status": "failed", "qualified": False,
                         "qualification_reasons": f"invalid manifest/result: {exc}", "Ef_vac_eV": math.nan,
                         "pair_distance_A": math.nan, "pair_direction_verified": "unknown"})
            continue
        qualification = qualify_case(case_dir)
        n_pristine = as_float(manifest.get("pristine_n_atoms"))
        n_vacancy = as_float(manifest.get("vacancy_n_atoms"))
        vacancy_count = n_pristine - n_vacancy
        ef = result.get("vacancy_formation_energy_eV", math.nan)
        repeat_label = str(manifest.get("conventional_repeat_label", ""))
        if not repeat_label:
            if "conventional_repeat_n" in manifest:
                n = as_float(manifest["conventional_repeat_n"])
                repeat_label = f"conv_{n:g}x{n:g}x{n:g}"
            else:
                repeat = manifest.get("conventional_repeat", ["?", "?", "?"])
                repeat_label = "conv_" + "x".join(str(v) for v in repeat)
        rows.append(
            {
                "scan": scan_name.replace("_scan", ""),
                "setting": str(manifest.get("setting", case_dir.name)),
                # Retained for old table consumers; done now means qualified.
                "done": qualification["qualified"],
                "result_exists": result_path.exists(),
                "cell_basis": str(manifest.get("cell_basis", "unknown")),
                "conventional_repeat_label": repeat_label,
                "N_pristine": n_pristine,
                "N_vacancy": n_vacancy,
                "vacancy_count": vacancy_count,
                "vacancy_concentration_percent": 100.0 * vacancy_count / n_pristine if n_pristine > 0 else math.nan,
                "pair_distance_A": result.get("pair_distance_A", manifest.get("pair_distance_A", math.nan)),
                "pair_selection": str(manifest.get("pair_selection", "")),
                "pair_direction_family": str(manifest.get("pair_direction_family", "")),
                "spacing_A": as_float(manifest.get("spacing_A")),
                "ecut_analogue_eV": as_float(manifest.get("ecut_analogue_eV")),
                "xc": str(result.get("xc", manifest.get("xc", "unknown"))),
                "kedf": str(result.get("kedf", manifest.get("kedf", "unknown"))),
                "kedf_x": as_float(result.get("kedf_x", manifest.get("kedf_x"))),
                "kedf_y": as_float(result.get("kedf_y", manifest.get("kedf_y"))),
                "fmax_eV_A": as_float(manifest.get("fmax_eV_per_A")),
                "ase_optimizer": str(result.get("ase_optimizer", "")),
                "pristine_final_fmax_eV_A": result.get("pristine_final_fmax_eV_A", math.nan),
                "vacancy_final_fmax_eV_A": result.get("vacancy_final_fmax_eV_A", math.nan),
                "pristine_energy_eV": result.get("pristine_energy_eV", math.nan),
                "vacancy_energy_eV": result.get("vacancy_energy_eV", math.nan),
                "Ef_vac_eV": ef,
                "relaxation_mode": result.get("relaxation_mode", "unknown"),
                "target_pressure_GPa": result.get("target_pressure_GPa", math.nan),
                "case_dir": str(case_dir),
                **qualification,
            }
        )
    return rows


def series_key(row: dict, xkey: str) -> tuple:
    """Do not connect different directions or different physical settings."""
    keys = ["xc", "kedf", "kedf_x", "kedf_y", "spacing_A", "N_pristine", "N_vacancy",
            "relaxation_mode", "target_pressure_GPa", "fmax_eV_A", "calculation_code",
            "pseudopotential_sha256", "initial_pristine_geometry_sha256", "cell_basis"]
    if xkey == "N_pristine":
        keys = [key for key in keys if key not in {"N_pristine", "N_vacancy", "initial_pristine_geometry_sha256"}]
    keys = [key for key in keys if key != xkey]
    if row.get("vacancy_count") == 2:
        keys.append("pair_direction_verified")
    return tuple((key, str(row.get(key, "unknown"))) for key in keys)


def add_deltas(rows: list[dict[str, object]], key: str, *, reverse=False) -> None:
    rows.sort(key=lambda row: as_float(row.get(key)), reverse=reverse)
    previous = {}
    for row in rows:
        group = series_key(row, key)
        ef = as_float(row.get("Ef_vac_eV")) if row.get("qualified") else math.nan
        row["delta_from_previous_eV"] = abs(ef - previous[group]) if group in previous and math.isfinite(ef) else math.nan
        if math.isfinite(ef):
            previous[group] = ef


def plot_scan(
    path: Path,
    rows: list[dict[str, object]],
    *,
    xkey: str,
    xlabel: str,
    title: str,
    ylabel: str = r"$E_f^{vac}$ (eV)",
) -> None:
    done = [row for row in rows if row.get("qualified") and math.isfinite(as_float(row.get("Ef_vac_eV")))
            and math.isfinite(as_float(row.get(xkey)))]
    if not done:
        # A rerun with no qualified points must not leave an older valid curve.
        path.unlink(missing_ok=True)
        return
    done.sort(key=lambda row: float(row[xkey]))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    groups = {}
    for row in done:
        groups.setdefault(series_key(row, xkey), []).append(row)
    for index, selected in enumerate(groups.values(), 1):
        direction = selected[0].get("pair_direction_verified", "unknown")
        label = direction if len(groups) == 1 else f"{direction}; series {index}"
        ax.plot([float(row[xkey]) for row in selected], [float(row["Ef_vac_eV"]) for row in selected],
                "-o" if direction != "unknown" or xkey != "pair_distance_A" else "o",
                linewidth=1.8, markersize=6, label=label)
    if len(groups) > 1:
        ax.legend(fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect DFTpy conventional fcc vacancy spacing/size scans."
    )
    parser.add_argument("--rootdir", required=True)
    parser.add_argument("--output", help="Separate output directory (default: rootdir/analysis_current)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rootdir = Path(args.rootdir).expanduser().resolve()
    if not rootdir.exists():
        raise FileNotFoundError(f"Missing rootdir: {rootdir}")
    output = Path(args.output).expanduser().resolve() if args.output else rootdir / "analysis_current"
    output.mkdir(parents=True, exist_ok=True)
    # These are this collector's managed artifacts. Clear stale families when
    # their inputs disappear, preserving all unrelated output files.
    for family in ('spacing', 'size', 'weight', 'pair'):
        for suffix in ('summary.csv', 'Ef.png'):
            (output / f'dftpy_conventional_{family}_{suffix}').unlink(missing_ok=True)

    spacing_rows = collect_scan(rootdir, "spacing_scan")
    size_rows = collect_scan(rootdir, "size_scan")
    weight_rows = collect_scan(rootdir, "weight_scan")
    pair_rows = collect_scan(rootdir, "pair_scan")

    if spacing_rows:
        # Larger spacing first in the table mirrors the usual convergence scan order.
        add_deltas(spacing_rows, "spacing_A", reverse=True)
        write_csv(output / "dftpy_conventional_spacing_summary.csv", spacing_rows)
        plot_scan(
            output / "dftpy_conventional_spacing_Ef.png",
            spacing_rows,
            xkey="spacing_A",
            xlabel="grid spacing (A)",
            title="DFTpy conventional fcc vacancy: spacing convergence",
        )

    if size_rows:
        add_deltas(size_rows, "N_pristine")
        write_csv(output / "dftpy_conventional_size_summary.csv", size_rows)
        plot_scan(
            output / "dftpy_conventional_size_Ef.png",
            size_rows,
            xkey="N_pristine",
            xlabel="pristine atom count",
            title="DFTpy conventional fcc vacancy: size/concentration convergence",
        )

    if weight_rows:
        add_deltas(weight_rows, "kedf_y")
        write_csv(output / "dftpy_conventional_weight_summary.csv", weight_rows)
        plot_scan(
            output / "dftpy_conventional_weight_Ef.png",
            weight_rows,
            xkey="kedf_y",
            xlabel="vW ratio y in TFvW",
            title="DFTpy conventional fcc vacancy: TF/vW weight scan",
        )

    if pair_rows:
        add_deltas(pair_rows, "pair_distance_A")
        write_csv(output / "dftpy_conventional_pair_summary.csv", pair_rows)
        directions = sorted(
            {
                str(row["pair_direction_verified"])
                for row in pair_rows
                if row.get("qualified")
            }
        )
        direction_suffix = f"fixed-direction scan along {directions[0]}" if len(directions) == 1 and directions[0] != "unknown" else "scan separated by verified direction"
        plot_scan(
            output / "dftpy_conventional_pair_Ef.png",
            pair_rows,
            xkey="pair_distance_A",
            xlabel="initial minimum-image vacancy-pair distance r (A)",
            ylabel=r"$E_f^{2vac}$ (eV)",
            title=f"DFTpy divacancy: {direction_suffix}",
        )

    all_rows = spacing_rows + size_rows + weight_rows + pair_rows
    write_csv(output / "dftpy_conventional_all_summary.csv", all_rows)

    print("============================================================")
    print("DFTpy conventional vacancy collection completed")
    print("============================================================")
    print(f"Root: {rootdir}")
    print(f"Spacing rows: {len(spacing_rows)}")
    print(f"Size rows   : {len(size_rows)}")
    print(f"Weight rows : {len(weight_rows)}")
    print(f"Pair rows   : {len(pair_rows)}")
    print(f"All summary : {output / 'dftpy_conventional_all_summary.csv'}")
    for status in ("missing", "failed", "unconverged", "qualified"):
        print(f"{status}: {sum(row.get('status') == status for row in all_rows)}")
    print("Qualified = numerical/evidence checks passed; thesis acceptance and finite-size convergence are not inferred.")


if __name__ == "__main__":
    main()
