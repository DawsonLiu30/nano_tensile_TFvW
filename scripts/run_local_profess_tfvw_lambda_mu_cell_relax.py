from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import subprocess
from pathlib import Path

import numpy as np


ENERGY_PATTERNS = {
    "tf_energy_eV": re.compile(r"Thomas-Fermi Kin\. Energy\s*=\s*([-+0-9.Ee]+)"),
    "vw_energy_eV": re.compile(r"Von-Weizsacker Kin\. Ener\s*=\s*([-+0-9.Ee]+)"),
    "kinetic_energy_eV": re.compile(r"TOTAL KINETIC ENERGY\s*=\s*([-+0-9.Ee]+)"),
    "total_energy_eV": re.compile(r"TOTAL ENERGY\s*=\s*([-+0-9.Ee]+)"),
}
LAT_RE = re.compile(
    r"LAT\s+[123]\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)"
)
STRESS_RE = re.compile(
    r"ST_GPa\s+[123]\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)"
)


def windows_to_wsl(path: Path) -> str:
    resolved = str(path.resolve())
    drive = resolved[0].lower()
    return f"/mnt/{drive}{resolved[2:].replace(os.sep, '/')}"


def last_float(text: str, pattern: re.Pattern[str]) -> float:
    matches = pattern.findall(text)
    return float(matches[-1]) if matches else math.nan


def last_matrix(text: str, pattern: re.Pattern[str]) -> np.ndarray:
    matches = pattern.findall(text)
    if len(matches) < 3:
        return np.full((3, 3), math.nan)
    return np.asarray(matches[-3:], dtype=float)


def run_case(
    rootdir: Path, setting: str, *, rerun_existing: bool
) -> dict[str, object]:
    case_dir = rootdir / "lambda_mu_scan" / setting
    manifest = json.loads((case_dir / "point_manifest.json").read_text(encoding="utf-8"))
    result_path = case_dir / "result.json"
    if result_path.exists() and not rerun_existing:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        return existing

    root_wsl = windows_to_wsl(rootdir)
    case_wsl = windows_to_wsl(case_dir)
    command = (
        f"cd '{case_wsl}' && chmod +x '{root_wsl}/PROFESS' && "
        f"'{root_wsl}/PROFESS' bulk > bulk.stdout 2> bulk.stderr"
    )
    completed = subprocess.run(
        ["wsl", "bash", "-lc", command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (case_dir / "runner.log").write_text(completed.stdout, encoding="utf-8")
    out_path = case_dir / "bulk.out"
    text = out_path.read_text(encoding="utf-8", errors="replace") if out_path.exists() else ""
    terminal_text = ""
    for name in ("bulk.stdout", "bulk.stderr"):
        path = case_dir / name
        if path.exists():
            terminal_text += path.read_text(encoding="utf-8", errors="replace")
    diagnostic_text = text + "\n" + terminal_text
    lattice = last_matrix(text, LAT_RE)
    stress = last_matrix(text, STRESS_RE)
    n_atoms = int(manifest["n_atoms"])
    total_energy = last_float(text, ENERGY_PATTERNS["total_energy_eV"])
    kinetic_energy = last_float(text, ENERGY_PATTERNS["kinetic_energy_eV"])
    tf_energy = last_float(text, ENERGY_PATTERNS["tf_energy_eV"])
    vw_energy = last_float(text, ENERGY_PATTERNS["vw_energy_eV"])
    volume = float(abs(np.linalg.det(lattice)))
    lattice_constant = volume ** (1.0 / 3.0)
    max_abs_stress = float(np.nanmax(np.abs(stress)))
    final_energy_block = "#                               FINAL ENERGIES" in text
    stopped_with_warning = "PROFESS Stop for the following reason:" in diagnostic_text
    calculation_completed = bool(
        completed.returncode == 0
        and "Run completed on:" in text
        and final_energy_block
        and not stopped_with_warning
        and np.isfinite(total_energy)
        and np.isfinite(kinetic_energy)
        and np.isfinite(volume)
    )
    cell_relax_converged = bool(
        calculation_completed
        and np.isfinite(max_abs_stress)
        and max_abs_stress <= float(manifest["max_stress_GPa"])
    )
    stable_fcc_equilibrium = bool(
        cell_relax_converged
        and float(manifest["min_solid_a0_A"])
        <= lattice_constant
        <= float(manifest["max_solid_a0_A"])
    )
    if stopped_with_warning:
        status = "STOPPED_WITH_WARNING"
    elif not calculation_completed:
        status = "INCOMPLETE"
    elif not cell_relax_converged:
        status = "UNCONVERGED_STRESS"
    elif not stable_fcc_equilibrium:
        status = "NO_STABLE_FCC_EQUILIBRIUM"
    else:
        status = "OK"
    result = {
        "setting": setting,
        "done": stable_fcc_equilibrium,
        "status": status,
        "calculation_completed": calculation_completed,
        "cell_relax_converged": cell_relax_converged,
        "stable_fcc_equilibrium": stable_fcc_equilibrium,
        "stopped_with_warning": stopped_with_warning,
        "returncode": completed.returncode,
        "relaxation_mode": manifest["relaxation_mode"],
        "lambda_tf": float(manifest["lambda_tf"]),
        "mu_vw": float(manifest["mu_vw"]),
        "initial_a0_A": float(manifest["initial_a0_A"]),
        "lattice_constant_A": lattice_constant,
        "final_volume_A3": volume,
        "final_lattice_vectors_A": lattice.tolist(),
        "final_stress_GPa": stress.tolist(),
        "final_max_abs_stress_GPa": max_abs_stress,
        "total_energy_eV": total_energy,
        "total_energy_eV_per_atom": total_energy / n_atoms,
        "kinetic_energy_eV": kinetic_energy,
        "kinetic_energy_eV_per_atom": kinetic_energy / n_atoms,
        "tf_energy_eV": tf_energy,
        "tf_energy_eV_per_atom": tf_energy / n_atoms,
        "vw_energy_eV": vw_energy,
        "vw_energy_eV_per_atom": vw_energy / n_atoms,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local PROFESS lambda-mu cell relaxations.")
    parser.add_argument("--rootdir", required=True)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--rerun-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rootdir = Path(args.rootdir).expanduser().resolve()
    settings = [
        line.strip()
        for line in (rootdir / "settings_lambda_mu_scan.txt").read_text().splitlines()
        if line.strip()
    ]
    stable = 0
    unstable = 0
    incomplete = 0
    exceptions = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {
            executor.submit(
                run_case,
                rootdir,
                setting,
                rerun_existing=args.rerun_existing,
            ): setting
            for setting in settings
        }
        for future in concurrent.futures.as_completed(futures):
            setting = futures[future]
            try:
                result = future.result()
                print(
                    f"{setting} status={result['status']} "
                    f"a0={result['lattice_constant_A']:.8f} "
                    f"E/N={result['total_energy_eV_per_atom']:.12f}"
                )
                if result["done"]:
                    stable += 1
                elif result.get("calculation_completed"):
                    unstable += 1
                else:
                    incomplete += 1
            except Exception as exc:
                exceptions += 1
                print(f"{setting} FAILED: {exc!r}")
    print("============================================================")
    print(f"Stable fcc equilibria : {stable}/{len(settings)}")
    print(f"Completed but unstable: {unstable}/{len(settings)}")
    print(f"Stopped/incomplete    : {incomplete}/{len(settings)}")
    print(f"Runner exceptions     : {exceptions}/{len(settings)}")
    if exceptions:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
