"""Read-only audit of downloaded vc-relax geometry, electronic logs and density.

Checks archived bytes and trajectories without launching any DFT calculation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from ase.io import read
from ase.io.trajectory import Trajectory
from scipy.spatial import cKDTree

CASE_IDS = ("0V", "1V", "2V_1NN_D110", "2V_2NN_D100", "2V_3NN_D211",
            "2V_4NN_D110", "2V_5NN_D310", "2V_6NN_D111")
FILES = ("result.json", "provenance.json", "initial.extxyz", "final.extxyz",
         "relax.traj", "ionic_evaluations.jsonl")
TERMS = ("KEDF-TF", "KEDF-VW", "XC", "HARTREE", "PSEUDO", "II")
EV_A3_TO_GPA = 160.21766208
SHIFTS = np.asarray(list(product((-1, 0, 1), repeat=3)))
ELECTRONIC_FILES = ("electronic_evaluations.jsonl", "dftpy_full_stdout.log",
                    "final_forces_eV_A.npy", "final_density.npz", "relax.log", "config.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label, atol=1e-9):
    require(np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)) and
            np.allclose(actual, expected, atol=atol, rtol=0), label + " mismatch")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def nonaffine_displacement(initial_fractional, final):
    """The initial fractional coordinates are affinely mapped to the final cell."""
    raw, _ = find_mic((final.get_scaled_positions(wrap=False) - initial_fractional) @ final.cell.array,
                      final.cell, pbc=True)
    translation = raw.mean(axis=0)
    return raw, raw - translation, translation


def nearest_site_diagnostic(atoms, parent_fractional, translation):
    """Voronoi occupancy of the affinely mapped parent lattice, with COM removed.

    Explicit 27-image sites suffice for these checked, near-cubic cells. This
    labels atomic site changes; it does not assign a trajectory to a vacancy.
    """
    cell = atoms.cell.array
    require(np.linalg.cond(cell) < 1.2 and np.max(np.abs(atoms.cell.angles() - 90)) < 5,
            "Nearest-site image diagnostic requires the near-cubic cell used here")
    fractional = ((atoms.positions - translation) @ np.linalg.inv(cell)) % 1
    tiled = ((parent_fractional[None, :, :] + SHIFTS[:, None, :]) @ cell).reshape(-1, 3)
    distances, indices = cKDTree(tiled).query(fractional @ cell, k=2)
    labels = indices[:, 0] % len(parent_fractional)
    return labels, distances[:, 0], distances[:, 1] - distances[:, 0]


def check_structure(atoms, n, label):
    require(len(atoms) == n and set(atoms.get_chemical_symbols()) == {"Al"} and
            np.all(atoms.pbc) and not atoms.constraints, label + " inventory/PBC mismatch")
    require(np.all(np.isfinite(atoms.positions)) and np.all(np.isfinite(atoms.cell.array)) and
            np.linalg.det(atoms.cell.array) > 0, label + " geometry invalid")


def audit_electronic(evidence_root, backup, folder, case_id, result, provenance, manifest, final):
    source = evidence_root / "dftpy" / case_id
    hashes = {}
    for name in ELECTRONIC_FILES:
        path = source / name
        entry = backup["files"][f"dftpy/{case_id}/{name}"]
        hashes[name] = sha256(path)
        require(hashes[name] == entry["sha256"] == result["artifact_sha256"][name] and
                path.stat().st_size == entry["bytes"], f"Electronic evidence bytes disagree: {case_id}/{name}")
    config = load(source / "config.json")
    require(config["GRID"]["nr"] == manifest["grid"] and config["EXC"]["xc"] == manifest["xc"] and
            config["KEDF"]["kedf"] == "TFVW" and config["KEDF"]["x"] == manifest["lambda_tf"] and
            config["KEDF"]["y"] == manifest["mu_vw"], "Archived DFTpy configuration differs")
    close(config["OPT"]["econv"], manifest["electronic"]["econv_Ha_per_atom"], "Configured econv", 1e-16)
    events = [json.loads(line) for line in (source / "electronic_evaluations.jsonl").read_text().splitlines() if line]
    ionic = [json.loads(line) for line in (folder / "ionic_evaluations.jsonl").read_text().splitlines() if line]
    require(len(events) == result["electronic_calls"] == len(ionic) and
            [e["evaluation"] for e in events] == list(range(1, len(events) + 1)),
            "Electronic/ionic event sequence mismatch")
    n = result["n_atoms"]
    for event, state in zip(events, ionic):
        require(event["accepted"] is True and event["converged_flag"] == 0 and
                event["grid_shape"] == manifest["grid"], "Electronic gate failed")
        close(event["effective_econv_Ha_total"], n * manifest["electronic"]["econv_Ha_per_atom"],
              "Effective electronic threshold", 1e-16)
        close(event["density_integral_electrons"], 3 * n, "Per-SCF electron count", 1e-6)
        close(event["cell_A"], state["cell_A"], "Electronic/ionic cell", 1e-10)
        close(event["energy_Ha"] * provenance["units"]["Hartree_eV"], state["energy_eV"],
              "Electronic/ionic energy", 1e-9)
    with np.load(source / "final_density.npz", allow_pickle=False) as density:
        rho = density["density_electrons_bohr3"]
        lattice = density["lattice_bohr"]
        require(density["grid"].tolist() == manifest["grid"] and list(rho.shape) == manifest["grid"] and
                np.isfinite(rho).all() and float(rho.min()) >= 0, "Final density grid/data invalid")
        close(lattice * provenance["units"]["Bohr_A"], final.cell.array, "Density/final cell", 1e-10)
        integral = float(rho.sum() * np.linalg.det(lattice) / rho.size)
        close(integral, 3 * n, "Independently integrated final density", 1e-6)
        close(integral, result["density_integral_electrons"], "Recorded/final density integral", 1e-8)
        density_range = [float(rho.min()), float(rho.max())]
    close(np.load(source / "final_forces_eV_A.npy", allow_pickle=False), final.get_forces(),
          "Independent force array/trajectory", 1e-12)
    stdout = (source / "dftpy_full_stdout.log").read_text()
    require("Traceback (most recent call last)" not in stdout, "Exception in archived DFTpy stdout")
    return {"verified_electronic_artifact_sha256": hashes, "electronic_events_rechecked": len(events),
            "independently_rechecked_electronic_event_log": True,
            "independently_reintegrated_density": True, "density_integral_electrons_recomputed": integral,
            "density_range_electrons_bohr3": density_range,
            "limitation": "SCF convergence uses the DFTpy energy-history flag; no separate residual convergence study was performed."}


def analyze_case(root, case_id, receipt, geometry, manifest, campaign, evidence_root, backup):
    folder = root / case_id
    hashes = {}
    for name in FILES:
        hashes[name] = sha256(folder / name)
        require(hashes[name] == receipt["files"][name]["sha256"], f"Download receipt mismatch: {case_id}/{name}")
    result, provenance = load(folder / "result.json"), load(folder / "provenance.json")
    for name in FILES[1:]:
        require(hashes[name] == result["artifact_sha256"][name], f"Runner artifact mismatch: {case_id}/{name}")
    require(result["case"] == provenance["case"] == case_id and
            result["status"] == "converged" and result["accepted"] is True and
            result["ionic_converged"] is True and result["all_electronic_calls_converged"] is True,
            case_id + " acceptance evidence missing")
    require(result["target_pressure_GPa"] == 0 and provenance["protocol"] == manifest,
            case_id + " pressure/protocol mismatch")
    case = geometry["cases"][case_id]
    require(provenance["geometry_case"] == case, "Frozen geometry case differs")
    q, n = case["vacancy_count"], case["n_atoms"]
    require(result["n_atoms"] == n == 108 - q and result["vacancy_count"] == q, "Atom count mismatch")
    require(provenance["input_sha256"]["geometry.json"] == sha256(campaign / "geometry.json") and
            provenance["input_sha256"]["dftpy_manifest.json"] == sha256(campaign / "dftpy_manifest.json"),
            "Manifest/input geometry bytes disagree")
    pseudo = campaign / manifest["pseudo"]["file"]
    require(provenance["input_sha256"][pseudo.name] == manifest["pseudo"]["sha256"] == sha256(pseudo),
            "Pseudopotential bytes disagree")
    for key, source_name in (("extxyz", "structure.extxyz"), ("poscar", "POSCAR")):
        require(sha256(campaign / case["structure_files"][key]) == case["structure_sha256"][key] ==
                provenance["input_sha256"][source_name], "Initial input structure bytes disagree")
    repo = Path(__file__).resolve().parents[1]
    for name, source in (("runner.py", repo / "scripts/run_bulk_vacancy_dftpy_vcrelax_20260925.py"),
                         ("dft_engine.py", repo / "app/dft_engine.py")):
        require(provenance["input_sha256"][name] == sha256(source) ==
                result["artifact_sha256"]["inputs/" + name], "Frozen executable source mismatch")

    units = provenance["units"]
    close(result["energy_conversion_eV_per_Ha"], units["Hartree_eV"], "Energy conversion", 1e-12)
    close(result["energy_Ha"] * units["Hartree_eV"], result["energy_eV"], "Hartree/eV energy", 1e-9)
    terms = result["energy_components_eV"]
    close(terms["TOTAL"], result["energy_eV"], "Component TOTAL energy", 1e-9)
    close(sum(terms[key] for key in TERMS), result["energy_eV"], "Non-duplicated energy components", 1e-8)
    close(terms["KEDF-TF"] + terms["KEDF-VW"], terms["KEDF"], "Weighted TF+vW", 1e-9)
    close(terms["KEDF"], result["kinetic_energy_eV"], "Kinetic scalar", 1e-9)
    close(result["density_integral_electrons"], 3 * n, "Recorded electron count", 1e-6)

    initial_xyz, final_xyz = read(folder / "initial.extxyz"), read(folder / "final.extxyz")
    ids, removed = case["remaining_parent_ids"], case["removed_parent_ids"]
    require(ids == [i for i in range(108) if i not in removed], "Parent mapping invalid")
    for label, atoms in (("initial", initial_xyz), ("final", final_xyz)):
        check_structure(atoms, n, label)
        require(atoms.arrays["parent_site_id"].tolist() == ids, "Persistent parent IDs mismatch")
    close(initial_xyz.positions, case["positions_A"], "Initial positions", 1e-8)
    close(initial_xyz.cell.array, geometry["cell_A"], "Initial cell", 1e-10)
    ionic = [json.loads(line) for line in (folder / "ionic_evaluations.jsonl").read_text().splitlines() if line]
    require([item["ionic_step"] for item in ionic] == list(range(result["ionic_steps"] + 1)),
            "Ionic sequence missing steps")
    require(result["electronic_calls"] >= len(ionic), "Fewer electronic evaluations than ionic states")
    parent_fractional = np.asarray(geometry["cases"]["0V"]["fractional"])
    per_step = []
    jumps = []
    with Trajectory(folder / "relax.traj") as trajectory:
        require(len(trajectory) == len(ionic), "Trajectory/log length mismatch")
        initial = trajectory[0]
        close(initial.positions, initial_xyz.positions, "Initial trajectory/XYZ", 1e-8)
        s0 = initial.get_scaled_positions(wrap=False)
        for step, (frame, event) in enumerate(zip(trajectory, ionic)):
            check_structure(frame, n, "Trajectory")
            close(frame.cell.array, event["cell_A"], "Recorded h(t)", 1e-10)
            close(frame.get_volume(), event["volume_A3"], "Recorded V(t)", 1e-8)
            close(frame.get_potential_energy(), event["energy_eV"], "Ionic energy", 1e-9)
            force_max = float(np.linalg.norm(frame.get_forces(), axis=1).max())
            close(force_max, event["fmax_eV_A"], "Ionic force", 1e-10)
            close(frame.get_stress(voigt=False), event["stress_eV_A3"], "Ionic stress", 1e-10)
            raw, detrended, translation = nonaffine_displacement(s0, frame)
            labels, site_distances, margin = nearest_site_diagnostic(frame, parent_fractional, translation)
            changed = [int(ids[i]) for i in np.flatnonzero(labels != ids)]
            empty = sorted(set(range(108)) - set(labels.tolist()))
            if changed or empty != sorted(removed):
                jumps.append({"ionic_step": step, "changed_parent_ids": changed, "empty_parent_sites": empty})
            per_step.append({"case": case_id, "ionic_step": step, "volume_A3": frame.get_volume(),
                             "nonaffine_detrended_max_A": float(np.linalg.norm(detrended, axis=1).max()),
                             "changed_nearest_parent_sites": len(changed),
                             "min_nearest_vs_second_site_margin_A": float(margin.min()),
                             "cell_A": frame.cell.array.tolist()})
        final = trajectory[-1]
    close(final.positions, final_xyz.positions, "Final trajectory/XYZ", 1e-8)
    close(final.cell.array, final_xyz.cell.array, "Final XYZ cell", 1e-10)
    close(final.cell.array, result["cell_A"], "Final result cell", 1e-10)
    close(final.get_potential_energy(), result["energy_eV"], "Final energy", 1e-9)
    close(final.get_volume(), result["volume_A3"], "Final volume", 1e-8)
    fmax = float(np.linalg.norm(final.get_forces(), axis=1).max())
    stress = final.get_stress(voigt=False) * units["eV_A3_to_GPa"]
    stress_max = float(np.abs(stress).max())
    close(fmax, result["fmax_eV_A"], "Final force norm", 1e-10)
    close(stress, result["stress_GPa"], "Final stress units", 1e-10)
    close(ionic[-1]["filter_fmax_eV_A"], result["filter_fmax_eV_A"], "Final filter force", 1e-10)
    require(fmax <= manifest["ionic"]["fmax_eV_A"] and
            result["filter_fmax_eV_A"] <= manifest["ionic"]["filter_fmax_eV_A"] and
            stress_max <= manifest["cell"]["stress_max_abs_GPa"], "Structural gate failed")
    raw, detrended, translation = nonaffine_displacement(s0, final)
    labels, site_distances, margin = nearest_site_diagnostic(final, parent_fractional, translation)
    pair = case.get("pair", {})
    row = {"case": case_id, "q": q, "n_atoms": n, "initial_pair_MIC_A": pair.get("distance_A"),
           "initial_shortest_image_multiplicity": pair.get("shortest_image_multiplicity"),
           "periodic_image_diagnostic": bool(pair.get("half_cell_image_degeneracy", False)),
           "energy_eV": result["energy_eV"], "Ts_eV": terms["KEDF"],
           "weighted_TF_eV": terms["KEDF-TF"], "weighted_vW_eV": terms["KEDF-VW"],
           "volume_A3": result["volume_A3"], "a_equiv_parent108_A": (4 * result["volume_A3"] / 108)**(1 / 3),
           "fmax_eV_A": fmax, "filter_fmax_eV_A": result["filter_fmax_eV_A"],
           "stress_max_abs_GPa": stress_max, "ionic_steps": result["ionic_steps"],
           "electronic_calls": result["electronic_calls"], "elapsed_s": result["elapsed_s"],
           "peak_rss_MiB": result["peak_rss_MiB"],
           "nonaffine_raw_max_A": float(np.linalg.norm(raw, axis=1).max()),
           "nonaffine_detrended_max_A": float(np.linalg.norm(detrended, axis=1).max()),
           "nonaffine_detrended_rms_A": float(np.sqrt(np.mean(np.sum(detrended**2, axis=1)))),
           "trajectory_nonaffine_detrended_max_A": max(s["nonaffine_detrended_max_A"] for s in per_step),
           "trajectory_steps_with_nearest_site_change": len(jumps),
           "final_nearest_site_changes": int(np.sum(labels != ids)),
           "final_min_site_assignment_margin_A": float(margin.min()),
           "vacancy_site_fraction": q / 108, "nominal_missing_sites_density_A3": q / result["volume_A3"],
           **{f"removed_translation_{axis}_A": float(translation[i]) for i, axis in enumerate("xyz")},
           **{f"stress_{axes}_GPa": float(stress[i, j]) for axes, i, j in
              (("xx", 0, 0), ("yy", 1, 1), ("zz", 2, 2), ("yz", 1, 2), ("xz", 0, 2), ("xy", 0, 1))}}
    atom_rows = []
    for i, parent_id in enumerate(ids):
        atom_rows.append({"case": case_id, "parent_site_id": parent_id,
                         "nearest_affinely_mapped_parent_site": int(labels[i]),
                         "nearest_site_distance_A": float(site_distances[i]),
                         **{f"{name}_{axis}_A": float(values[i, j])
                            for name, values in (("initial", initial.positions), ("final", final.positions),
                                                 ("nonaffine_raw", raw), ("nonaffine_detrended", detrended))
                            for j, axis in enumerate("xyz")}})
    evidence = {"case": case_id, "source": receipt["source"], "verified_downloaded_sha256": hashes,
                "input_sha256": provenance["input_sha256"], "packages": provenance["packages"],
                "units": units, "energy_components_eV": terms, "initial_cell_A": initial.cell.array.tolist(),
                "final_cell_A": final.cell.array.tolist(), "final_cell_angles_deg": final.cell.angles().tolist(),
                "removed_parent_ids": removed, "remaining_parent_ids": ids,
                "initial_pair_metadata": pair, "site_change_events": jumps,
                "all_electronic_calls_converged_recorded": result["all_electronic_calls_converged"],
                **audit_electronic(evidence_root, backup, folder, case_id, result, provenance, manifest, final)}
    return row, atom_rows, per_step, evidence


def add_differences(rows):
    zero, single = rows[:2]
    for row in rows:
        factor = (108 - row["q"]) / 108
        for field, label in (("energy_eV", "E"), ("Ts_eV", "Ts"),
                             ("weighted_TF_eV", "weighted_TF"), ("weighted_vW_eV", "weighted_vW")):
            row[label + "_formation_eV"] = row[field] - factor * zero[field]
            row[label + "_binding_eV"] = 2 * single[field] - row[field] - zero[field] if row["q"] == 2 else None
        row["relaxation_volume_A3"] = row["volume_A3"] - zero["volume_A3"]
        row["formation_volume_A3"] = row["volume_A3"] - factor * zero["volume_A3"]
        row["delta_a_equiv_from_0V_A"] = row["a_equiv_parent108_A"] - zero["a_equiv_parent108_A"]


def compare_fixed(rows, fixed_csv, new_hartree):
    with fixed_csv.open(encoding="utf-8-sig", newline="") as stream:
        old = {row["case"]: row for row in csv.DictReader(stream)}
    old_hartree = load(fixed_csv.parent / "audit.json")["energy_units"]["common_eV_per_Ha"]
    scale = new_hartree / old_hartree
    comparison = []
    for row in rows:
        prev = old[row["case"]]
        entry = {"case": row["case"], "fixed_initial_pair_MIC_A": prev["initial_pair_MIC_A"],
                 "vc_initial_pair_MIC_A": row["initial_pair_MIC_A"],
                 "fixed_a_ref_A": 3.9545804060131293, "vc_initial_a_ref_A": 4.039848,
                 "fixed_to_vc_energy_unit_scale": scale}
        for key in ("E_formation_eV", "E_binding_eV", "Ts_formation_eV", "Ts_binding_eV"):
            value = float(prev[key]) * scale if prev[key] else None
            entry["fixed_" + key] = value
            entry["vc_" + key] = row[key]
            entry["vc_minus_fixed_" + key] = row[key] - value if value is not None else None
        comparison.append(entry)
    return comparison


def make_plot(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pairs = rows[2:]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.linewidth": .9,
                         "savefig.dpi": 350, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    fig.subplots_adjust(left=.13, right=.97, top=.84, bottom=.24)
    regular = [r for r in pairs if not r["periodic_image_diagnostic"]]
    ax.axhline(0, color="#6b7280", linewidth=.9, linestyle="--", zorder=1)
    ax.scatter([r["initial_pair_MIC_A"] for r in regular],
               [1000 * r["E_binding_eV"] for r in regular], color="#175b88", s=65,
               edgecolors="white", linewidths=.7, zorder=3, label="Distinct initial pair configurations")
    diagnostic = [r for r in pairs if r["periodic_image_diagnostic"]]
    ax.scatter([r["initial_pair_MIC_A"] for r in diagnostic],
               [1000 * r["E_binding_eV"] for r in diagnostic], marker="D", s=75,
               facecolors="none", edgecolors="#b35c00", linewidths=1.6, zorder=4,
               label="5NN: two equidistant periodic images")
    offsets = [(9, 0), (7, 9), (-5, 13), (-10, -24), (-24, 12), (-5, -24)]
    for i, (row, offset) in enumerate(zip(pairs, offsets), 1):
        ax.annotate(f"{i}NN", (row["initial_pair_MIC_A"], row["E_binding_eV"] * 1000),
                    xytext=offset, textcoords="offset points", fontsize=10,
                    color="#b35c00" if row["periodic_image_diagnostic"] else "#173247")
    ax.set(xlabel="Initial removed-site minimum-image distance (Å)",
           ylabel=r"Binding energy $E_b$ (meV)", xlim=(2.55, 7.35), ylim=(-10, 63))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.18)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.text(.13, .94, "Al divacancy binding after full cell relaxation", fontsize=14, weight="bold")
    fig.text(.13, .88, "OF-DFT / local LDA  ·  TFvW λ = 0.9, μ = 0.1  ·  108 parent sites  ·  p = 0", fontsize=9.5)
    fig.text(.13, .105, r"$E_b=2E_{1V}-E_{2V}-E_{0V}$; positive means favorable binding within this finite periodic model.", fontsize=9)
    fig.text(.13, .06, "Independent relaxed states; distance is an initial geometry label, not a migration coordinate.", fontsize=9)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def summary(rows, comparison, evidence):
    zero, one = rows[:2]
    lines = ["# Bulk Al 變晶胞鬆弛：proposal 初步結果", "",
             f"DFTpy 八組皆有 accepted 結果。1V 形成能為 **{one['E_formation_eV']:.6f} eV**；"
             f"1NN 結合能為 **{rows[2]['E_binding_eV']*1000:.3f} meV**。完整晶體的體積等效晶格常數為 "
             f"**{zero['a_equiv_parent108_A']:.6f} Å**。", "",
             "保留原設定：LDA/local al.lda.recpot，TFvW λ=0.9、μ=0.1，固定 78³ 格點；"
             "108 個父晶格位置，三維 PBC，全原子／完整晶胞於零外壓鬆弛。共用理想起點 a_init=4.039848 Å、"
             "cell=12.119544 Å；此起點不是 DFTpy 的平衡晶格常數。", "",
             "| 案例 | 初始 MIC 距離 Å | E_f eV | E_b meV | ΔT_s^f eV | ΔT_s^b meV |",
             "|---|---:|---:|---:|---:|---:|"]
    number = lambda v, scale=1: "—" if v is None else f"{v*scale:.6f}"
    for row in rows:
        lines.append(f"| {row['case']} | {number(row['initial_pair_MIC_A'])} | {number(row['E_formation_eV'])} | "
                     f"{number(row['E_binding_eV'],1000)} | {number(row['Ts_formation_eV'])} | {number(row['Ts_binding_eV'],1000)} |")
    lines += ["", "E_f(q)=E_q−(108−q)E_0/108；E_b=2E_1−E_2−E_0。各能量均來自同一方法、"
              "同一零外壓協定，各自最終體積可以不同。T_s 使用相同線性組合；其 TF、vW 分量已包含權重，"
              "動能分量不能單獨替代總形成能或總結合能。", "",
              "| 案例 | V Å³ | a_eq,108 Å | ΔV_rel Å³ | V_form Å³ | 最大非仿射位移 Å |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['case']} | {row['volume_A3']:.6f} | {row['a_equiv_parent108_A']:.6f} | "
                     f"{row['relaxation_volume_A3']:.6f} | {row['formation_volume_A3']:.6f} | {row['nonaffine_detrended_max_A']:.6f} |")
    lines += ["", "a_eq,108=(4V/108)^(1/3) 以父晶格位置數定義。缺陷晶胞可略微非立方，"
              "因此這是體積等效尺度；完整 h(t) 和角度保留在 JSON。ΔV_rel=V_q−V_0；"
              "V_form=V_q−(108−q)V_0/108；兩者不可混稱。", "",
              "原子位移先用同一 parent_site_id 對應，計算 MIC[(s_final−s_initial)h_final]，"
              "再扣除剩餘 Al 原子的平均平移；已排除整體晶胞仿射變形。CSV 保留 raw／去平移向量及初終座標。"]
    jump_count = sum(row["trajectory_steps_with_nearest_site_change"] for row in rows)
    if jump_count == 0:
        lines += ["逐步軌跡的最近父晶格位置檢查未見原子跳至另一晶格位置；各步空的父晶格位置也與最初刪除位點一致。"
                  "這是近立方晶胞、去 COM 後的 27 週期影像最近格點診斷，支持此批為各起始缺陷構型的鬆弛終態。"]
    else:
        lines += [f"發現 {jump_count} 個軌跡步有最近父晶格位置變更，詳見 audit.json；需個別結構檢查。"]
    lines += ["**不能因 1NN 能量較低便聲稱已觀察到空缺凝聚動力學。** 最佳化步不是物理時間，"
              "凝聚路徑與能障需要另外的路徑計算。", "",
              "5NN [310] 起始構型位於半盒長影像退化位置，有兩個等距最近影像，圖中以橙色空心菱形標記。"
              "它保留作週期影像診斷，不當作孤立 pair potential 的單一距離點；6NN 也未被當作無限分離參考。"
              "4NN／6NN 的亞 meV 結合能不能在未確認整體數值與尺寸誤差前作確定正負判定。", "",
              "| 案例 | 原子 fmax eV/Å | filter fmax | max abs(σ_ij) GPa | 離子步 | 時間 s |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['case']} | {row['fmax_eV_A']:.8f} | {row['filter_fmax_eV_A']:.8f} | "
                     f"{row['stress_max_abs_GPa']:.6f} | {row['ionic_steps']} | {row['elapsed_s']:.1f} |")
    lines += ["", "全部滿足原子 fmax≤0.001 eV/Å、filter fmax≤0.001、完整應力張量 max|σ_ij|≤0.05 GPa；"
              "filter fmax 是包含晶胞的廣義力停止值，不能等同原子力。", "",
              "驗證範圍：96 個已下載 DFTpy 檔案逐一核對下載 manifest 與 runner artifact hash（result 自身由下載 manifest 核對）；"
              "核對本地原始 geometry／manifest／偽勢／runner／engine 雜湊、原子 ID、各步晶胞、能量、原子力與應力。"
              "總能、Ha→eV、分量加總與 T_s 交叉一致。481 次電子事件均通過 flag=0、78³ 格點、電子數與對應離子幀的能量／晶胞檢查；"
              "8 組最終密度已獨立重新積分，均等於 3N 電子，密度晶格也與最終原子晶胞一致。"
              "SCF 收斂依 DFTpy 的能量歷史判据，尚未另作電子殘差或更嚴門檻的收斂研究。", "",
              "與 9/24 固定晶胞比較僅使用各批已正確配對的形成／結合能，不交叉拼接 E_0、E_1、E_2；"
              "comparison_fixed_vs_vc.csv 同時保存兩套初始距離、邊界條件與換算常數差的校正。"
              f"單空缺形成能由 {comparison[1]['fixed_E_formation_eV']:.6f} 降至 {one['E_formation_eV']:.6f} eV；"
              f"1NN 結合能由 {comparison[2]['fixed_E_binding_eV']*1000:.3f} 增至 {rows[2]['E_binding_eV']*1000:.3f} meV。", "",
              "proposal 可使用此批作 OF-DFT 的能量與結構初步結果。QE 採 PBE/PAW，而 DFTpy 為 LDA/local，"
              "後續兩者差異同時包含 XC、偽勢、KEDF 和鬆弛結構差異，不能全部歸咎 TFvW。"
              "本報告專注 DFTpy，未加入 QE 數值。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--fixed-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replace-generated", action="store_true",
                        help="Refresh only an existing analysis directory whose complete output hashes still match")
    args = parser.parse_args()
    if args.output_dir.exists():
        require(args.replace_generated, "Output directory exists; choose a new directory or --replace-generated")
        previous = load(args.output_dir / "output_sha256.json")
        inventory = {str(p.relative_to(args.output_dir)) for p in args.output_dir.rglob("*")
                     if p.is_file() and p.name != "output_sha256.json"}
        require(set(previous) == inventory and all(sha256(args.output_dir / name) == digest
                for name, digest in previous.items()), "Existing analysis was modified or has unrelated files")
    receipt = load(args.data_root / "download_receipt.json")
    backup = load(args.evidence_root / "BACKUP_MANIFEST.json")
    geometry, manifest = load(args.campaign / "geometry.json"), load(args.campaign / "dftpy_manifest.json")
    require(geometry["case_order"] == list(CASE_IDS) and set(receipt["cases"]) == set(CASE_IDS), "Case inventory mismatch")
    rows, atoms, steps, evidence = [], [], [], []
    for case in CASE_IDS:
        row, atom_rows, case_steps, audit = analyze_case(args.data_root, case, receipt["cases"][case], geometry,
                                                       manifest, args.campaign, args.evidence_root, backup)
        rows.append(row)
        atoms.extend(atom_rows)
        steps.extend(case_steps)
        evidence.append(audit)
    require(len({entry["units"]["Hartree_eV"] for entry in evidence}) == 1, "Mixed energy units")
    add_differences(rows)
    comparison = compare_fixed(rows, args.fixed_csv, evidence[0]["units"]["Hartree_eV"])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "downloaded_electronic_density_and_structure_audit_passed",
              "source_data_root": str(args.data_root), "download_receipt_sha256": sha256(args.data_root / "download_receipt.json"),
              "analysis_script_sha256": sha256(__file__), "verified_downloaded_files": 96,
              "source_evidence_root": str(args.evidence_root),
              "backup_manifest_sha256": sha256(args.evidence_root / "BACKUP_MANIFEST.json"),
              "protocol": manifest, "results": rows, "case_evidence": evidence,
              "trajectory_diagnostics": steps, "fixed_vs_vc_comparison": comparison,
              "fixed_csv_sha256": sha256(args.fixed_csv),
              "formulas": {"Ef": "Eq-(108-q)/108*E0", "Eb": "2E1-E2-E0",
                           "a_equiv": "(4V/108)^(1/3)", "relaxation_volume": "Vq-V0",
                           "formation_volume": "Vq-(108-q)/108*V0",
                           "nonaffine": "MIC[(s_final-s_initial) h_final], then subtract mean over remaining atoms"}}
    args.output_dir.mkdir(parents=True, exist_ok=args.replace_generated)
    write_csv(args.output_dir / "results.csv", rows)
    write_csv(args.output_dir / "displacements.csv", atoms)
    write_csv(args.output_dir / "comparison_fixed_vs_vc.csv", comparison)
    by_case = args.output_dir / "displacements_by_case"
    by_case.mkdir(exist_ok=args.replace_generated)
    for case in CASE_IDS:
        write_csv(by_case / (case + ".csv"), [row for row in atoms if row["case"] == case])
    (args.output_dir / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "summary_zh.md").write_text(summary(rows, comparison, evidence), encoding="utf-8")
    make_plot(args.output_dir / "binding_energy_vs_initial_separation.png", rows)
    (args.output_dir / "output_sha256.json").write_text(json.dumps(
        {str(path.relative_to(args.output_dir)): sha256(path) for path in sorted(args.output_dir.rglob("*"))
         if path.is_file() and path.name != "output_sha256.json"}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "cases": len(rows), "atom_displacements": len(atoms),
                      "trajectory_frames": len(steps), "steps_with_site_changes": sum(row["trajectory_steps_with_nearest_site_change"] for row in rows),
                      "one_vacancy_formation_eV": rows[1]["E_formation_eV"], "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
