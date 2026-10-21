"""Audit the complete eight-case 20260924 OF-DFT campaign, then make tables.

Read-only with respect to production data. No partial scientific report is
written: all eight attempts must pass first. Output directories must be new.
Uses NumPy/ASE only, and never attaches a live calculator or launches DFTpy.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from ase.geometry import find_mic
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.units import Bohr


CASE_IDS = ("0V", "1V", "2V_1NN_D110", "2V_2NN_D100", "2V_3NN_D211",
            "2V_4NN_D110", "2V_5NN_D310", "2V_6NN_D111")
N_PARENT = 108
A_REF = 3.9545804060131293
CELL = np.eye(3) * (3 * A_REF)
GRID = [78, 78, 78]
FMAX = 0.001
# Preserve the two actual conversion constants in the frozen runner/runtime.
# ASE 3.29.0 and DFTpy 2.1.2 ENERGY_CONV use the first value; the runner's
# energy_components() uses the second. No energy or geometry is recomputed.
RAW_ASE_HA_TO_EV = 27.211386024367243
COMMON_HA_TO_EV = 27.211386245988
EV_A3_TO_GPA = 160.21766208
COMMON_INPUT_HASHES = {
    "geometry.json": "aa5b79d49467c6aa36090eb5f423849c1af3f5ed42188069bb708b001bc67870",
    "al.lda.recpot": "c2fd4300aa39aa2466149a315e1c7cd87aac6615032db5ded8f237e9c40df803",
    "runner.py": "c796adaecdb89b46fd05e1098629f7b3a6256c6621f8494f1887e38130bff6ac",
    "dft_engine.py": "1083ce291fc449c6053e856f8cef0bc45d44a0c5d9d4d19dcd2af0111644f371",
}
TERMS = ("KEDF-TF", "KEDF-VW", "XC", "HARTREE", "PSEUDO", "II")
REQUIRED_ARTIFACTS = {
    "config.json", "provenance.json", "initial.extxyz", "initial.traj",
    "final.extxyz", "final.traj", "final.vasp", "final_forces_eV_A.npy",
    "final_density.npz", "electronic_evaluations.jsonl", "ionic_evaluations.jsonl",
    "relax.traj", "relax.log", "dftpy_full_stdout.log",
    *("inputs/" + name for name in (*COMMON_INPUT_HASHES, "structure.extxyz", "POSCAR")),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name, atol=1e-9):
    require(np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)) and
            np.allclose(actual, expected, rtol=0, atol=atol), name + " mismatch")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def safe_child(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root) and path != root, "Artifact path escapes attempt")
    return path


def verify_artifacts(attempt, result):
    hashes = result["artifact_sha256"]
    require(REQUIRED_ARTIFACTS <= set(hashes), "Required audited artifacts missing")
    for relative, expected in hashes.items():
        path = safe_child(attempt, relative)
        require(path.is_file() and sha256(path) == expected,
                f"Artifact hash mismatch: {path}")
    return hashes


def check_atoms(atoms, n_atoms, label):
    require(len(atoms) == n_atoms and set(atoms.get_chemical_symbols()) == {"Al"},
            label + " atom inventory mismatch")
    require(np.all(atoms.pbc) and not atoms.constraints, label + " boundary/constraint mismatch")
    close(atoms.cell.array, CELL, label + " cell", 1e-10)
    require(np.all(np.isfinite(atoms.positions)), label + " positions nonfinite")


def normalize_energy(raw_ase_eV, components, last_energy_Ha):
    """Require independent records to agree before applying the unit correction."""
    converted = raw_ase_eV / RAW_ASE_HA_TO_EV * COMMON_HA_TO_EV
    close(converted, components["TOTAL"], "ASE/component energy conversion", 1e-8)
    close(last_energy_Ha * COMMON_HA_TO_EV, components["TOTAL"],
          "Last electronic/component energy", 1e-8)
    close(sum(components[key] for key in TERMS), components["TOTAL"],
          "Non-duplicated component sum", 1e-8)
    close(components["KEDF-TF"] + components["KEDF-VW"], components["KEDF"],
          "Weighted TF+vW kinetic energy", 1e-9)
    return float(components["TOTAL"])


def displacements(initial_positions, final_positions, cell):
    """Match persistent atom IDs before calling; these vectors describe atoms."""
    raw, _ = find_mic(np.asarray(final_positions) - initial_positions, cell, pbc=True)
    mean = raw.mean(axis=0)  # Identical Al masses: arithmetic mean is COM shift.
    detrended = raw - mean
    return raw, detrended, mean


def audit_case(root, case_id, attempt_name):
    attempt = safe_child(root, f"{case_id}/{attempt_name}")
    result_path = attempt / "result.json"
    result = load_json(result_path)
    require(result.get("status") == "converged" and result.get("accepted") is True,
            case_id + " is not an accepted converged result")
    require(result["case"] == case_id and result["attempt"] == attempt_name,
            "Result identity mismatch")
    hashes = verify_artifacts(attempt, result)
    provenance = load_json(attempt / "provenance.json")
    config = load_json(attempt / "config.json")
    geometry = load_json(attempt / "inputs/geometry.json")
    case = geometry["cases"][case_id]
    q = 0 if case_id == "0V" else 1 if case_id == "1V" else 2
    n_atoms = N_PARENT - q
    require(result["n_atoms"] == provenance["n_atoms"] == case["n_atoms"] == n_atoms,
            "Atom count mismatch")
    require(provenance["case"] == case_id and case["q"] == case["vacancy_count"] == q,
            "Vacancy count/identity mismatch")
    require(geometry["case_order"] == list(CASE_IDS) and
            geometry["n_parent_sites"] == N_PARENT and provenance["geometry_case"] == case,
            "Campaign geometry provenance mismatch")
    for name, expected in COMMON_INPUT_HASHES.items():
        require(provenance["input_sha256"][name] == expected == hashes["inputs/" + name],
                "Frozen common input mismatch: " + name)
    for key, name in (("extxyz", "structure.extxyz"), ("poscar", "POSCAR")):
        require(case["structure_sha256"][key] == provenance["input_sha256"][name] ==
                hashes["inputs/" + name], "Frozen case structure mismatch: " + name)
    require(provenance["config_sha256"] == hashes["config.json"] and
            provenance["resolved_config"] == config, "Configuration provenance mismatch")
    require(provenance["packages"]["dftpy"] == "2.1.2" and
            provenance["packages"]["ase"] == "3.29.0", "Unexpected scientific runtime")
    require(config["GRID"]["nr"] == GRID and config["KEDF"]["kedf"] == "TFVW" and
            config["KEDF"]["x"] == 0.9 and config["KEDF"]["y"] == 0.1 and
            config["EXC"]["libxc"] == ["lda_x", "lda_c_pz"] and
            config["OPT"]["econv"] == 1e-9 and config["OPT"]["ncheck"] == 3 and
            config["DENSITY"]["densityini"] == "heg", "Scientific configuration mismatch")
    require(provenance["protocol"]["ionic"]["fmax_eV_A"] == FMAX,
            "Force tolerance mismatch")

    ids = case["remaining_parent_ids"]
    removed = case["removed_parent_ids"]
    require(len(set(removed)) == q and ids == [i for i in range(N_PARENT) if i not in removed],
            "Parent-site deletion mapping mismatch")
    initial_xyz = read(attempt / "initial.extxyz")
    final_xyz = read(attempt / "final.extxyz")
    initial = read(attempt / "initial.traj")
    final = read(attempt / "final.traj")
    for label, atoms in (("initial XYZ", initial_xyz), ("final XYZ", final_xyz),
                         ("initial trajectory", initial), ("final trajectory", final)):
        check_atoms(atoms, n_atoms, label)
    for atoms in (initial_xyz, final_xyz):
        require(atoms.arrays["parent_site_id"].tolist() == ids, "Persistent parent ID mismatch")
    close(initial.positions, initial_xyz.positions, "Initial representation", 1e-8)
    close(final.positions, final_xyz.positions, "Final representation", 1e-8)
    close(initial.positions, case["positions_A"], "Initial ideal geometry", 1e-8)
    parent_positions = np.asarray(geometry["cases"]["0V"]["positions_A"])
    close(initial.positions, parent_positions[ids], "Matched parent-site positions", 1e-8)
    close(result["cell_A"], CELL, "Result cell", 1e-10)
    close(result["volume_A3"], np.linalg.det(CELL), "Result volume", 1e-8)

    forces = np.load(attempt / "final_forces_eV_A.npy", allow_pickle=False)
    require(forces.shape == (n_atoms, 3), "Force shape mismatch")
    close(forces, final.get_forces(), "Final force artifact", 1e-12)
    fmax = float(np.linalg.norm(forces, axis=1).max())
    require(math.isfinite(fmax) and fmax <= FMAX and result["ionic_converged"] is True,
            "Final ionic force gate failed")
    close(fmax, result["final_fmax_eV_A"], "Final force norm", 1e-12)
    close(final.get_potential_energy(), result["energy_eV"], "Final trajectory energy", 1e-9)
    close(final.get_stress(voigt=False), result["stress_eV_A3"], "Final stress", 1e-12)
    close(np.asarray(result["stress_eV_A3"]) * EV_A3_TO_GPA, result["stress_GPa"],
          "Stress unit conversion", 1e-12)

    electronic = load_jsonl(attempt / "electronic_evaluations.jsonl")
    require(len(electronic) == result["electronic_calls"] > 0 and
            result["all_electronic_calls_converged"] is True, "Electronic gate record missing")
    for index, event in enumerate(electronic, 1):
        require(event["evaluation"] == index and event["accepted"] is True and
                event["converged_flag"] == 0 and event["grid_shape"] == GRID and
                math.isfinite(event["energy_Ha"]), "Electronic call rejected or corrupt")
        close(event["density_integral_electrons"], 3 * n_atoms, "Electronic electron count", 1e-6)
        close(event["effective_econv_Ha_total"], n_atoms * 1e-9,
              "Electronic total energy threshold", 1e-15)
    ionic = load_jsonl(attempt / "ionic_evaluations.jsonl")
    require([e["ionic_step"] for e in ionic] == list(range(result["ionic_steps"] + 1)),
            "Ionic step sequence incomplete")
    close(ionic[-1]["energy_eV"], result["energy_eV"], "Final ionic energy", 1e-9)
    close(ionic[-1]["fmax_eV_A"], fmax, "Final ionic force", 1e-12)
    with Trajectory(attempt / "relax.traj") as trajectory:
        require(len(trajectory) == len(ionic), "Ionic trajectory/log length mismatch")
        for frame, event in zip(trajectory, ionic):
            check_atoms(frame, n_atoms, "Ionic frame")
            close(frame.get_potential_energy(), event["energy_eV"], "Ionic frame energy", 1e-9)
            close(np.linalg.norm(frame.get_forces(), axis=1).max(), event["fmax_eV_A"],
                  "Ionic frame force", 1e-12)
        close(trajectory[-1].positions, final.positions, "Final ionic geometry", 1e-12)
    with np.load(attempt / "final_density.npz", allow_pickle=False) as saved:
        rho = saved["density_electrons_bohr3"]
        require(saved["grid"].tolist() == GRID and rho.shape == tuple(GRID) and
                np.all(np.isfinite(rho)) and float(rho.min()) >= -1e-12,
                "Final density grid/value mismatch")
        close(saved["lattice_bohr"] * Bohr, CELL, "Final density lattice", 1e-8)
        electrons = float(rho.sum() * abs(np.linalg.det(saved["lattice_bohr"])) / rho.size)
        close(electrons, 3 * n_atoms, "Integrated final density", 1e-6)
        close(electrons, result["density_integral_electrons"], "Final density provenance", 1e-6)

    terms = result["energy_components_eV"]
    energy = normalize_energy(result["energy_eV"], terms, electronic[-1]["energy_Ha"])
    close(terms["KEDF"], result["kinetic_energy_eV"], "Kinetic scalar", 1e-9)
    close(terms["KEDF-TF"], result["tf_energy_eV"], "Weighted TF scalar", 1e-9)
    close(terms["KEDF-VW"], result["vw_energy_eV"], "Weighted vW scalar", 1e-9)
    raw, detrended, mean = displacements(initial.positions, final.positions, CELL)
    raw_norm = np.linalg.norm(raw, axis=1)
    detrended_norm = np.linalg.norm(detrended, axis=1)
    separation = None
    if q == 2:
        _, distance = find_mic(parent_positions[removed[1]] - parent_positions[removed[0]], CELL)
        separation = float(distance)
    for key in ("elapsed_s", "peak_rss_MiB"):
        require(math.isfinite(result[key]) and result[key] > 0, key + " invalid")
    utc_elapsed = (datetime.fromisoformat(result["finished_utc"]) -
                   datetime.fromisoformat(result["started_utc"])).total_seconds()
    row = {
        "case": case_id, "q": q, "n_atoms": n_atoms,
        "initial_pair_MIC_A": separation, "vacancy_site_fraction": q / N_PARENT,
        "nominal_missing_sites_per_A3": q / result["volume_A3"],
        "raw_energy_eV_ASE": result["energy_eV"], "raw_components_TOTAL_eV": terms["TOTAL"],
        "energy_eV_common": energy, "energy_unit_adjustment_eV": energy - result["energy_eV"],
        "Ts_eV": terms["KEDF"], "weighted_TF_eV": terms["KEDF-TF"],
        "weighted_vW_eV": terms["KEDF-VW"],
        "fmax_eV_A": fmax, "ionic_steps": result["ionic_steps"],
        "electronic_calls": len(electronic), "elapsed_monotonic_s": result["elapsed_s"],
        "elapsed_UTC_s": utc_elapsed, "electronic_elapsed_sum_s": sum(e["elapsed_s"] for e in electronic),
        "peak_rss_MiB": result["peak_rss_MiB"],
        "u_raw_RMS_A": float(np.sqrt(np.mean(raw_norm**2))), "u_raw_max_A": float(raw_norm.max()),
        "u_detrended_RMS_A": float(np.sqrt(np.mean(detrended_norm**2))),
        "u_detrended_max_A": float(detrended_norm.max()),
        **{f"mean_translation_{axis}_A": float(mean[i]) for i, axis in enumerate("xyz")},
        **{f"stress_{axes}_GPa": float(np.asarray(result["stress_GPa"])[i, j])
           for axes, i, j in (("xx", 0, 0), ("yy", 1, 1), ("zz", 2, 2),
                              ("yz", 1, 2), ("xz", 0, 2), ("xy", 0, 1))},
    }
    atom_rows = []
    for i, parent_id in enumerate(ids):
        atom_rows.append({"case": case_id, "parent_site_id": parent_id,
                         **{f"{name}_{axis}_A": float(values[i, j])
                            for name, values in (("initial", initial.positions), ("final", final.positions),
                                                 ("u_raw", raw), ("u_detrended", detrended))
                            for j, axis in enumerate("xyz")},
                         "u_raw_norm_A": float(raw_norm[i]),
                         "u_detrended_norm_A": float(detrended_norm[i])})
    audit = {"case": case_id, "attempt_dir": str(attempt), "result_sha256": sha256(result_path),
             "verified_artifacts": len(hashes), "input_sha256": provenance["input_sha256"],
             "artifact_sha256": hashes, "protocol": provenance["protocol"],
             "packages": provenance["packages"], "threads": provenance["threads"],
             "energy_components_eV_common": terms, "removed_parent_ids": removed,
             "remaining_parent_ids": ids, "force_and_all_electronic_gates_passed": True}
    return row, atom_rows, audit


def add_energy_differences(rows):
    base, single = rows[0], rows[1]
    require(base["case"] == "0V" and single["case"] == "1V", "Reference order mismatch")
    for row in rows:
        for field, label in (("energy_eV_common", "E"), ("Ts_eV", "Ts"),
                             ("weighted_TF_eV", "weighted_TF"), ("weighted_vW_eV", "weighted_vW")):
            row[label + "_formation_eV"] = row[field] - (N_PARENT - row["q"]) / N_PARENT * base[field]
            row[label + "_binding_eV"] = (2 * single[field] - row[field] - base[field]
                                          if row["q"] == 2 else None)


def csv_write(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_markdown(rows):
    lines = ["# 12 Å bulk Al：DFTpy 八案例固定晶胞鬆弛結果", "",
             "八個案例均通過輸入與產物 SHA-256、電子收斂旗標、密度積分、固定晶胞、原子 ID、軌跡與最終力驗證。",
             "這是 108 個理想 FCC 晶格位置的有限週期晶胞結果；尚未建立尺寸、網格、電子閾值收斂或與 QE 的一致性。", "",
             "共用 a_ref = 3.9545804060131293 Å；晶胞邊長 11.863741218039388 Å；78³ 網格；LDA/PZ；"
             "TFvW λ = 0.9、μ = 0.1；al.lda.recpot；全原子自由鬆弛，晶胞固定。", "",
             "| 案例 | 初始空缺位點 MIC 距離 (Å) | E_f (eV) | E_b (eV) | ΔT_s^f (eV) | ΔT_s^b (eV) | 最大原子位移 (Å) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda value: "—" if value is None else f"{value:.6f}"
    for row in rows:
        lines.append("| " + row["case"] + " | " + " | ".join(fmt(row[key]) for key in (
            "initial_pair_MIC_A", "E_formation_eV", "E_binding_eV", "Ts_formation_eV", "Ts_binding_eV",
            "u_detrended_max_A")) + " |")
    lines += ["", "E_f(q) = E_q − (108−q)E_0/108；E_b(r) = 2E_1 − E_2(r) − E_0。"
              "正 E_b 表示相對兩個單空缺參考的結合有利；此參考仍受有限週期晶胞影響。"
              "T_s 使用同樣線性組合，為電子動能分量，不是離子溫度或完整凝聚驅動力。"
              "weighted_TF 與 weighted_vW 已包含 λ、μ，不再乘權重。", "",
              "表中最大位移是同一 parent_site_id 原子由起始位置到最終位置的最小影像向量，"
              "再扣除所有剩餘 Al 原子的平均平移（同質量下等於 COM 平移）。"
              "displacements.csv 同時保留原始與去平移向量；初始空缺位點只是刪除位置標籤，"
              "不將空缺當作被追蹤的物理粒子，也不據此推定遷移路徑或凝聚速率。", "",
              "| 案例 | 最終 fmax (eV/Å) | 離子步 | 電子次數 | monotonic 時間 (s) | UTC 跨度 (s) | 峰值 RSS (MiB) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['case']} | {row['fmax_eV_A']:.7f} | {row['ionic_steps']} | "
                     f"{row['electronic_calls']} | {row['elapsed_monotonic_s']:.2f} | "
                     f"{row['elapsed_UTC_s']:.2f} | {row['peak_rss_MiB']:.1f} |")
    lines += ["", "force gate 為最大原子力向量範數 ≤ 0.001 eV/Å。電子每次 evaluation 都必須"
              " accepted=true、converged_flag=0、78³ 網格、3N 個價電子；旗標對應設定下的能量歷史判準，"
              "不代表已另行完成電子殘差或能量差收斂研究。monotonic 時間保留 runner 原值；"
              "UTC 跨度另列，若主機休眠兩者可能不同，不能直接視作純計算效能比較。", "",
              "單位稽核：原始 result.energy_eV 與 ASE trajectory 使用 1 Ha = "
              "27.211386024367243 eV；runner energy_components_eV 使用 27.211386245988 eV。"
              "本表將總能與各分量統一為後者，並與最後一次 electronic energy_Ha 交叉驗證。"
              "results.csv 保存 raw_energy_eV_ASE、raw_components_TOTAL_eV 與每案例換算差；未重算電子態。", "",
              "results.csv 另含原始應力、名義空缺位點分率 q/108 與位點缺少數密度 q/V；"
              "它們與電子密度 n(r) 是不同量。應力仍依原 ASE 換算保留；"
              "彈性偶極或鬆弛體積分析需另確認符號、扣除共用晶胞 0V 殘餘應力及彈性常數。", "",
              "完整來源、公式、單位慣例與檔案雜湊見 audit.json；總能各分量保留於該檔。", ""]
    return "\n".join(lines)


def analyze(root, attempt_name, output_dir=None):
    root = Path(root).resolve()
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", attempt_name) and ".." not in attempt_name,
            "Unsafe attempt name")
    missing = [case for case in CASE_IDS if not (root / case / attempt_name / "result.json").is_file()]
    require(not missing, "Campaign incomplete; no report written: " + ", ".join(missing))
    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        require(not output_dir.exists(), "Output directory already exists; refusing overwrite")
        require(not output_dir.is_relative_to(root), "Analysis output must be outside production root")
    rows, atoms, audits = [], [], []
    for case in CASE_IDS:
        row, atom_rows, audit = audit_case(root, case, attempt_name)
        rows.append(row)
        atoms.extend(atom_rows)
        audits.append(audit)
    add_energy_differences(rows)
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "all_eight_cases_audited",
        "analysis_script_sha256": sha256(__file__), "run_root": str(root), "attempt": attempt_name,
        "case_order": list(CASE_IDS), "cases": audits,
        "energy_units": {"raw_ASE_eV_per_Ha": RAW_ASE_HA_TO_EV,
                         "common_eV_per_Ha": COMMON_HA_TO_EV,
                         "source": "Frozen runner constant and DFTpy 2.1.2 / ASE 3.29.0 runtime constants",
                         "E_common": "raw_energy_eV_ASE / raw_ASE_eV_per_Ha * common_eV_per_Ha; verified against components TOTAL and final electronic energy_Ha"},
        "definitions": {"formation": "X_q - (108-q)/108 * X_0",
                        "binding": "2*X_1 - X_2(r) - X_0; positive E binding is favorable",
                        "X": "total E, electronic Ts, weighted TF or weighted vW component",
                        "displacement": "MIC(final-initial) matched by immutable parent_site_id",
                        "detrended": "raw atom displacement minus its mean over remaining Al atoms",
                        "vacancy_labels": "initial deleted parent sites; no vacancy particle trajectory inferred"},
        "validation": "All artifact/input hashes; electronic flags and electron counts; fixed cell and atom IDs; full ionic trajectory energies/forces; force norm; density integral; energy/component unit reconciliation",
        "scientific_scope": "finite 108-site fixed-cell results; no finite-size/grid/electronic-tolerance or KS-DFT convergence claim",
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=False)
        csv_write(output_dir / "results.csv", rows)
        csv_write(output_dir / "displacements.csv", atoms)
        (output_dir / "displacements_by_case").mkdir()
        for case_id in CASE_IDS:
            csv_write(output_dir / "displacements_by_case" / (case_id + ".csv"),
                      [row for row in atoms if row["case"] == case_id])
        (output_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2,
                                                         allow_nan=False) + "\n", encoding="utf-8")
        (output_dir / "summary_zh.md").write_text(report_markdown(rows), encoding="utf-8")
        (output_dir / "output_sha256.json").write_text(json.dumps(
            {str(p.relative_to(output_dir)): sha256(p)
             for p in sorted(output_dir.rglob("*")) if p.is_file()}, indent=2) + "\n",
            encoding="utf-8")
    return rows, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--attempt", default="attempt_001")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    rows, audit = analyze(args.run_root, args.attempt, args.output_dir)
    print(json.dumps({"status": audit["status"], "cases": len(rows),
                      "output_dir": str(args.output_dir) if args.output_dir else None}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as exc:
        print("[REJECTED] " + str(exc), file=sys.stderr)
        raise SystemExit(2)
