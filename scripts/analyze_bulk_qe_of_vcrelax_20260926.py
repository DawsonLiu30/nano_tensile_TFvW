"""Analyze archived QE attempts and accepted OF-DFT vc-relax results, read-only.

No QE/DFTpy calculation is launched. Unusable states are diagnostics only and
never enter reported formation or binding combinations.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
from ase import Atoms
from ase.io import read

import run_bulk_vacancy_qe_vcrelax_20260925 as qe
from analyze_bulk_vacancy_vcrelax_20260925 import (
    CASE_IDS, close, load, nearest_site_diagnostic, nonaffine_displacement, require, sha256,
)


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def frame_atoms(frame):
    return Atoms(frame["symbols"], positions=frame["positions_A"], cell=frame["cell_A"], pbc=True)


def smearing_info(output):
    """Keep QE's printed smearing free energy separate from the E correction."""
    final = re.split(r"Final\s+scf\s+calculation\s+at\s+the\s+relaxed\s+structure", output, flags=re.I)
    if len(final) < 2:
        return None
    values = re.findall(r"smearing contrib\.\s*\(-TS\)\s*=\s*(" + qe.NUMBER + r")\s*Ry", final[-1])
    return qe.number(values[-1]) * qe.RY_TO_EV if values else None


def blank_row(method, case_id, geometry):
    case = geometry["cases"][case_id]
    return {"method": method, "case": case_id, "q": case["vacancy_count"], "n_atoms": case["n_atoms"],
            "usable": False, "status": "missing", "initial_pair_MIC_A": case.get("pair", {}).get("distance_A"),
            "energy_eV": None, "Ts_eV": None, "smearing_minus_TS_eV": None,
            "volume_A3": None, "a_equiv_parent108_A": None, "fmax_eV_A": None,
            "stress_max_abs_GPa": None, "nonaffine_raw_max_A": None, "nonaffine_detrended_max_A": None,
            "nonaffine_detrended_rms_A": None, "final_nearest_site_changes": None,
            "trajectory_frames_with_nearest_site_change": None, "elapsed_s": None,
            "E_formation_eV": None, "E_binding_eV": None, "Ts_formation_eV": None, "Ts_binding_eV": None,
            "relaxation_volume_A3": None, "formation_volume_A3": None}


def analyze_qe(attempt, campaign, geometry):
    record = load(attempt / "attempt.json")
    case_id = record["case_id"]
    case = geometry["cases"][case_id]
    raw_result = load(attempt / "result.json")
    inspected = qe.inspect_attempt(attempt, campaign)
    require(inspected["status"] != "provenance_failed", f"{case_id}: {inspected.get('reasons')}")
    require(raw_result["usable"] == inspected["usable"], case_id + " archived/current usability disagreement")
    output = (attempt / "pw.out").read_text(encoding="utf-8", errors="replace")
    parsed = qe.parse_input((attempt / "pw.in").read_text(encoding="utf-8"))
    close(parsed["positions_A"], case["positions_A"], "QE input/campaign parent ordering", 1e-8)
    close(parsed["cell_A"], geometry["cell_A"], "QE initial cell", 1e-9)
    initial = read(attempt / "source/structure.extxyz")
    ids = initial.arrays["parent_site_id"].tolist()
    require(ids == case["remaining_parent_ids"], "QE parent site mapping mismatch")
    s0 = initial.get_scaled_positions(wrap=False)
    parent_s = np.asarray(geometry["cases"]["0V"]["fractional"])
    row = blank_row("QE_PBE_PAW", case_id, geometry)
    row.update(usable=inspected["usable"], status=inspected["status"])
    elapsed = (datetime.fromisoformat(record["finished_utc"]) - datetime.fromisoformat(record["started_utc"])).total_seconds()
    frames = inspected["trajectory"]
    frame_diagnostics = []
    for index, frame in enumerate(frames):
        atoms = frame_atoms(frame)
        raw, detrended, translation = nonaffine_displacement(s0, atoms)
        labels, distances, margin = nearest_site_diagnostic(atoms, parent_s, translation)
        frame_diagnostics.append({"case": case_id, "frame_index": index, "source_line": frame["source_line"],
                                  "source_is_final_section": frame.get("final_section_complete", False),
                                  "volume_A3": atoms.get_volume(), "cell_A": atoms.cell.array.tolist(),
                                  "nonaffine_detrended_max_A": float(np.linalg.norm(detrended, axis=1).max()),
                                  "nearest_site_changes": int(np.sum(labels != ids)),
                                  "min_site_assignment_margin_A": float(margin.min())})
    last_force = inspected.get("max_force_norm_eV_A")
    last_forces = inspected.get("forces_eV_A")
    diagnostic = {"case": case_id, "attempt_path": str(attempt), "runner_usable": inspected["usable"],
                  "status": inspected["status"], "returncode": record.get("returncode"),
                  "stop_reason": inspected.get("ionic_stop_reason"), "time_limit": inspected["time_limit"],
                  "electronic_converged": inspected["electronic_converged"],
                  "ionic_converged": inspected["ionic_converged"], "job_done": inspected["job_done"],
                  "final_scf_verified": inspected["final_scf_verified"],
                  "last_printed_energy_eV_DIAGNOSTIC": inspected["energy_eV"],
                  "last_printed_fmax_eV_A_DIAGNOSTIC": last_force,
                  "last_printed_max_force_component_eV_A_DIAGNOSTIC": float(np.abs(last_forces).max()) if last_forces else None,
                  "last_scf_accuracy_Ry": inspected["last_scf_accuracy_Ry"],
                  "last_printed_stress_max_abs_GPa_DIAGNOSTIC": inspected["max_stress_residual_kbar"] / 10
                       if inspected["max_stress_residual_kbar"] is not None else None,
                  "last_printed_geometry_volume_A3_DIAGNOSTIC": frames[-1]["volume_A3"] if frames else None,
                  "last_geometry_nonaffine_max_A_DIAGNOSTIC": frame_diagnostics[-1]["nonaffine_detrended_max_A"] if frames else None,
                  "geometry_frame_count": len(frames), "elapsed_s": elapsed,
                  "reasons": "; ".join(inspected["reasons"] + inspected["data_quality_reasons"]),
                  "diagnostic_alignment_note": "For failed attempts, last printed force, SCF energy and geometry may refer to different evaluation stages; no accepted energy difference is constructed."}
    atom_rows = []
    if inspected["usable"]:
        require(inspected["final_scf_verified"] and inspected["ionic_converged"] and inspected["electronic_converged"],
                "Usable QE case lacks final convergence")
        final = frame_atoms(inspected["final_geometry"])
        close(final.get_volume(), inspected["final_volume_A3"], "QE volume determinant", 1e-8)
        close(inspected["energy_Ry"] * qe.RY_TO_EV, inspected["energy_eV"], "QE Ry/eV", 1e-8)
        raw, detrended, translation = nonaffine_displacement(s0, final)
        labels, distances, margin = nearest_site_diagnostic(final, parent_s, translation)
        row.update(energy_eV=inspected["energy_eV"], smearing_minus_TS_eV=smearing_info(output),
                   volume_A3=final.get_volume(), a_equiv_parent108_A=(4 * final.get_volume() / 108)**(1/3),
                   fmax_eV_A=last_force, stress_max_abs_GPa=inspected["max_stress_residual_kbar"] / 10,
                   nonaffine_raw_max_A=float(np.linalg.norm(raw, axis=1).max()),
                   nonaffine_detrended_max_A=float(np.linalg.norm(detrended, axis=1).max()),
                   nonaffine_detrended_rms_A=float(np.sqrt(np.mean(np.sum(detrended**2, axis=1)))),
                   final_nearest_site_changes=int(np.sum(labels != ids)),
                   trajectory_frames_with_nearest_site_change=sum(d["nearest_site_changes"] > 0 for d in frame_diagnostics),
                   elapsed_s=elapsed)
        for i, parent_id in enumerate(ids):
            atom_rows.append({"method": row["method"], "case": case_id, "parent_site_id": parent_id,
                              "nearest_parent_site": int(labels[i]),
                              **{f"{name}_{axis}_A": float(values[i, j])
                                 for name, values in (("initial", initial.positions), ("final", final.positions),
                                                     ("nonaffine_raw", raw), ("nonaffine_detrended", detrended))
                                 for j, axis in enumerate("xyz")}})
    evidence = {"case": case_id, "protocol": record["protocol"], "source_input_sha256": record["source_input_sha256"],
                "runtime_input_sha256": record["runtime_input_sha256"], "pseudo_sha256": record["pseudo_sha256"],
                "output_sha256": sha256(attempt / "pw.out"), "attempt_sha256": sha256(attempt / "attempt.json"),
                "result_sha256": sha256(attempt / "result.json"),
                "parent_id_evidence": "Frozen structure.extxyz IDs, matched to unchanged input row ordering; QE retains atom order.",
                "frame_diagnostics": frame_diagnostics}
    return row, diagnostic, atom_rows, evidence


def energy_differences(rows):
    """Only same-method usable inputs may enter any reported difference."""
    for method in {r["method"] for r in rows}:
        group = {r["case"]: r for r in rows if r["method"] == method}
        zero, one = group["0V"], group["1V"]
        for row in group.values():
            if not row["usable"] or not zero["usable"]:
                continue
            factor = (108 - row["q"]) / 108
            for value, prefix in (("energy_eV", "E"), ("Ts_eV", "Ts")):
                if row[value] is not None and zero[value] is not None:
                    row[prefix + "_formation_eV"] = row[value] - factor * zero[value]
                    if row["q"] == 2 and one["usable"] and one[value] is not None:
                        row[prefix + "_binding_eV"] = 2 * one[value] - row[value] - zero[value]
            row["relaxation_volume_A3"] = row["volume_A3"] - zero["volume_A3"]
            row["formation_volume_A3"] = row["volume_A3"] - factor * zero["volume_A3"]


def plot(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9))
    fig.subplots_adjust(left=.075, right=.98, top=.82, bottom=.26, wspace=.3)
    for method, color, marker, label in (("DFTpy_LDA_local", "#175b88", "o", "OF-DFT: LDA / local"),
                                         ("QE_PBE_PAW", "#b35c00", "s", "QE: PBE / PAW")):
        group = {r["case"]: r for r in rows if r["method"] == method}
        valid = [(i, group[c]) for i, c in enumerate(CASE_IDS[1:]) if group[c]["E_formation_eV"] is not None]
        axes[0].scatter([i for i, r in valid], [r["E_formation_eV"] for i, r in valid], s=48,
                        color=color, marker=marker, label=label, zorder=3)
        pairs = [(i, group[c]) for i, c in enumerate(CASE_IDS[2:], 1) if group[c]["E_binding_eV"] is not None]
        axes[1].scatter([i for i, r in pairs], [r["E_binding_eV"] * 1000 for i, r in pairs],
                        s=48, color=color, marker=marker, label=label, zorder=3)
    axes[0].set_xticks(range(7), ["1V", "1NN", "2NN", "3NN", "4NN", "5NN*", "6NN"])
    axes[0].set(ylabel="Formation energy / functional (eV)", title="Accepted states only")
    axes[0].legend(fontsize=9, frameon=False, loc="lower right")
    axes[1].set_xticks(range(1, 7), ["1NN", "2NN", "3NN", "4NN", "5NN*", "6NN"])
    axes[1].set(ylabel="Divacancy binding (meV)", title="QE binding remains unavailable")
    axes[1].axhline(0, linestyle="--", color="#777777", linewidth=.8)
    axes[1].text(.96, .92, "No accepted QE 1V reference\nNo QE binding values inferred", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=9, color="#8d4800")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Bulk Al vacancies: available OF-DFT / QE comparison", fontsize=14, y=.96, weight="bold")
    fig.text(.075, .885, "108 parent sites · independent full-cell relaxation · zero external pressure", fontsize=10)
    fig.text(.075, .12, "QE uses the printed smearing functional at 0.02 Ry; XC and pseudopotentials differ between methods.", fontsize=9)
    fig.text(.075, .065, "Blank QE entries are unavailable results. *5NN has two equidistant periodic images; this is a finite-cell comparison.", fontsize=9)
    fig.savefig(path, dpi=350)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def summary(rows, diagnostics):
    by_method = {m: {r["case"]: r for r in rows if r["method"] == m} for m in {r["method"] for r in rows}}
    qrows, of = by_method["QE_PBE_PAW"], by_method["DFTpy_LDA_local"]
    number = lambda x, n=6: "—" if x is None else f"{x:.{n}f}"
    usable = [r["case"] for r in qrows.values() if r["usable"]]
    lines = ["# Bulk Al：QE／OF-DFT 變晶胞結果與停止原因分析", "",
             f"DFTpy 8/8 accepted；QE {len(usable)}/8 usable：{'、'.join(usable)}。"
             "QE 1V 未通過既定驗收，因此 **QE 單空缺形成能與全部雙空缺結合能目前仍缺**。"
             "已通過的 QE 2V 可搭配 QE 0V 計算形成能，不需要借用失敗的 1V。", "",
             "## 可使用的同方法能量差", "",
             "| 構型 | OF E_f eV | QE E_f eV | OF E_b meV | QE E_b meV |",
             "|---|---:|---:|---:|---:|"]
    for case in CASE_IDS[1:]:
        o, q = of[case], qrows[case]
        lines.append(f"| {case} | {number(o['E_formation_eV'])} | {number(q['E_formation_eV'])} | "
                     f"{number(o['E_binding_eV']*1000 if o['E_binding_eV'] is not None else None,3)} | "
                     f"{number(q['E_binding_eV']*1000 if q['E_binding_eV'] is not None else None,3)} |")
    lines += ["", "E_f(q)=E_q−(108−q)E_0/108；E_b=2E_1−E_2−E_0。每個式子只使用同方法且 accepted／usable 的零壓結果。"
              "空白不是零值；失敗構型的最後數字只保留在 diagnostics.csv。", "",
              "QE 的 ! total energy 為該 MP smearing 設定下列印的 F=E−TS；此表使用此量作一致線性組合，"
              "未暗自轉成零溫能量。CSV 另保留 smearing contribution (−TS)。DFTpy 為 LDA/local、TFvW .9/.1，"
              "QE 為 PBE/PAW；比較差異包含 XC、偽勢、KEDF、smearing 和鬆弛結構，不能全部歸因 TFvW。", "",
              "DFTpy 的 T_s、ΔT_s 與加權 TF/vW 來源見既有完整分析。QE one-electron contribution 並非純電子動能，"
              "本輪尚無經驗證的純 T_s 提取，因此 QE 動能欄位保留缺值。", "",
              "## 體積與結構", "",
              "| 方法／构型 | V Å³ | a_eq,108 Å | ΔV_rel Å³ | V_form Å³ | 最大非仿射位移 Å |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["usable"]:
            lines.append(f"| {row['method']}/{row['case']} | {number(row['volume_A3'])} | "
                         f"{number(row['a_equiv_parent108_A'])} | {number(row['relaxation_volume_A3'])} | "
                         f"{number(row['formation_volume_A3'])} | {number(row['nonaffine_detrended_max_A'])} |")
    lines += ["", "a_eq,108=(4V/108)^(1/3)，是以父格點數定義的體積等效尺度；缺陷晶胞未必立方。"
              "ΔV_rel=V_q−V_0；V_form=V_q−(108−q)V_0/108，兩者不同。"
              "位移按凍結輸入的 parent_site_id／QE 原子列順序配對，計算 MIC[(s_final−s_initial)h_final]，"
              "再扣除剩餘 Al 的平均平移，排除整體仿射變形。原始／去平移向量保存在原子 CSV。", "",
              "## 收斂狀態與停止原因", "",
              "| 構型 | QE 狀態 | 最後列印 fmax eV/Å | 最後 SCF accuracy Ry | 最終 relaxed-cell SCF | 執行 h |",
              "|---|---|---:|---:|---|---:|"]
    for d in diagnostics:
        lines.append(f"| {d['case']} | {d['status']} | {number(d['last_printed_fmax_eV_A_DIAGNOSTIC'],8)} | "
                     f"{d['last_scf_accuracy_Ry']} | {d['final_scf_verified']} | {d['elapsed_s']/3600:.2f} |")
    lines += ["", "上表未通過案例的『最後』值僅作診斷：最後列印的幾何、能量與力可能來自不同評估階段，"
              "不可組裝成已收斂終態。全部原始 attempt、停止原因及驗證證據保留。", "",
              "1V 的 BFGS 因 trust-radius／history reset 限制停止。即使最後力向量最大值小於外部 0.001 eV/Å，"
              "仍缺明確離子收斂及最終 relaxed-cell SCF；原始輸入另設 0.0005 eV/Å 的力分量門檻與能量／應力条件。"
              "這些是不同驗收條件，不能只看一個力值便將失敗改成成功。", "",
              "1NN、3NN、5NN 在本輪 QE max_seconds 約 11 小時限制停止。尚未完成的末輪 SCF 與仍大的原子力，"
              "表示目前沒有同精度終態；時間到並不等於物理不穩定。若後續延續，應先檢查保留的幾何／restart證據，"
              "用新 attempt 記錄延續或數值協定修訂，不倒改本輪結果。", "",
              "調整 force tolerance 不能解決所有停止原因：本輪同時涉及時間預算、電子迭代與 BFGS 停滯。"
              "若為 proposal 採較寬門檻，應先在代表性構型上量化 E_f/E_b 與結構偏差；4NN／6NN 的亚 meV OF 結合能"
              "尤其不能由寬力門檻直接保証符號。此分析沒有重送作業或更改任何科學輸入。", "",
              "## Proposal 可寫與尚缺", "",
              "已可放完整 OF 八構型及 QE 通過的 0V、2NN、4NN、6NN 的結構／形成能；QE binding 明列 pending。"
              "5NN 兩個等距影像仍作尺寸／影像診斷。靜態鬆弛不是凝聚動力學或 NEB 能障。"
              "後續優先補合格 QE 1V 參考及失敗端點，並完成 nano 最小配對矩陣、線性彈性／密度分析和代表性路徑範圍。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--of-analysis", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "Use a new analysis output directory")
    geometry = load(args.campaign / "geometry.json")
    of_report = load(args.of_analysis / "audit.json")
    of_hashes = load(args.of_analysis / "output_sha256.json")
    require(sha256(args.of_analysis / "audit.json") == of_hashes["audit.json"], "OF audit hash mismatch")
    require(of_report["status"] == "downloaded_electronic_density_and_structure_audit_passed", "OF audit not complete")
    rows, diagnostics, atom_rows, evidence = [], [], [], []
    for source in of_report["results"]:
        row = blank_row("DFTpy_LDA_local", source["case"], geometry)
        row.update({k: source[k] for k in row if k in source})
        row.update(method="DFTpy_LDA_local", usable=True, status="converged")
        rows.append(row)
    attempts = {}
    for path in args.archive.rglob("attempt.json"):
        record = load(path)
        if record.get("bulk_vacancy_qe_attempt") == 1:
            case = record["case_id"]
            require(case not in attempts, "Multiple QE attempts need explicit selection")
            attempts[case] = path.parent
    require(set(attempts) == set(CASE_IDS), "Archive does not contain all eight QE attempts")
    for case in CASE_IDS:
        row, diagnostic, atoms, audit = analyze_qe(attempts[case], args.campaign, geometry)
        rows.append(row)
        diagnostics.append(diagnostic)
        atom_rows.extend(atoms)
        evidence.append(audit)
    energy_differences(rows)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "archived_attempt_analysis_complete",
              "archive": str(args.archive), "source_script_sha256": sha256(__file__),
              "qe_parser_sha256": sha256(qe.__file__), "of_audit_sha256": sha256(args.of_analysis / "audit.json"),
              "qe_Ry_eV": qe.RY_TO_EV, "qe_Bohr_A": qe.BOHR_TO_A,
              "rows": rows, "diagnostics": diagnostics, "qe_evidence": evidence,
              "formulas": {"Ef": "Eq-(108-q)/108*E0, same method usable only", "Eb": "2E1-E2-E0, same method usable only",
                           "dVrel": "Vq-V0", "Vformation": "Vq-(108-q)/108*V0"}}
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "accepted_comparison.csv", rows)
    write_csv(args.output_dir / "qe_attempt_diagnostics.csv", diagnostics)
    write_csv(args.output_dir / "qe_accepted_atom_displacements.csv", atom_rows)
    (args.output_dir / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "summary_zh.md").write_text(summary(rows, diagnostics), encoding="utf-8")
    plot(args.output_dir / "accepted_energy_comparison.png", rows)
    (args.output_dir / "output_sha256.json").write_text(json.dumps({p.name: sha256(p) for p in args.output_dir.iterdir()
                if p.is_file()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"QE_usable": [r["case"] for r in rows if r["method"] == "QE_PBE_PAW" and r["usable"]],
                      "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
