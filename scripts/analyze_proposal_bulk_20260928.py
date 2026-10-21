"""Merge archived bulk-vacancy results without running a calculator.

Uses each archive's frozen QE inspector. All structural fields use the original
September25 ideal parent lattice, never a continuation starting geometry.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
from itertools import product
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.io import read, write

from analyze_bulk_vacancy_vcrelax_20260925 import (
    CASE_IDS, close, load, require, sha256, nonaffine_displacement, nearest_site_diagnostic,
)
from analyze_bulk_qe_of_vcrelax_20260926 import write_csv, smearing_info, energy_differences, blank_row

METHODS = ("DFTpy_LDA_local", "QE_PBE_PAW")
CONTINUE = ("1V", "2V_1NN_D110", "2V_3NN_D211", "2V_5NN_D310")


def initial_image_multiplicity(delta_s, cell):
    images=(np.asarray(delta_s)[None,:]+np.asarray(list(product((-1,0,1),repeat=3))))@cell
    distance=np.linalg.norm(images,axis=1)
    return int(np.sum(np.abs(distance-distance.min())<1e-7))


def import_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Archive:
    def __init__(self, root):
        self.root = root
        self.manifest = load(root / "ARCHIVE_MANIFEST.json")
        self.checked = {}

    def verify(self, path):
        relative = path.relative_to(self.root).as_posix()
        if relative not in self.checked:
            item = self.manifest["files"][relative]
            require(path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"],
                    "Archive content mismatch: " + relative)
            self.checked[relative] = item["sha256"]
        return path

    def json(self, path):
        return load(self.verify(path))


def elapsed(record):
    return (datetime.fromisoformat(record["finished_utc"]) - datetime.fromisoformat(record["started_utc"])).total_seconds()


def qe_reader(archive, continuation=False):
    snapshot = archive.root / "source_snapshot"
    script = snapshot / "scripts" / ("run_bulk_vacancy_qe_continue_20260926.py" if continuation
                                      else "run_bulk_vacancy_qe_vcrelax_20260925.py")
    module = import_source(archive.verify(script), "frozen_continuation" if continuation else "frozen_original")
    base = module.base if continuation else module
    archive.verify(snapshot / "scripts/run_bulk_vacancy_qe_vcrelax_20260925.py")
    campaign = snapshot / "campaigns" / ("al_bulk_vacancy_vcrelax_continue_20260926" if continuation
                                         else "al_bulk_vacancy_vcrelax_20260925")
    verified = base.verify_campaign(campaign)
    # Verification is read-only and identical for every attempt; avoid repeated
    # /mnt/c traversal of all parent evidence while retaining per-attempt checks.
    base.verify_campaign = lambda path: verified
    return base, campaign


def inspect_qe(archive, base, campaign, case_id, continuation, original_geometry):
    roots = list((archive.root / ("qe_continuation" if continuation else "qe") / case_id).glob("*/attempt.json"))
    require(len(roots) == 1, "Expected exactly one archived attempt per phase/case")
    folder = roots[0].parent
    record = archive.json(folder / "attempt.json")
    saved = archive.json(folder / "result.json")
    for name in record["immutable_file_sha256"]:
        archive.verify(folder / name)
    for name in ("pw.out", "pw.err"):
        archive.verify(folder / name)
    result = base.inspect_attempt(folder, campaign)
    require(result["status"] != "provenance_failed", f"{case_id} provenance: {result.get('reasons')}")
    require(result["usable"] == saved["usable"], "Frozen inspector and archived acceptance disagree")
    case = original_geometry["cases"][case_id]
    if continuation:
        continuation_geometry = load(folder / "source/geometry.json")["cases"][case_id]
        require(continuation_geometry["remaining_parent_ids"] == case["remaining_parent_ids"], "Continuation parent identity changed")
        original = load(folder / "source/parent_parent_geometry.json")
        require(original == original_geometry, "Continuation parent geometry differs from original campaign")
        parsed_old = base.parse_input((folder / "source/parent_parent_pw_in.in").read_text())
        parsed_new = base.parse_input((folder / "pw.in").read_text())
        for parsed in (parsed_old, parsed_new):
            parsed["scientific"]["cards"].pop("CELL_PARAMETERS")
            parsed["scientific"]["cards"].pop("ATOMIC_POSITIONS")
        require(parsed_old["scientific"] == parsed_new["scientific"], "Continuation physical settings changed")
    else:
        source_atoms = read(folder / "source/structure.extxyz")
        require(source_atoms.arrays["parent_site_id"].tolist() == case["remaining_parent_ids"], "Original parent identity changed")
        close(source_atoms.positions, case["positions_A"], "Original QE parent reference", 1e-8)
    phase = "continuation_20260926" if continuation else "original_20260925"
    diagnostic = {"method": METHODS[1], "case": case_id, "phase": phase, "attempt": folder.name,
                  "usable": result["usable"], "status": result["status"], "elapsed_s": elapsed(record),
                  "elapsed_h": elapsed(record) / 3600, "returncode": record.get("returncode"),
                  "started": record["started_utc"], "finished": record["finished_utc"],
                  "final_scf_verified": result["final_scf_verified"], "ionic_converged": result["ionic_converged"],
                  "electronic_converged": result["electronic_converged"],
                  "last_printed_fmax_eV_A_diagnostic": result["max_force_norm_eV_A"],
                  "last_printed_scf_accuracy_Ry_diagnostic": result["last_scf_accuracy_Ry"],
                  "last_printed_energy_eV_diagnostic": result["energy_eV"],
                  "reasons": "; ".join(result["reasons"] + result["data_quality_reasons"]),
                  "source": str(folder),
                  "alignment_note": "Unusable attempts: last printed columns are diagnostics, not a matched accepted final state."}
    row = blank_row(METHODS[1], case_id, original_geometry)
    row.update(usable=result["usable"], status=result["status"], selected_phase=phase, selected_attempt=folder.name,
               latest_attempt_elapsed_s=elapsed(record), elapsed_s=elapsed(record),
               energy_semantics="QE printed F=E-TS smearing functional; MP 0.02 Ry")
    final = None
    if result["usable"]:
        require(result["final_scf_verified"] and result["ionic_converged"] and result["electronic_converged"], "Final QE gates absent")
        frame = result["final_geometry"]
        final = Atoms(frame["symbols"], positions=frame["positions_A"], cell=frame["cell_A"], pbc=True)
        close(final.get_volume(), result["final_volume_A3"], "QE final volume", 1e-8)
        close(result["energy_Ry"] * base.RY_TO_EV, result["energy_eV"], "QE energy units", 1e-9)
        row.update(energy_eV=result["energy_eV"], volume_A3=final.get_volume(),
                   a_equiv_parent108_A=(final.get_volume() / 27)**(1/3),
                   fmax_eV_A=result["max_force_norm_eV_A"],
                   stress_max_abs_GPa=result["max_stress_residual_kbar"] / 10,
                   smearing_minus_TS_eV=smearing_info((folder / "pw.out").read_text()))
    return row, diagnostic, final, {"case": case_id, "phase": phase, "input_hash": record["runtime_input_sha256"],
            "pseudo_sha256": record["pseudo_sha256"], "protocol": record["protocol"], "archive_path": str(folder),
            "inspected_with": str(base.__file__), "geometry_frames": result["trajectory"]}


def read_of(archive, case_id, geometry, prior):
    files = list((archive.root / "dftpy" / case_id).glob("*/result.json"))
    require(len(files) == 1, "Expected one OF attempt")
    folder = files[0].parent
    result = archive.json(folder / "result.json")
    provenance = archive.json(folder / "provenance.json")
    require(result["accepted"] and result["all_electronic_calls_converged"] and result["ionic_converged"], "OF acceptance missing")
    archive.verify(folder / "final.extxyz")
    require(sha256(folder / "final.extxyz") == result["artifact_sha256"]["final.extxyz"], "OF geometry artifact mismatch")
    final = read(folder / "final.extxyz")
    require(final.arrays["parent_site_id"].tolist() == geometry["cases"][case_id]["remaining_parent_ids"], "OF parent IDs changed")
    close(final.get_volume(), result["volume_A3"], "OF volume", 1e-8)
    close(result["energy_Ha"] * provenance["units"]["Hartree_eV"], result["energy_eV"], "OF energy units", 1e-9)
    close(result["energy_eV"], prior["energy_eV"], "OF prior audited energy", 1e-9)
    row = blank_row(METHODS[0], case_id, geometry)
    row.update({key: prior[key] for key in row if key in prior})
    terms = result["energy_components_eV"]
    row.update(usable=True, status="converged", selected_phase="original_20260925", selected_attempt=folder.name,
               energy_semantics="OF electronic total energy", energy_eV=result["energy_eV"], Ts_eV=terms["KEDF"],
               weighted_TF_eV=terms["KEDF-TF"], weighted_vW_eV=terms["KEDF-VW"],
               latest_attempt_elapsed_s=result["elapsed_s"], cumulative_case_elapsed_s=result["elapsed_s"],
               cumulative_case_elapsed_h=result["elapsed_s"]/3600, attempt_count=1)
    diagnostic = {"method": METHODS[0], "case": case_id, "phase": "original_20260925", "attempt": folder.name,
                  "usable": True, "status": result["status"], "elapsed_s": result["elapsed_s"],
                  "elapsed_h": result["elapsed_s"]/3600, "last_printed_fmax_eV_A_diagnostic": result["fmax_eV_A"],
                  "electronic_converged": result["all_electronic_calls_converged"], "ionic_converged": result["ionic_converged"],
                  "source": str(folder), "reasons": ""}
    return row, diagnostic, final


def structural_fields(row, final, geometry, method_zero_cell):
    """Kinematic diagnostics: no elastic constants, dipoles, or kinetics inferred."""
    case = geometry["cases"][row["case"]]
    ids = case["remaining_parent_ids"]
    ids_array = np.asarray(ids)
    s0 = np.asarray(case["fractional"])
    all_s = np.asarray(geometry["cases"]["0V"]["fractional"])
    h0, h = np.asarray(geometry["cell_A"]), final.cell.array
    raw, u, translation = nonaffine_displacement(s0, final)
    labels, _, margin = nearest_site_diagnostic(final, all_s, translation)
    row.update(nonaffine_raw_max_A=float(np.linalg.norm(raw, axis=1).max()),
               nonaffine_detrended_max_A=float(np.linalg.norm(u, axis=1).max()),
               nonaffine_detrended_rms_A=float(np.sqrt(np.mean(np.sum(u*u, axis=1)))),
               final_nearest_site_changes=int(np.sum(labels != ids_array)),
               final_min_site_assignment_margin_A=float(margin.min()),
               vacancy_site_fraction=row["q"]/108, nominal_vacancy_number_density_Aminus3=row["q"]/final.get_volume(),
               mean_valence_electron_density_e_A3=3*len(final)/final.get_volume())
    tensors = {}
    for tag, reference in (("from_initial", h0), ("defect_vs_method_0V", method_zero_cell)):
        F = h.T @ np.linalg.inv(reference.T)
        eps = .5*(F+F.T)-np.eye(3)
        green = .5*(F.T@F-np.eye(3))
        tensors[tag] = {"F_column_convention": F.tolist(), "infinitesimal_strain": eps.tolist(),
                        "Green_Lagrange_strain": green.tolist(), "detF_minus1": float(np.linalg.det(F)-1)}
        row[tag+"_volumetric_strain"] = float(np.linalg.det(F)-1)
        for i, j, key in ((0,0,"xx"),(1,1,"yy"),(2,2,"zz"),(0,1,"xy"),(0,2,"xz"),(1,2,"yz")):
            row[tag+"_small_strain_"+key] = float(eps[i,j])
    # Original 12-neighbour FCC graph is fixed before the vacancies are removed.
    nearest_distance = geometry["a_ref_A"] / np.sqrt(2)
    eps_local, residuals, ranks, counts, condition = [], [], [], [], []
    for i, parent in enumerate(ids):
        ideal, distance = find_mic((s0-s0[i])@h0, h0, pbc=True)
        neighbors = np.flatnonzero(np.abs(distance-nearest_distance) < 1e-6)
        x, _ = find_mic((s0[neighbors]-s0[i])@h, h, pbc=True)
        y = x+u[neighbors]-u[i]
        B, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
        require(rank == 3 and len(neighbors) >= 6, "Local neighbour fit lacks 3D rank")
        eps_local.append(.5*(B+B.T)-np.eye(3))
        residuals.append(float(np.sqrt(np.mean(np.sum((x@B-y)**2, axis=1)))))
        ranks.append(int(rank)); counts.append(len(neighbors)); condition.append(float(singular[0]/singular[-1]))
    eps_local = np.asarray(eps_local)
    local_trace = np.trace(eps_local, axis1=1, axis2=2)
    row.update(local_nonaffine_strain_trace_min=float(local_trace.min()), local_nonaffine_strain_trace_max=float(local_trace.max()),
               local_neighbor_fit_rms_max_A=max(residuals), local_fit_min_neighbors=min(counts), local_fit_min_rank=min(ranks))
    per_atom, radial = [], []
    removed = case["removed_parent_ids"]
    vacancy_frac = all_s[removed]
    for i, parent in enumerate(ids):
        atom = {"method": row["method"], "case": row["case"], "parent_site_id": parent,
                "nearest_final_parent_site": int(labels[i]), "local_fit_neighbor_count": counts[i],
                "local_fit_rank": ranks[i], "local_fit_condition": condition[i], "local_fit_rms_A": residuals[i],
                "local_nonaffine_strain_trace": float(local_trace[i]),
                **{f"local_nonaffine_strain_{key}": float(eps_local[i,j,k])
                   for j,k,key in ((0,0,"xx"),(1,1,"yy"),(2,2,"zz"),(0,1,"xy"),(0,2,"xz"),(1,2,"yz"))},
                **{f"{tag}_{axis}": float(values[i,j]) for tag,values in
                   (("initial_s",s0),("initial_position_A",s0@h0),("final_position_A",final.positions),
                    ("u_raw_A",raw),("u_detrended_A",u)) for j,axis in enumerate("xyz")}}
        if removed:
            r0vec, r0 = find_mic((s0[i]-vacancy_frac)@h0, h0, pbc=True)
            rvec, radius = find_mic((s0[i]-vacancy_frac)@h, h, pbc=True)
            tie = np.flatnonzero(np.abs(r0-r0.min()) < 1e-7)
            nearest = int(tie[0])
            for k, vac_id in enumerate(removed):
                radial.append({"method": row["method"], "case": row["case"], "parent_site_id": parent,
                               "removed_parent_site_id": vac_id, "initial_parent_distance_A": float(r0[k]),
                               "affinely_mapped_parent_distance_A": float(radius[k]),
                               "u_radial_A": float(u[i]@rvec[k]/radius[k]),
                               "is_nearest_vacancy_reference": bool(k in tie), "nearest_vacancy_tie": len(tie)>1,
                               "initial_parent_MIC_image_multiplicity":initial_image_multiplicity(s0[i]-vacancy_frac[k],h0),
                               "vacancy_reference_note": "Deleted ideal lattice site affinely mapped to final cell; not a moving vacancy particle"})
            atom.update(nearest_removed_parent_site_id=removed[nearest], nearest_vacancy_tie=len(tie)>1,
                        nearest_initial_MIC_image_multiplicity=initial_image_multiplicity(s0[i]-vacancy_frac[nearest],h0),
                        nearest_initial_parent_distance_A=float(r0[nearest]),
                        nearest_u_radial_A=float(u[i]@rvec[nearest]/radius[nearest]))
        per_atom.append(atom)
    exported = final.copy()
    for name, values in (("parent_site_id", ids_array), ("u_nonaffine_raw_A",raw), ("u_nonaffine_A",u),
                         ("local_strain_tensor",eps_local.reshape(-1,9)), ("local_strain_trace",local_trace),
                         ("local_fit_rms_A",np.asarray(residuals)),("local_neighbor_count",np.asarray(counts)),("local_fit_rank",np.asarray(ranks))):
        exported.set_array(name,np.asarray(values))
    exported.info.update(displacement_reference="20260925 ideal parent; final-cell affine part and mean translation removed",
                         local_strain_definition="LSQ symmetric gradient relative final-cell affine ideal FCC 1NN graph; vacancies removed",
                         method=row["method"], case=row["case"], selected_attempt=row["selected_attempt"])
    return per_atom, radial, tensors, exported


def derived_fields(rows):
    energy_differences(rows)
    for method in METHODS:
        group = {r["case"]:r for r in rows if r["method"]==method}
        zero, single, first = group["0V"], group["1V"], group["2V_1NN_D110"]
        for row in group.values():
            row.update(delta_a_equiv_from_method_0V_A=None, pair_relative_to_1NN_eV=None,
                       weighted_TF_formation_eV=None, weighted_TF_binding_eV=None,
                       weighted_vW_formation_eV=None, weighted_vW_binding_eV=None)
            if not row["usable"]:
                continue
            row["delta_a_equiv_from_method_0V_A"] = row["a_equiv_parent108_A"]-zero["a_equiv_parent108_A"]
            if row["q"]==2 and first["usable"]:
                row["pair_relative_to_1NN_eV"] = row["energy_eV"]-first["energy_eV"]
            if method==METHODS[0]:
                for term in ("weighted_TF", "weighted_vW"):
                    row[term+"_formation_eV"] = row[term+"_eV"]-(108-row["q"])/108*zero[term+"_eV"]
                    if row["q"]==2:
                        row[term+"_binding_eV"] = 2*single[term+"_eV"]-row[term+"_eV"]-zero[term+"_eV"]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def make_figures(folder, rows, atoms_rows, radial, geometry, exports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,
                         "axes.spines.top":False,"axes.spines.right":False})
    colors={METHODS[0]:"#17628d",METHODS[1]:"#bb5b08"}
    names={METHODS[0]:"OF-DFT: LDA / local",METHODS[1]:"QE: PBE / PAW"}
    data={m:{r['case']:r for r in rows if r['method']==m} for m in METHODS}
    pair_labels=["1NN","2NN","3NN","4NN","5NN*","6NN"]
    all_labels=["1V"]+pair_labels
    def save(fig,name):
        fig.savefig(folder/(name+".png"),dpi=350)
        fig.savefig(folder/(name+".pdf"))
        plt.close(fig)
    def axes_style(axes):
        for ax in np.asarray(axes).ravel():ax.grid(axis="y",alpha=.2)
    def series(ax,method,cases,field,scale=1):
        points=[(i,data[method][c][field]) for i,c in enumerate(cases) if data[method][c].get(field) is not None]
        ax.scatter([i for i,v in points],[v*scale for i,v in points],s=46,color=colors[method],
                   marker="o" if method==METHODS[0] else "s",label=names[method],zorder=3)
    fig,ax=plt.subplots(1,2,figsize=(10,4.9));fig.subplots_adjust(left=.085,right=.97,bottom=.25,top=.82,wspace=.3)
    for m in METHODS:
        series(ax[0],m,CASE_IDS[2:],"E_formation_eV")
        series(ax[1],m,CASE_IDS[2:],"pair_relative_to_1NN_eV",1000)
    for a in ax:a.set_xticks(range(6),pair_labels)
    ax[0].set(ylabel="Divacancy formation functional (eV)",title="Accepted 2V + same-method 0V")
    ax[1].set(ylabel=r"$E_{2V}(r)-E_{2V}(1NN)$ (meV)",title="Relative ordering of six pair states")
    ax[1].axhline(0,color="#777",ls="--",lw=.8)
    ax[0].legend(frameon=False,fontsize=9)
    axes_style(ax)
    fig.suptitle("Bulk Al: divacancy energetics",fontsize=15,weight="bold",y=.96)
    fig.text(.085,.13,"No accepted QE 1V reference: QE formation(1V) and all QE binding energies remain unavailable.",fontsize=9)
    fig.text(.085,.065,"108 parent sites; zero pressure. QE: MP 0.02 Ry functional. *5NN has two equidistant periodic images.",fontsize=9)
    save(fig,"01_energetics")
    fig,ax=plt.subplots(1,2,figsize=(10,4.8));fig.subplots_adjust(left=.085,right=.97,bottom=.23,top=.82,wspace=.31)
    for m in METHODS:
        series(ax[0],m,CASE_IDS[1:],"relaxation_volume_A3")
        series(ax[1],m,CASE_IDS[1:],"nonaffine_detrended_max_A")
    for a in ax:a.set_xticks(range(7),all_labels)
    ax[0].set(ylabel=r"Relaxation volume $V_q-V_0$ ($\mathrm{\AA}^3$)",title="Each method uses its own relaxed 0V")
    ax[1].set(ylabel=r"Maximum $|u_{\rm nonaffine}|$ ($\mathrm{\AA}$)",title="Affine cell change and translation removed")
    ax[0].legend(frameon=False,fontsize=9);axes_style(ax)
    fig.suptitle("Bulk Al: cell response and atomic displacement",fontsize=15,weight="bold",y=.96)
    fig.text(.085,.12,"Reference: original 2026-09-25 ideal parent IDs, including continuation cases. QE 1V is missing.",fontsize=9)
    fig.text(.085,.06,"Cell relaxation is global; the plotted atomic displacement isolates the nonaffine part. *5NN: image diagnostic.",fontsize=9)
    save(fig,"02_volume_displacement")
    # Same original-parent slice and a shared scale, with no spatial interpolation.
    case="2V_1NN_D110";removed=geometry["cases"][case]["removed_parent_ids"]
    all_s=np.asarray(geometry["cases"]["0V"]["fractional"]);h0=np.asarray(geometry["cell_A"])
    origin=all_s[removed[0]]
    sdata={m:[r for r in atoms_rows if r['method']==m and r['case']==case] for m in METHODS}
    plane={}
    for m in METHODS:
        coords=np.asarray([[r['initial_s_'+a] for a in 'xyz'] for r in sdata[m]])
        vectors,_=find_mic((coords-origin)@h0,h0,pbc=True)
        keep=np.abs(vectors[:,2])<1e-6
        plane[m]=([r for r,k in zip(sdata[m],keep) if k],vectors[keep])
    vacxy,_=find_mic((all_s[removed]-origin)@h0,h0,pbc=True)
    for field,name,title in (("displacement","03_1NN_displacement","1NN divacancy: nonaffine atomic displacement"),
                             ("strain","04_1NN_local_strain","1NN divacancy: local nonaffine strain")):
        fig,ax=plt.subplots(1,2,figsize=(10,5.5));fig.subplots_adjust(left=.07,right=.84,bottom=.23,top=.80,wspace=.2)
        if field=="displacement":
            vmax=max(np.linalg.norm([r['u_detrended_A_'+a] for a in 'xyz']) for m in METHODS for r in plane[m][0])
            norm=Normalize(0,vmax);cmap="viridis";clabel=r"$|u_{\rm nonaffine}|$ ($\mathrm{\AA}$)"
        else:
            vmax=max(abs(r['local_nonaffine_strain_trace'])*100 for m in METHODS for r in plane[m][0])
            norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax);cmap="RdBu_r";clabel=r"$\mathrm{tr}(\epsilon_{\rm local,nonaffine})$ (%)"
        for a,m in zip(ax,METHODS):
            values,coords=plane[m];u=np.asarray([[r['u_detrended_A_'+x] for x in 'xyz'] for r in values])
            color=np.linalg.norm(u,axis=1) if field=="displacement" else np.asarray([r['local_nonaffine_strain_trace']*100 for r in values])
            points=a.scatter(coords[:,0],coords[:,1],c=color,norm=norm,cmap=cmap,s=85,edgecolor="white",linewidth=.6,zorder=3)
            if field=="displacement":a.quiver(coords[:,0],coords[:,1],u[:,0]*8,u[:,1]*8,angles='xy',scale_units='xy',scale=1,width=.006,color="#232323",zorder=4)
            for xy,pid in zip(vacxy,removed):
                a.scatter(xy[0],xy[1],marker='X',s=110,c='#d92135',edgecolor='white',linewidth=.7,zorder=5)
                a.annotate(f"V(p{pid})",xy[:2],xytext=(5,8),textcoords='offset points',fontsize=8,color='#a80016')
            for i in np.argsort(np.linalg.norm(u,axis=1))[-3:]:
                a.annotate(f"p{values[i]['parent_site_id']}",coords[i,:2],xytext=(5,-13),textcoords='offset points',fontsize=8)
            a.set(title=names[m],xlabel=r"Original-parent $x-x_{V1}$ ($\mathrm{\AA}$)",xlim=(-6.5,6.5),ylim=(-6.5,6.5),aspect='equal')
            a.grid(alpha=.15)
        ax[0].set_ylabel(r"Original-parent $y-y_{V1}$ ($\mathrm{\AA}$)")
        cax=fig.add_axes([.87,.27,.022,.46]);fig.colorbar(points,cax=cax,label=clabel)
        fig.suptitle(title,fontsize=15,weight='bold',y=.95)
        fig.text(.07,.855,"Same ideal-parent z plane through both deleted sites; p = persistent parent ID; V = deleted parent site.",fontsize=9)
        fig.text(.07,.125,"Arrows = nonaffine displacement ×8; site markers are the original reference, not moving vacancy particles." if field=='displacement' else
                 "Local LSQ fit uses surviving original FCC nearest neighbours, referenced to the final-cell affine ideal lattice.",fontsize=9)
        fig.text(.07,.067,"Finite periodic-cell kinematic diagnostic; static relaxation does not establish a migration path or elastic dipole.",fontsize=9)
        save(fig,name)
    fig,ax=plt.subplots(1,2,figsize=(10,4.9));fig.subplots_adjust(left=.085,right=.97,bottom=.24,top=.81,wspace=.3)
    for a,case in zip(ax,("1V","2V_1NN_D110")):
        for m in METHODS:
            values=[r for r in atoms_rows if r['method']==m and r['case']==case and not r.get('nearest_vacancy_tie',True)
                    and r.get('nearest_initial_MIC_image_multiplicity',1)==1]
            if not values:continue
            shells={}
            for r in values:
                shell=round((2*r['nearest_initial_parent_distance_A']/geometry['a_ref_A'])**2)
                shells.setdefault(shell,[]).append(r)
            x=[np.mean([r['nearest_initial_parent_distance_A'] for r in group]) for _,group in sorted(shells.items())]
            y=[np.mean([r['nearest_u_radial_A'] for r in group]) for _,group in sorted(shells.items())]
            err=[np.std([r['nearest_u_radial_A'] for r in group]) for _,group in sorted(shells.items())]
            a.errorbar(x,y,yerr=err,fmt='o' if m==METHODS[0] else 's',color=colors[m],ms=5,capsize=2,label=names[m])
        a.axhline(0,color='#777',ls='--',lw=.8)
        a.set(title=case+(' (QE unavailable)' if case=='1V' else ''),xlabel=r"Distance to nearest deleted ideal site ($\mathrm{\AA}$)",ylabel=r"Radial nonaffine displacement ($\mathrm{\AA}$)")
        a.legend(frameon=False,fontsize=8);a.grid(alpha=.2)
    fig.suptitle("Radial atomic relaxation around the vacancy reference sites",fontsize=14,weight='bold',y=.96)
    fig.text(.085,.13,"Positive = outward; negative = inward. Points: shell mean; bars: within-shell standard deviation, not uncertainty.",fontsize=9)
    fig.text(.085,.065,"Ties between vacancy sites or periodic images excluded from means; full atomic values retained. No elastic fit assumed.",fontsize=9)
    save(fig,"05_radial_displacement")
    fig,ax=plt.subplots(figsize=(7.4,4.8));fig.subplots_adjust(left=.13,right=.97,bottom=.23,top=.82)
    x=np.arange(6);of=data[METHODS[0]]
    ax.bar(x-.18,[of[c]['E_binding_eV']*1000 for c in CASE_IDS[2:]],width=.36,color='#17628d',label=r'Total $E_b$')
    ax.bar(x+.18,[of[c]['Ts_binding_eV']*1000 for c in CASE_IDS[2:]],width=.36,color='#a7592c',label=r'Kinetic contribution $\Delta T_s^b$')
    ax.set_xticks(x,pair_labels);ax.set_ylabel('Binding combination (meV)');ax.axhline(0,color='#555',lw=.8)
    ax.legend(frameon=False,fontsize=9);ax.grid(axis='y',alpha=.15)
    fig.suptitle('OF-DFT: total binding and kinetic contribution',fontsize=13,weight='bold',y=.95)
    fig.text(.13,.12,r'Both use $2X_{1V}-X_{2V}-X_{0V}$; kinetic terms alone do not determine binding.',fontsize=9)
    fig.text(.13,.06,'QE kinetic decomposition is unavailable; *5NN is an image diagnostic.',fontsize=9)
    save(fig,'06_OF_binding_kinetic')


def method_notes(rows,diagnostics):
    group={m:{r['case']:r for r in rows if r['method']==m} for m in METHODS}
    q=group[METHODS[1]];of=group[METHODS[0]]
    lines=['# 9/28 合併分析：科學定義及結果限制','',
           '正式表16列：DFTpy 8/8 accepted、QE 7/8 usable。QE 1V兩次均失敗，因此單空缺形成能及全部QE結合能保留null；'
           '失敗末步不進入正式能量組合。QE 0V/2NN/4NN/6NN來自第一輪，1NN/3NN/5NN來自經驗證的continuation。','',
           '兩批各用其frozen runner／wrapper inspect，核對原始輸出和不可變輸入。Continuation wrapper驗證除初始幾何和操作預算外的物理設定與父attempt一致。'
           '所有位移都從9/25共同理想FCC parent lattice配對，不能以continuation開始位置當零位移。','',
           '| pair | OF E_f eV | QE E_f eV | OF E_b meV | QE E2−E2(1NN) meV |','|---|---:|---:|---:|---:|']
    for c in CASE_IDS[2:]:lines.append(f"| {c} | {of[c]['E_formation_eV']:.6f} | {q[c]['E_formation_eV']:.6f} | {of[c]['E_binding_eV']*1000:.3f} | {q[c]['pair_relative_to_1NN_eV']*1000:.3f} |")
    lines+=['','E_f(q)=E_q−(108−q)E_0/108；E_b=2E_1−E_2−E_0；ΔT_s及weighted TF/vW採同式。'
            '同方法2V相對能E_2(r)−E_2(1NN)只比較相同106原子，不需要1V，**不是binding**。正負以1NN為零點，非無限距離參考。','',
            'QE能量採final relaxed-cell SCF列印的F=E−TS（MP smearing 0.02 Ry），保留−TS值；不是默認零溫內能。'
            'DFTpy LDA/local與QE PBE/PAW不同，差異不能僅歸因TFvW。QE one-electron contribution不是純T_s；本輪QE ΔTkin缺值。','',
            'a_eq=(4V/108)^(1/3)，Δa_eq=a_eq(q)−a_eq(0V)。缺陷胞不必立方，不能把a_eq冒充三個晶格向量都相等。'
            'Relaxation volume V_q−V_0；formation volume V_q−(108−q)V_0/108，分別報告。','',
            '## 位移、局部與全局應變','',
            '非仿射位移u_raw=MIC[(s_final−s_ideal)h_final]，再扣除剩餘Al平均平移u=u_raw−mean(u_raw)。'
            '原子parent_site_id與QE輸入列順序固定。空缺標記是刪除的理想父格點，沒有被當作物理粒子追蹤。','',
            '局部neighbor-fit使用9/25理想FCC的固定第一近鄰圖，刪除缺失的vacancy neighbors；不按鬆弛後距離重新挑鄰居。'
            'X_j=MIC[(s_j−s_i)h_final]，Y_j=X_j+(u_j−u_i)，最小平方法X B≈Y，ε_local=(B+Bᵀ)/2−I。'
            '此量相對「最終晶胞仿射映射的理想晶格」，已去除全局仿射變形。它是局部kinematic小應變診斷，非總應變、continuum elastic-dipole或無限域strain field。'
            '每原子保留neighbor count、rank、設計矩陣condition number、RMS向量擬合殘差及六應變分量；未作平滑插值。','',
            '全局F=h_finalᵀ h_ref^(−T)，另報sym(F)−I、Green-Lagrange=(FᵀF−I)/2與detF−1；'
            'h_ref分別取9/25初始cell及同方法relaxed 0V。全局量與local_nonaffine量不能混稱。零壓殘餘stress不能直接當elastic dipole。','',
            'Radial displacement取u沿「最終晶胞仿射映射父格點→原子理想位點」MIC向量投影；正為外移、負為內移。'
            'CSV保留每原子對每個deleted site的投影；圖使用最近位點，且剔除最近空缺tie與理想參考中同一空缺的等距週期影像tie。'
            '存在影像tie的原始radial投影依ASE MIC選擇，方向非唯一，不能單獨解讀；CSV保留image multiplicity。error bars為殼內標準差，非統計/數值不確定度。','',
            '## 濃度、密度、成本與限制','',
            'Vacancy site fraction=q/108（無因次）；nominal vacancy number density=q/V（Å⁻³）；'
            'mean valence electron density=3N/V（e Å⁻³）。三者不同，均不是電子密度的局部切面。未用局部電子密度值代替空缺濃度。','',
            '耗時分latest_attempt_elapsed_s與cumulative_case_elapsed_s/h，後者包括同case失敗原attempt及continuation；'
            'all_attempt_diagnostics保留20個attempt（8 OF＋12 QE）。不能把continuation耗時當整個構型總成本。','',
            '5NN有兩個等距週期影像；6NN不是無限分離。亞meV差異尚需尺寸與數值誤差評估。'
            '所有結果是靜態鬆弛，不是NEB、擴散路徑、凝聚速率或自發凝聚。','',
            '04_structures包含每個accepted方法/構型的POSCAR、extxyz與metadata。extxyz帶parent IDs、位移、local strain、fit residual供OVITO；'
            'POSCAR不含per-atom附加欄，故不可單靠POSCAR恢復所有標籤。','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-archive",required=True,type=Path)
    parser.add_argument("--new-archive",required=True,type=Path)
    parser.add_argument("--of-analysis",required=True,type=Path)
    parser.add_argument("--output-root",required=True,type=Path)
    args=parser.parse_args()
    old, new = Archive(args.old_archive), Archive(args.new_archive)
    analysis, figures, structures = (args.output_root/x for x in ("02_analysis","03_figures","04_structures"))
    require(not analysis.exists() and not figures.exists() and not structures.exists(), "Analysis output folders already exist")
    original_campaign = old.root/"source_snapshot/campaigns/al_bulk_vacancy_vcrelax_20260925"
    geometry=old.json(original_campaign/"geometry.json")
    prior=load(args.of_analysis/"audit.json")
    require(sha256(args.of_analysis/"audit.json")==load(args.of_analysis/"output_sha256.json")["audit.json"],"Prior OF audit bytes changed")
    require(prior["status"]=="downloaded_electronic_density_and_structure_audit_passed","OF prior audit incomplete")
    prior_rows={r["case"]:r for r in prior["results"]}
    rows, diagnostics, qe_evidence, finals = [], [], [], {}
    for case in CASE_IDS:
        row, diag, atoms=read_of(old,case,geometry,prior_rows[case])
        rows.append(row); diagnostics.append(diag); finals[(METHODS[0],case)]=atoms
    print("OF accepted records verified",flush=True)
    old_base,old_campaign=qe_reader(old)
    new_base,new_campaign=qe_reader(new,True)
    for case in CASE_IDS:
        original=inspect_qe(old,old_base,old_campaign,case,False,geometry)
        diagnostics.append(original[1]); qe_evidence.append(original[3]); selected=original
        cumulative=original[1]["elapsed_s"]
        if case in CONTINUE:
            continued=inspect_qe(new,new_base,new_campaign,case,True,geometry)
            diagnostics.append(continued[1]); qe_evidence.append(continued[3]); selected=continued
            cumulative+=continued[1]["elapsed_s"]
        row,diag,atoms,_=selected
        row.update(cumulative_case_elapsed_s=cumulative,cumulative_case_elapsed_h=cumulative/3600,
                   attempt_count=2 if case in CONTINUE else 1)
        rows.append(row)
        if atoms is not None: finals[(METHODS[1],case)]=atoms
        print(case+": "+row["status"],flush=True)
    derived_fields(rows)
    atoms_rows, radial_rows, tensors, exports = [],[],{},{}
    for row in rows:
        if not row["usable"]: continue
        key=(row["method"],row["case"])
        atom, radial, tensor, exported=structural_fields(row,finals[key],geometry,finals[(row["method"],"0V")].cell.array)
        atoms_rows.extend(atom); radial_rows.extend(radial); tensors['/'.join(key)]=tensor; exports[key]=exported
    for folder in (analysis,figures,structures):folder.mkdir(parents=True)
    write_csv(analysis/"accepted_comparison_16rows.csv",rows)
    write_csv(analysis/"all_attempt_diagnostics.csv",diagnostics)
    write_csv(analysis/"per_atom_displacement_local_strain.csv",atoms_rows)
    write_csv(analysis/"radial_displacement_per_removed_site.csv",radial_rows)
    save_json(analysis/"core_results.json",{"rows":rows,"diagnostics":diagnostics,"global_strain":tensors})
    save_json(analysis/"analysis.json",{"rows":rows,"diagnostics":diagnostics,"global_strain":tensors})
    print("CORE_RESULTS_READY "+str(analysis/"core_results.json"),flush=True)
    for (method,case),atoms in exports.items():
        folder=structures/method/case;folder.mkdir(parents=True)
        write(folder/"final.extxyz",atoms,format="extxyz")
        write(folder/"POSCAR",atoms,format="vasp",direct=True,vasp5=True)
        save_json(folder/"metadata.json",{"method":method,"case":case,"parent_ids":geometry["cases"][case]["remaining_parent_ids"],
                  "removed_parent_ids":geometry["cases"][case]["removed_parent_ids"],"selected_attempt":atoms.info["selected_attempt"],
                  "POSCAR_note":"Parent IDs and per-atom fields are preserved in final.extxyz and metadata, not in POSCAR."})
    make_figures(figures,rows,atoms_rows,radial_rows,geometry,exports)
    notes=method_notes(rows,diagnostics)
    (analysis/"analysis_notes_zh.md").write_text(notes,encoding="utf-8")
    save_json(analysis/"audit.json",{"created_utc":datetime.now(timezone.utc).isoformat(),"status":"merged_archived_analysis_passed",
              "analysis_script_sha256":sha256(__file__),"original_reference_geometry_sha256":sha256(original_campaign/"geometry.json"),
              "of_prior_audit_sha256":sha256(args.of_analysis/"audit.json"),"old_archive_verified_used_files":old.checked,
              "new_archive_verified_used_files":new.checked,"qe_attempt_evidence":qe_evidence,"global_strain":tensors,
              "rows":rows,"all_attempt_diagnostics":diagnostics,"notes":notes})
    save_json(analysis/"output_sha256.json",{str(p.relative_to(args.output_root)).replace('\\','/'):sha256(p)
              for folder in (analysis,figures,structures) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='output_sha256.json'})
    print("DONE "+str(args.output_root),flush=True)


if __name__=="__main__":
    main()
