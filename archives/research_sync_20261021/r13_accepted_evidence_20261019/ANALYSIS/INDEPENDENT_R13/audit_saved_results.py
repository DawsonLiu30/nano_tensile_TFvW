"""Portable read-only audit. Run with case archive root as sole argument.

Only saved arrays and ASE SinglePointCalculator results are used; no DFT.
The frozen ASE 3.25 filter source is preserved beside this script.
"""
from pathlib import Path
import csv, hashlib, importlib.util, json, sys
import numpy as np
from ase.io import read
from ase.units import Bohr
from ase.geometry import find_mic

HERE=Path(__file__).resolve().parent
D=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE.parents[1]
A=D/'R13_0V'; OUT=HERE
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
jl=lambda name:[json.loads(s) for s in (A/name).read_text().splitlines()]
checks=[]
def check(name, ok):
    checks.append({'name':name,'passed':bool(ok)})
    if not ok: raise ValueError(name)
def dump(name, data): (OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

r=load(A/'result.json'); cp=load(A/'checkpoint.json'); p=load(A/'provenance.json')['protocol']; state=load(A/'optimizer_state.json')
download=load(D/'REMOTE_FILE_MANIFEST.json'); refs=load(D/'REMOTE_REFERENCES.json')
hashmap={x['archive']:x['sha256'] for x in download+refs}
for row in download:
    path=D/row['archive']
    check('download:'+row['archive'],sha(path)==row['sha256'] and path.stat().st_size==row['bytes'] and row.get('stable_during_read',True))
for rel,h in r['artifact_sha256'].items():check('result_artifact:'+rel,hashmap['R13_0V/'+rel]==h)
check('accepted_endpoint',r['accepted'] and r['status']=='converged' and r['ionic_converged'])
check('core_frozen',sha(A/'inputs/recovery_runner.py')=='dc36263b50496c3b03822cbc94e3b1d7fd153d38a78dc547fd2b717d0ffe5b72')
check('revision_frozen',r['recovery_revision_manifest_sha256']==sha(A/'inputs/axial_revision_manifest.json')=='0873dc84e7e796fea4f743ab37e1f2dcc1a34dd86bee2cf9fef70cc7e3970bbf')
check('physical_settings',p['xc']=='LDA' and p['lambda_tf']==.9 and p['mu_vw']==.1 and p['grid']==[140,140,96])
check('strict_thresholds',p['ionic']['fmax_eV_A']==p['ionic']['filter_fmax_eV_A']==.001 and p['cell']['stress_max_abs_GPa']==.05)
check('electronic_settings',p['electronic']['method']=='TN' and p['electronic']['econv_Ha_per_atom']==1e-11 and p['electronic']['ncheck']==3)
check('axial_only',p['cell']['mask']==[0,0,1,0,0,0])
with np.load(A/'checkpoint_bundle.npz',allow_pickle=False) as z:b={k:z[k].copy() for k in z.files}
meta=json.loads(str(b['metadata_json'])); rho=b['density_electrons_bohr3']; cell=b['cell_A']; n=len(b['numbers'])
check('bundle_hash',sha(A/'checkpoint_bundle.npz')==cp['checkpoint_bundle_sha256'])
check('complete_step',meta['ionic_step']==cp['ionic_step']==r['ionic_steps']==46)
check('atom_identity',n==76 and np.all(b['numbers']==13) and len(set(b['parent_site_id']))==76 and r['vacancy_count']==0)
check('grid',rho.shape==(140,140,96) and b['grid'].tolist()==[140,140,96])
check('density_positive_finite',np.isfinite(rho).all() and np.all(rho>0))
ne=float(rho.mean()*np.linalg.det(b['lattice_bohr']));check('electron_count',abs(ne-228)<1e-6)
check('density_cell',np.allclose(b['lattice_bohr']*Bohr,cell,atol=1e-8,rtol=0))
check('full_pbc',np.all(b['pbc']) and np.diag(cell).min()>=10 and np.max(abs(cell-np.diag(np.diag(cell))))<1e-10)
check('checkpoint_force',abs(np.linalg.norm(b['forces_eV_A'],axis=1).max()-cp['fmax_eV_A'])<1e-12)
check('force_pass',r['fmax_eV_A']<=.001 and r['filter_fmax_eV_A']<=.001)
check('axial_stress_pass',abs(r['stress_GPa'][2][2])<=.05)
check('energy_component_sum',abs(sum(r['energy_components_eV'][k] for k in ['KEDF','XC','HARTREE','PSEUDO','II'])-r['energy_eV'])<1e-8)
check('kinetic_component_sum',abs(r['tf_energy_eV']+r['vw_energy_eV']-r['kinetic_energy_eV'])<1e-9)
traj=read(A/'relax.traj',index=':'); rows=jl('ionic_evaluations.jsonl'); ls=jl('ionic_line_search.jsonl'); elec=jl('electronic_evaluations.jsonl'); ev=jl('optimizer_events.jsonl')
check('47_completed_frames',len(traj)==len(rows)==47 and [x['ionic_step'] for x in rows]==list(range(47)))
spec=importlib.util.spec_from_file_location('frozen_filters',HERE/'ase.filters.py'); ff=importlib.util.module_from_spec(spec);spec.loader.exec_module(ff)
xs=[];gs=[];es=[];periodic_dist=[];disps=[]
origin=traj[0].cell.array.copy()
for i,(a,row) in enumerate(zip(traj,rows)):
    check(f'frame{i}:energy',abs(a.get_potential_energy()-row['energy_eV'])<1e-9)
    check(f'frame{i}:force',abs(np.linalg.norm(a.get_forces(),axis=1).max()-row['fmax_eV_A'])<1e-12)
    check(f'frame{i}:stress',np.allclose(a.get_stress(voigt=False),row['stress_eV_A3'],atol=1e-12,rtol=0))
    check(f'frame{i}:cell',np.allclose(a.cell.array,row['cell_A'],atol=1e-12,rtol=0))
    check(f'frame{i}:transverse_fixed',np.max(abs(a.cell.array[:2]-origin[:2]))<1e-12 and a.cell[2,0]==a.cell[2,1]==0)
    filt=ff.FrechetCellFilter(a,mask=[0,0,1,0,0,0],exp_cell_factor=float(n));filt.orig_cell=origin.copy()
    x=filt.get_positions().ravel();g=-filt.get_forces().ravel();xs.append(x);gs.append(g);es.append(a.get_potential_energy())
    check(f'frame{i}:filter',abs(np.linalg.norm(g.reshape(-1,3),axis=1).max()-row['filter_fmax_eV_A'])<1e-10)
    dist=a.get_all_distances(mic=True);dist[np.diag_indices(n)]=np.inf;periodic_dist.append(float(dist.min()))
    check(f'frame{i}:no_overlap',periodic_dist[-1]>1.5)
    mapped=traj[0].positions@np.linalg.solve(origin,a.cell.array);d=find_mic(a.positions-mapped,a.cell,a.pbc)[0];disps.append(np.linalg.norm(d,axis=1))
xs=np.array(xs);gs=np.array(gs);es=np.array(es);disps=np.array(disps)
for name in ['final.traj','checkpoint.traj','final.extxyz','final.vasp']:
    a=read(A/name);check('final_structure:'+name,np.allclose(a.positions,b['positions_A'],atol=1e-7,rtol=0) and np.allclose(a.cell.array,cell,atol=1e-7,rtol=0))
check('final_force_array',np.allclose(np.load(A/'final_forces_eV_A.npy'),b['forces_eV_A'],atol=1e-12,rtol=0))
check('final_energy',abs(es[-1]-r['energy_eV'])<1e-9 and abs(cp['energy_eV']-r['energy_eV'])<1e-9)
check('state_step_hash',state['checkpoint_ionic_step']==state['state']['nsteps']==46 and state['checkpoint_bundle_sha256']==sha(A/'checkpoint_bundle.npz'))
check('state_energy',abs(state['checkpoint_energy_eV']-r['energy_eV'])<1e-12)
check('state_geometry',np.array_equal(state['positions_A'],b['positions_A']) and np.array_equal(state['cell_A'],cell))
check('state_origin',np.array_equal(state['orig_cell_A'],origin) and state['exp_cell_factor']==n and state['cell_factor']==n)
check('state_generalized_geometry',np.max(abs(np.asarray(state['generalized_x']).ravel()-xs[-1]))<1e-10)
check('state_previous_geometry',np.max(abs(np.asarray(state['state']['r0']).ravel()-xs[-2]))<1e-10)
check('state_previous_gradient',np.max(abs(np.asarray(state['state']['g0']).ravel()*state['state']['alpha']-gs[-2]))<1e-5)
h=np.asarray(state['state']['H']);check('state_hessian_shape_finite',h.shape==(3*(n+3),3*(n+3)) and np.isfinite(h).all())
check('state_hessian_symmetric',np.max(abs(h-h.T))<1e-10)
check('state_saved_all_completed_steps',len([x for x in ev if x.get('event')=='optimizer_state_saved'])==47)
check('185_electronic_all_native_success',len(elec)==185 and all(x['accepted'] and x['converged_flag']==0 for x in elec))
check('all_electronic_grid_ne',all(x['grid_shape']==[140,140,96] and abs(x['density_integral_electrons']-228)<1e-6 for x in elec))
check('no_retry',r['electronic_attempt_summary']['physical_attempts']==185 and r['electronic_attempt_summary']['failed_physical_attempts']==r['electronic_attempt_summary']['HEG_retries']==0)
outer=[x for x in ls if x['kind']=='line_search_result'];check('46_native_success_searches',len(outer)==46 and all(x['task']=='CONVERGENCE' and not x.get('error') and x['alpha'] is not None for x in outer))
check('all_steps_energy_descending',np.all(np.diff(es)<0))
parent=load(A/'inputs/parent_result.json');initial=read(A/'initial.traj')
check('parent_warm_restart_geometry',np.allclose(origin,initial.cell.array,atol=1e-12,rtol=0) and np.allclose(traj[0].positions,initial.positions,atol=1e-12,rtol=0))
check('parent_bundle_provenance',parent['artifact_sha256']['checkpoint_bundle.npz']==r['parent_checkpoint']['bundle_sha256']==hashmap['R13_0V/inputs/initial_checkpoint_bundle.npz'])
step_rows=[]
for i in range(46):
    dx=xs[i+1]-xs[i];g0=float(gs[i]@dx);g1=float(gs[i+1]@dx);de=es[i+1]-es[i]
    armijo=.23*g0-de;curv=abs(g1)/max(abs(g0),1e-30)
    # Saved forces and re-evaluated initial SCF can differ at ~1e-9 eV; tolerance is recorded.
    check(f'step{i+1}:descent',g0<0)
    check(f'step{i+1}:Armijo',armijo>=-1e-7)
    check(f'step{i+1}:curvature',curv<=.46+1e-3)
    step_rows.append({'step':i+1,'energy_eV':float(es[i+1]),'delta_energy_meV':float(de*1000),'cumulative_energy_meV':float((es[i+1]-es[0])*1000),'fmax_eV_A':rows[i+1]['fmax_eV_A'],'filter_fmax_eV_A':rows[i+1]['filter_fmax_eV_A'],'sigma_zz_GPa':rows[i+1]['stress_GPa'][2][2],'Lz_A':rows[i+1]['cell_A'][2][2],'armijo_margin_meV':armijo*1000,'curvature_ratio':curv,'trapezoid_residual_meV':float((de-.5*(g0+g1))*1000),'min_pair_A':periodic_dist[i+1],'nonaffine_max_from_start_A':float(disps[i+1].max()),'nonaffine_rms_from_start_A':float(np.sqrt(np.mean(disps[i+1]**2))),'energy_trial_count':sum(x['ionic_step']==i and x['kind']=='energy_trial' for x in ls),'gradient_trial_count':sum(x['ionic_step']==i and x['kind']=='gradient_trial' for x in ls)})
with (OUT/'completed_step_diagnostics.csv').open('w',newline='',encoding='utf8') as f:
    wr=csv.DictWriter(f,fieldnames=list(step_rows[0]));wr.writeheader();wr.writerows(step_rows)
with (OUT/'trajectory_metrics.csv').open('w',newline='',encoding='utf8') as f:
    wr=csv.writer(f);wr.writerow(['step','energy_eV','cumulative_energy_meV','fmax_eV_A','filter_fmax_eV_A','sigma_zz_GPa','Lz_A','nonaffine_max_from_start_A','nonaffine_rms_from_start_A'])
    for i,row in enumerate(rows):wr.writerow([i,es[i],(es[i]-es[0])*1000,row['fmax_eV_A'],row['filter_fmax_eV_A'],row['stress_GPa'][2][2],row['cell_A'][2][2],disps[i].max(),np.sqrt(np.mean(disps[i]**2))])
top=sorted(step_rows,key=lambda x:x['delta_energy_meV'])[:8]
summary={'job_id':'2255218','accepted_by_saved_data_audit':True,'scope':'Strict numerical endpoint acceptance under the recorded model; not evidence of retained FCC morphology, k/grid/vacuum convergence, or a global minimum','completed_steps':46,'electronic_calls':185,'initial_energy_eV':float(es[0]),'final_energy_eV':float(es[-1]),'energy_change_meV':float((es[-1]-es[0])*1000),'initial_fmax_eV_A':rows[0]['fmax_eV_A'],'final_fmax_eV_A':r['fmax_eV_A'],'final_filter_fmax_eV_A':r['filter_fmax_eV_A'],'final_sigma_zz_GPa':r['stress_GPa'][2][2],'initial_Lz_A':float(origin[2,2]),'final_Lz_A':float(cell[2,2]),'axial_change_percent':float((cell[2,2]/origin[2,2]-1)*100),'maximum_fmax_along_path_eV_A':max(x['fmax_eV_A'] for x in rows),'minimum_pair_A_along_path':min(periodic_dist),'final_pair_min_A':periodic_dist[-1],'nonaffine_displacement_max_A':float(disps[-1].max()),'nonaffine_displacement_rms_A':float(np.sqrt(np.mean(disps[-1]**2))),'largest_energy_drop_steps':top,'last_5_steps_energy_change_meV':float((es[-1]-es[-6])*1000),'last_10_steps_energy_change_meV':float((es[-1]-es[-11])*1000),'density_integral_electrons':ne,'density_min':float(rho.min()),'state_hessian_min_eigenvalue':float(np.linalg.eigvalsh(h).min()),'all_46_steps_independent_wolfe_pass':True,'source_bundle_sha256':sha(A/'checkpoint_bundle.npz'),'source_optimizer_state_sha256':sha(A/'optimizer_state.json'),'canonical_change_recommendation':'May upgrade R13_0V numerical acceptance; separately review R13_0V versus R13_1V structural comparability before interpreting a five-radius vacancy trend.'}
dump('audit_summary.json',summary);dump('QA.json',{'passed':sum(x['passed'] for x in checks),'total':len(checks),'checks':checks,'scope':'Saved-data validation, not additional DFT calculations.'})
print(json.dumps({'checks':len(checks),'summary':summary},indent=2))
