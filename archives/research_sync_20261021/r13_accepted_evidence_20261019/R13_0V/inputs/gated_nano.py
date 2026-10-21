"""Finite, audited gate then independent corrected-stress nano relaxations.

validate: R15 0V/1V total frozen-density + self-consistent axial derivatives,
unchanged-force derivative spot checks, corrected-stress warm relaxations,
and paired energy/geometry stability. Produces gate_pass.json only on success.
run: validates the gate and immutable inputs before one independent case.
No queue submission, retry, or installed package changes occur in this file.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['MPLBACKEND']='Agg'
import argparse, copy, hashlib, json, math, sys, time, traceback, subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from ase.io import read
from ase.geometry import find_mic
from ase.units import Bohr,Hartree
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.ions import Ions
from dftpy.interface import ConfigParser
from dftpy.optimization.optimization import Optimization
import base_runner as base
from stress_candidate import install
ROOT=Path(__file__).resolve().parents[1];CAM=ROOT/'campaign'
GPA=Hartree/Bohr**3*160.21766208

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):base.dump(p,d)
def verify_package():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    for rel,h in manifest.items():
        if sha(ROOT/rel)!=h:raise ValueError('Package hash mismatch: '+rel)
    return sha(ROOT/'MANIFEST.json')
def get_seed(cid):
    sd=CAM/'seeds'/cid;z=np.load(sd/'final_density.npz');a=read(sd/'final.extxyz')
    if np.max(abs(z['lattice_bohr']-np.asarray(a.cell)/Bohr))>1e-6:raise ValueError('Seed density/cell mismatch')
    grid=DirectGrid(lattice=np.asarray(a.cell)/Bohr,nr=z['density_electrons_bohr3'].shape)
    rho=DirectField(grid=grid,griddata_3d=z['density_electrons_bohr3'])
    if abs(float(rho.integral())-3*len(a))>1e-6:raise ValueError('Seed electron mismatch')
    return a,rho
def prep(cid):
    checked,a,snap=base.preflight(CAM,cid)
    p=base.contained_path(CAM,checked['protocol']['pseudo']['file'])
    cfg=base.make_config(a,p,checked['protocol'])
    if cfg['EXC']['xc']!='LDA' or cfg['KEDF']['x']!=.9 or cfg['KEDF']['y']!=.1 or cfg['KEDF']['sigma'] is not None:
        raise ValueError('Physical protocol changed')
    for name,p in {'stress_candidate.py':Path(__file__).with_name('stress_candidate.py'),
                   'gated_nano.py':Path(__file__),'gate.json':CAM/'gate.json','package_manifest.json':ROOT/'MANIFEST.json'}.items():
        snap[name]=p;checked['input_sha256'][name]=sha(p)
    checked['stress_override']='process-local positive-density sigma=None discrete vW derivative'
    checked['package_manifest_sha256']=sha(ROOT/'MANIFEST.json')
    return checked,a,snap,cfg
def evaluator(a,rho,cfg):
    cfg,o=ConfigParser(copy.deepcopy(cfg),ions=Ions.from_ase(a),rhoini=rho,grid=rho.grid)
    return o['E_v_Evaluator'],o['rho']
def self_consistent(a,initial,cfg,out):
    ev,rho=evaluator(a,initial,cfg)
    optcfg=copy.deepcopy(cfg['OPT']);optcfg['econv']*=len(a)
    opt=Optimization(EnergyEvaluator=ev,optimization_method='LBFGS',optimization_options=optcfg)
    t=time.time();rho=opt.optimize_rho(guess_rho=rho)
    event={'converged_flag':int(opt.converged),'elapsed_s':time.time()-t,'grid':list(map(int,rho.grid.nr)),
           'electrons':float(rho.integral()),'effective_econv_Ha_total':optcfg['econv']}
    dump(out.with_suffix('.electronic.json'),event)
    if event['converged_flag']!=0 or abs(event['electrons']-3*len(a))>1e-6:
        raise RuntimeError('Self-consistent diagnostic failed electronic acceptance')
    ens=ev.get_energy_potential(rho,calcType={'E'},split=True)
    ss=ev.get_stress(rho,split=True);ff=ev.get_forces(rho,split=True)
    result={'energy_eV':float(ens['TOTAL'].energy)*Hartree,'stress_GPa':(ss['TOTAL']*GPA).tolist(),
            'forces_eV_A':(ff['TOTAL']*Hartree/Bohr).tolist(),'event':event}
    if not all(np.all(np.isfinite(np.asarray(result[k]))) for k in ['energy_eV','stress_GPa','forces_eV_A']):
        raise ValueError('Nonfinite SCF diagnostic')
    dump(out.with_suffix('.json'),result)
    return result,rho
def deform(a,rho,eps):
    b=a.copy();cell=np.asarray(a.cell).copy();cell[2]*=1+eps;b.set_cell(cell,scale_atoms=True)
    g=DirectGrid(lattice=np.asarray(b.cell)/Bohr,nr=rho.grid.nr)
    return b,DirectField(grid=g,griddata_3d=np.asarray(rho)/(1+eps))
def force_spot(a,rho,cfg,forces,step):
    f=np.asarray(forces);i,j=np.unravel_index(np.argmax(abs(f)),f.shape);ene=[]
    for sign in [-1,1]:
        b=a.copy();b.positions[i,j]+=sign*step
        ev,rr=evaluator(b,rho,cfg);ene.append(float(ev.get_energy_potential(rr,calcType={'E'}).energy)*Hartree)
    fd=-(ene[1]-ene[0])/(2*step)
    return {'atom_index':int(i),'parent_site_id':int(a.arrays['parent_site_id'][i]),'axis':int(j),
            'step_A':step,'analytic_eV_A':float(f[i,j]),'fd_eV_A':fd,'abs_error_eV_A':abs(fd-f[i,j]),
            'scope':'One force component per R15 endpoint; fixed saved SCF density; force implementation is unchanged by stress override'}
def final_periodic_geometry(a):
    """Validate final actual lattice, not merely the initial requested period."""
    cell=np.asarray(a.cell,dtype=float)
    if not np.all(a.pbc):raise ValueError('Final structure lost three-direction PBC')
    if not np.all(np.isfinite(cell)) or np.any(np.diag(cell)<=0):raise ValueError('Invalid final cell')
    if np.max(abs(cell-np.diag(np.diag(cell))))>1e-8:
        raise ValueError('Expected orthogonal final nanocolumn cell')
    shortest=float(np.min(np.diag(cell)))
    return {'minimum_periodic_translation_A':shortest,'axial_period_A':float(cell[2,2]),
            'minimum_required_A':10.,'accepted':bool(shortest>=10.)}
def run_relax(cid,runroot,budget,rho=None,gate_path=None,warm_path=None):
    checked,a,snap,cfg=prep(cid)
    if rho is None and (CAM/'seeds'/cid).exists():
        _,rho=get_seed(cid);warm_path=CAM/'seeds'/cid/'final_density.npz'
    # Save the exact warm seed as part of the attempt's auditable inputs.
    if warm_path:
        p=Path(warm_path);snap['initial_density.npz']=p;checked['input_sha256']['initial_density.npz']=sha(p)
    if gate_path:
        p=Path(gate_path);snap['validation_gate_pass.json']=p;checked['input_sha256']['validation_gate_pass.json']=sha(p)
    checked['density_initialization']='saved density warm restart' if rho is not None else 'fresh HEG'
    import dftpy.api.api4ase as api
    original=api.DFTpyCalculator
    class WarmCalculator(original):
        def __init__(self,*args,**kw):
            if rho is not None:kw['rho']=rho.copy()
            super().__init__(*args,**kw)
    api.DFTpyCalculator=WarmCalculator
    args=SimpleNamespace(output_root=Path(runroot),case=cid,attempt='attempt_001',campaign=CAM,max_runtime_s=budget)
    try:
        code=base.run(args,checked,a,snap,cfg)
        if code==0:
            attempt=Path(runroot)/cid/args.attempt
            resultpath=attempt/'result.json';result=json.loads(resultpath.read_text())
            try:
                final=read(attempt/'final.extxyz');geometry_gate=final_periodic_geometry(final)
                result['final_periodic_geometry_gate']=geometry_gate
                if not geometry_gate['accepted']:raise ValueError('Final periodic image separation is below 10 A')
            except Exception as exc:
                result.update(accepted=False,status='final_periodic_geometry_rejected',error=repr(exc))
                code=1
            dump(resultpath,result)
        return code
    finally:api.DFTpyCalculator=original
def stability(cid,resultdir,gate):
    sd=CAM/'seeds'/cid;old=json.loads((sd/'result.json').read_text());new=json.loads((resultdir/'result.json').read_text())
    a=read(sd/'final.extxyz');b=read(resultdir/'final.extxyz')
    mapped=a.get_scaled_positions()@np.asarray(b.cell)
    u=find_mic(b.positions-mapped,b.cell,b.pbc)[0];u-=u.mean(axis=0)
    r={'case':cid,'accepted_endpoint':new['accepted'],'delta_energy_eV':new['energy_eV']-old['energy_eV'],
       'delta_Lz_A':float(b.cell[2,2]-a.cell[2,2]),'nonaffine_rms_drift_A':float(np.sqrt(np.mean(np.sum(u*u,axis=1)))),
       'fmax_eV_A':new['fmax_eV_A'],'filter_fmax_eV_A':new['filter_fmax_eV_A'],'sigma_zz_GPa':new['stress_GPa'][2][2],
       'result_sha256':sha(resultdir/'result.json')}
    r['accepted']=bool(r['accepted_endpoint'] and abs(r['delta_Lz_A'])<=gate['R15_cell_drift_max_A'] and r['nonaffine_rms_drift_A']<=gate['R15_rms_position_drift_max_A'])
    return r
def validate(args,packhash,gate):
    root=base.validate_output_root(args.output_root);root.mkdir(parents=True,exist_ok=True)
    vd=root/'validation';vd.mkdir(exist_ok=False);deadline=time.monotonic()+args.max_runtime_s
    records=[]
    try:
      for cid in gate['R15_validation_cases']:
        caseout=vd/cid;caseout.mkdir()
        with (caseout/'frozen_total_stdout.log').open('w') as log:
            subprocess.run([sys.executable,str(Path(__file__).with_name('validate_saved_total.py')),
                            '--case-dir',str(CAM/'seeds'/cid),'--output',str(caseout/'frozen_total.json')],
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,deadline-time.monotonic()))
        checked,a,snap,cfg=prep(cid);a,rho=get_seed(cid)
        with base.budget_guard(max(1,deadline-time.monotonic())),base.capture(caseout/'self_consistent_stdout.log'):
            origin,rho0=self_consistent(a,rho,cfg,caseout/'scf_0')
            base.save_density(caseout/'scf_0_density.npz',rho0)
            rows=[]
            for h in gate['scf_strains']:
                energies=[]
                for sign in [-1,1]:
                    b,rr=deform(a,rho0,sign*h)
                    res,_=self_consistent(b,rr,cfg,caseout/f'scf_{sign*h:+.6f}')
                    energies.append(res['energy_eV'])
                fd=(energies[1]-energies[0])/(2*h*a.get_volume())*160.21766208
                rows.append({'strain':h,'fd_total_sigma_zz_GPa':fd,'analytic_sigma_zz_GPa':origin['stress_GPa'][2][2],
                             'abs_error_GPa':abs(fd-origin['stress_GPa'][2][2])})
            force=force_spot(a,rho0,cfg,origin['forces_eV_A'],gate['force_fd_step_A'])
        passed=bool(all(r['abs_error_GPa']<=gate['scf_axial_fd_tolerance_GPa'] for r in rows)
                    and abs(rows[0]['fd_total_sigma_zz_GPa']-rows[1]['fd_total_sigma_zz_GPa'])<=gate['scf_fd_step_stability_GPa']
                    and force['abs_error_eV_A']<=gate['force_fd_tolerance_eV_A'])
        r={'case':cid,'self_consistent_total_axial_rows':rows,'force_spot_check':force,'accepted':passed}
        dump(caseout/'derivative_gate.json',r);records.append(r)
        if not passed:raise RuntimeError('Self-consistent stress/force diagnostic rejected '+cid)
        # Same verified production thresholds and fixed transverse cell.
        if run_relax(cid,root,max(1,deadline-time.monotonic()),rho=rho0,warm_path=caseout/'scf_0_density.npz')!=0:
            raise RuntimeError('Corrected warm relaxation rejected '+cid)
        r['warm_relax_stability']=stability(cid,root/cid/'attempt_001',gate)
        dump(caseout/'derivative_gate.json',r)
        if not r['warm_relax_stability']['accepted']:raise RuntimeError('Reference geometry stability gate failed '+cid)
      drift=records[1]['warm_relax_stability']['delta_energy_eV']-records[0]['warm_relax_stability']['delta_energy_eV']
      passed=bool(abs(drift)<=gate['R15_pair_Ef_drift_max_eV'])
      result={'accepted':passed,'scope':'R15 pair: frozen and self-consistent total axial derivatives; unchanged-force spot derivatives; corrected warm axial relax and same-bulk-mu Ef drift',
              'package_manifest_sha256':packhash,'candidate_sha256':sha(Path(__file__).with_name('stress_candidate.py')),
              'R15_pair_formation_energy_drift_eV':drift,'records':records,'completed_utc':base.utc()}
      dump(vd/'gate_result.json',result)
      if not passed:raise RuntimeError('Paired formation-energy stability gate failed')
      dump(root/'gate_pass.json',result)
      return 0
    except BaseException as exc:
      dump(vd/'gate_failure.json',{'accepted':False,'error':repr(exc),'traceback':traceback.format_exc(),'records':records,'utc':base.utc()})
      raise
def verify_gate(path,packhash,settings):
    path=Path(path);g=json.loads(path.read_text())
    if (g.get('accepted') is not True or g.get('package_manifest_sha256')!=packhash
        or g.get('candidate_sha256')!=sha(Path(__file__).with_name('stress_candidate.py'))):
        raise ValueError('Gate missing, failed, or mismatched frozen package')
    records=g.get('records',[])
    if [r.get('case') for r in records]!=settings['R15_validation_cases']:
        raise ValueError('Gate must contain both R15 references in order')
    if abs(g['R15_pair_formation_energy_drift_eV'])>settings['R15_pair_Ef_drift_max_eV']:
        raise ValueError('Pair drift failed')
    for rec in records:
        st=rec['warm_relax_stability'];resultpath=path.parent/rec['case']/'attempt_001/result.json'
        if rec.get('accepted') is not True or st.get('accepted') is not True or sha(resultpath)!=st['result_sha256']:
            raise ValueError('Reference gate record changed or rejected')
        endpoint=json.loads(resultpath.read_text())
        if endpoint.get('accepted') is not True:raise ValueError('Reference endpoint rejected')
    return g
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['dry','validate','run']);p.add_argument('--case')
    p.add_argument('--output-root',type=Path);p.add_argument('--gate',type=Path);p.add_argument('--max-runtime-s',type=float,default=6600)
    args=p.parse_args();packhash=verify_package();gate=json.loads((CAM/'gate.json').read_text());install()
    if args.mode=='dry':
        cases=json.loads((CAM/'dftpy_cases.json').read_text());reports=[]
        for cid in cases:
            checked,a,_,cfg=prep(cid)
            if a.cell[2,2]<10:raise ValueError('Axial period <10 A')
            reports.append({'case':cid,'n_atoms':len(a),'grid':cfg['GRID']['nr'],'initial_spacing_A':checked['initial_grid_spacing_A'],'warm_restart':(CAM/'seeds'/cid).exists()})
        print(json.dumps({'accepted':True,'package_manifest_sha256':packhash,'cases':reports},indent=2));return 0
    if args.output_root is None:raise ValueError('output-root is required')
    if args.mode=='validate':return validate(args,packhash,gate)
    if args.case not in gate['production_tasks']:raise ValueError('Unknown production case')
    if args.gate is None:raise ValueError('Accepted validation gate required')
    verify_gate(args.gate,packhash,gate)
    return run_relax(args.case,args.output_root,args.max_runtime_s,gate_path=args.gate)
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print('[REJECTED] '+repr(exc),file=sys.stderr);raise
