"""Bounded R13 electronic/force gate then an independently guarded ionic finish.

verify never computes DFT. diagnose <=8 logical/14 native SCFs; run only after
revalidating the completed gate. Original sources and parent results are immutable.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['MPLBACKEND']='Agg';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse, copy, hashlib, importlib.util, json, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from ase import Atoms
from ase.units import Bohr,Hartree
from electronic_trace import ElectronicTrace,validate_density,make_heg
from staged_optimizer import POLICY

HERE=Path(__file__).resolve().parent
EXPECTED={'case':'R13_0V','parent_manifest':'e2a78d9c360f5fbef8ec267a0565e2f74d17542e5641262071cf3bbb5fbe1674',
 'parent_gate':'b1ddc3cb06fed8e341eaf2197e9436c5841e25f6f34236a056ca70527374acd0',
 'smallstep_revision':'fff9270f92d11dbdd820150675196eb14ed75dc9c6eb5890febeb8ec0e05c0e1',
 'parent_result':'45fc08cf8cbeb527428f9e2495868d3033dc452eed8737432784549cd46731f3',
 'bundle':'93398e09a8470c502a54d0f67053af3e50ed19ae03f8225826c533780c71631f',
 'stress_candidate':'06c86e2aa20408042746f0d457765991ca91afe1ebf20737562108928b0acfeb'}
SETTINGS={'candidate_method':'TN','candidate_econv_Ha_per_atom':1e-11,'ncheck':3,
 'repeat_energy_max_eV':1e-5,'repeat_force_max_eV_A':1e-4,'repeat_sigma_zz_max_GPa':.001,
 'force_fd_error_max_eV_A':1e-4,'force_fd_step_difference_max_eV_A':5e-5,
 'force_fd_steps_A':[.001,.0005],'force_parent_site_id':58,'force_axis':2,
 'logical_scf_max':8,'physical_scf_max':14,'diagnostic_seconds_max':6600,
 'ionic_steps_max':1000,'ionic_seconds_max':79200,'force_target_eV_A':.001,'sigma_zz_target_GPa':.05,
 'retry_policy':'TN warm native flag=1 -> at most one independent HEG attempt; all other failures stop'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def revision():
    m=load(HERE/'REVISION_MANIFEST.json');require(m['expected']==EXPECTED and m['settings']==SETTINGS and m['ionic_policy']==POLICY,'Revision policy changed')
    for rel,h in m['files'].items():
        p=(HERE/rel).resolve();require(p.is_relative_to(HERE) and sha(p)==h,'Revision file changed: '+rel)
    return sha(HERE/'REVISION_MANIFEST.json')

def load_bundle(path,geometry,checkpoint,result):
    with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    need={'metadata_json','positions_A','cell_A','numbers','pbc','parent_site_id','forces_eV_A','density_electrons_bohr3','lattice_bohr','grid','origin_bohr'}
    require(need.issubset(d),'Incomplete checkpoint bundle');m=json.loads(str(d['metadata_json']))
    require(m['ionic_step']==checkpoint['ionic_step']==100 and result['ionic_steps']==101,'Wrong completed/proposed step')
    require(result['accepted'] is False and result['status']=='failed' and result['electronic_calls']==102,'Wrong parent failure')
    require(result['recovery_revision_manifest_sha256']==EXPECTED['smallstep_revision'],'Wrong smallstep source revision')
    require(len(d['numbers'])==76 and np.all(d['numbers']==13) and np.all(d['pbc']),'Wrong atoms/PBC')
    require(d['positions_A'].shape==(76,3) and d['forces_eV_A'].shape==(76,3),'Wrong atomic shape')
    require(d['parent_site_id'].tolist()==geometry['remaining_parent_ids'],'Parent site identity changed')
    require(d['grid'].tolist()==[140,140,96] and d['density_electrons_bohr3'].shape==(140,140,96),'Grid changed')
    for k in need-{'metadata_json'}:require(np.isfinite(d[k]).all(),'Nonfinite bundle '+k)
    cell=d['cell_A'];require(cell.shape==(3,3) and np.min(np.diag(cell))>=10 and np.max(abs(cell-np.diag(np.diag(cell))))<1e-8,'Cell/PBC length invalid')
    require(np.allclose(d['lattice_bohr']*Bohr,cell,atol=1e-8,rtol=0) and np.allclose(d['origin_bohr'],0,atol=1e-12),'Density lattice/origin mismatch')
    require(np.all(d['density_electrons_bohr3']>0),'Nonpositive density')
    ne=float(d['density_electrons_bohr3'].mean()*np.linalg.det(d['lattice_bohr']));require(abs(ne-228)<1e-6,'Wrong electron count')
    for k in ('ionic_step','cell_A','energy_eV','fmax_eV_A','filter_fmax_eV_A','stress_GPa'):require(m[k]==checkpoint[k],'Metadata mismatch '+k)
    require(np.array_equal(cell,np.asarray(m['cell_A'])) and abs(np.linalg.norm(d['forces_eV_A'],axis=1).max()-m['fmax_eV_A'])<1e-12,'Metadata force/cell mismatch')
    require(np.allclose(cell,np.asarray(result['initial_cell_A']),atol=1e-12,rtol=0),'Parent fixed cell changed')
    frozen=np.ones((3,3),dtype=bool);frozen[2,2]=False
    require(np.allclose(cell[frozen],np.asarray(geometry['cell_A'])[frozen],atol=1e-8,rtol=0),'Transverse cell changed')
    require(abs(m['stress_GPa'][2][2])<=.05 and max(m['fmax_eV_A'],m['filter_fmax_eV_A'])<=.01,'Not the agreed low-force checkpoint')
    atoms=Atoms(numbers=d['numbers'],positions=d['positions_A'],cell=cell,pbc=d['pbc']);atoms.set_array('parent_site_id',d['parent_site_id'])
    return atoms,d,m

def prepare(args):
    rh=revision();p=args.parent_snapshot.resolve();pa=args.parent_attempt.resolve()
    require(sha(p/'MANIFEST.json')==EXPECTED['parent_manifest'],'Parent snapshot changed')
    sys.path.insert(0,str(p/'scripts'));g=module('gated_nano',p/'scripts/gated_nano.py')
    require(g.verify_package()==EXPECTED['parent_manifest'],'Frozen package mismatch')
    require(sha(p/'scripts/stress_candidate.py')==EXPECTED['stress_candidate'],'Stress candidate changed')
    gate=args.parent_runroot/'gate_pass.json';require(sha(gate)==EXPECTED['parent_gate'],'Original stress gate changed')
    g.verify_gate(gate,EXPECTED['parent_manifest'],load(p/'campaign/gate.json'))
    require(sha(pa/'result.json')==EXPECTED['parent_result'],'Parent result changed');result=load(pa/'result.json')
    for name in ['config.json','checkpoint.json','checkpoint_bundle.npz','inputs/recovery_revision_manifest.json']:
        require(sha(pa/name)==result['artifact_sha256'][name],'Parent artifact changed '+name)
    require(sha(pa/'checkpoint_bundle.npz')==EXPECTED['bundle'],'Parent bundle changed')
    require(load(pa/'checkpoint.json')['checkpoint_bundle_sha256']==EXPECTED['bundle'],'Parent bundle/header mismatch')
    checked,_,snap,cfg=g.prep('R13_0V');protocol=checked['protocol']
    require(protocol['xc']=='LDA' and protocol['lambda_tf']==.9 and protocol['mu_vw']==.1 and protocol['grid']==[140,140,96],'Physical model changed')
    require(protocol['ionic']==dict(optimizer='LBFGS',fmax_eV_A=.001,filter_fmax_eV_A=.001,max_steps=1000,maxstep_A=.15),'Ionic acceptance changed')
    require(protocol['cell']['mask']==[0,0,1,0,0,0] and protocol['cell']['stress_max_abs_GPa']==.05,'Cell policy changed')
    oldcfg=load(pa/'config.json');comparison=copy.deepcopy(cfg);comparison['PATH']['pppath']=oldcfg['PATH']['pppath']
    require(comparison==oldcfg,'Parent electronic configuration changed')
    require(cfg['OPT']['method']=='LBFGS' and cfg['OPT']['econv']==1e-9 and cfg['OPT']['ncheck']==3,'Wrong reference electronic settings')
    atoms,d,meta=load_bundle(pa/'checkpoint_bundle.npz',checked['geometry_case'],load(pa/'checkpoint.json'),result)
    g.install()
    report={'accepted':True,'scope':'preflight only; no DFT','revision_manifest_sha256':rh,'parent_bundle_sha256':EXPECTED['bundle'],'completed_ionic_step':100,'candidate':SETTINGS,'cell_A':atoms.cell.array.tolist(),'electrons':228,'n_atoms':76,'same_physical_model':True}
    return g,checked,atoms,snap,cfg,d,meta,report

def field(d):
    from dftpy.grid import DirectGrid
    from dftpy.field import DirectField
    grid=DirectGrid(lattice=d['lattice_bohr'],nr=d['grid'],origin=d['origin_bohr'])
    return DirectField(grid=grid,griddata_3d=d['density_electrons_bohr3'])

def scf(g,a,rho,cfg,out,trace,label,mode):
    from dftpy.optimization.optimization import Optimization
    trace.label=label;trace.initial_mode=mode;out=Path(out);out.mkdir(exist_ok=False)
    g.dump(out/'config.json',cfg);started=time.monotonic()
    try:
        with g.base.capture(out/'stdout.log'):
            ev,rr=g.evaluator(a,rho.copy(),cfg)
            options=copy.deepcopy(cfg['OPT']);options['econv']*=len(a)
            opt=Optimization(EnergyEvaluator=ev,optimization_method=cfg['OPT']['method'],optimization_options=options)
            rr=opt.optimize_rho(guess_rho=rr)
            validate_density(rr,len(a),cfg['GRID']['nr'],a.cell.array)
            require(opt.converged==0,'Native flag must be zero before force/stress')
            ens=ev.get_energy_potential(rr,calcType={'E'},split=True);fs=ev.get_forces(rr,split=True);ss=ev.get_stress(rr,split=True)
            data={'accepted':True,'native_converged_flag':int(opt.converged),'label':label,'initial_density':mode,'method':cfg['OPT']['method'],'econv_Ha_per_atom':cfg['OPT']['econv'],'ncheck':cfg['OPT']['ncheck'],
                  'energy_eV':float(ens['TOTAL'].energy)*Hartree,'forces_eV_A':(fs['TOTAL']*Hartree/Bohr).tolist(),'stress_GPa':(ss['TOTAL']*g.GPA).tolist(),'cell_A':a.cell.array.tolist(),'positions_A':a.positions.tolist(),'grid':cfg['GRID']['nr'],'electrons':float(rr.integral())}
            require(all(np.isfinite(np.asarray(data[k])).all() for k in ['energy_eV','forces_eV_A','stress_GPa']),'Nonfinite SCF result')
        data['physical_attempts']=[x for x in trace.records if x['label']==label]
        data['elapsed_s']=time.monotonic()-started;g.dump(out/'result.json',data)
        return data,rr
    except BaseException as exc:
        data={'accepted':False,'label':label,'method':cfg['OPT']['method'],'econv_Ha_per_atom':cfg['OPT']['econv'],'error':repr(exc),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-started,'physical_attempts':[x for x in trace.records if x['label']==label]}
        g.dump(out/'result.json',data);raise

def evaluate_gate(records):
    needed=['TN_reference','TN_tight_center','TN_tight_HEG','FD_m1000','FD_p1000','FD_m0500','FD_p0500']
    for name in needed:
        r=records[name];require(r['accepted'] is True and r['native_converged_flag']==0,'SCF rejected: '+name)
        require(r['grid']==[140,140,96] and abs(r['electrons']-228)<1e-6,'SCF shape/charge changed')
        require(r['ncheck']==3 and r['method']=='TN','SCF protocol changed')
        expected=1e-9 if name=='TN_reference' else 1e-11
        require(r['econv_Ha_per_atom']==expected,'SCF threshold changed')
        att=r['physical_attempts'];require(1<=len(att)<=2 and att[-1]['accepted'] is True and att[-1]['native_converged_flag']==0,'Missing native acceptance')
        if len(att)==2:require(att[0]['accepted'] is False and att[0]['native_converged_flag']==1 and att[0]['initial_density']=='warm' and att[1]['initial_density']=='HEG_retry','Invalid retry')
    c=records['TN_tight_center'];r=records['TN_tight_HEG'];ref=records['TN_reference']
    require(r['initial_density']=='HEG' and len(r['physical_attempts'])==1,'Independent repeat must start from HEG')
    for name in needed:
        require(np.array_equal(records[name]['cell_A'],c['cell_A']),'FD cell changed')
    for name in ('TN_reference','TN_tight_HEG'):require(np.array_equal(records[name]['positions_A'],c['positions_A']),'Center geometry changed')
    deltaE=abs(r['energy_eV']-c['energy_eV']);deltaF=float(np.linalg.norm(np.asarray(r['forces_eV_A'])-c['forces_eV_A'],axis=1).max());deltaS=abs(r['stress_GPa'][2][2]-c['stress_GPa'][2][2])
    require(deltaE<=SETTINGS['repeat_energy_max_eV'] and deltaF<=SETTINGS['repeat_force_max_eV_A'] and deltaS<=SETTINGS['repeat_sigma_zz_max_GPa'],'Independent density/force reproduction failed')
    i=c['force_atom_index'];j=SETTINGS['force_axis'];force=c['forces_eV_A'][i][j];rows=[]
    for h,suffix in zip(SETTINGS['force_fd_steps_A'],['1000','0500']):
        m=records['FD_m'+suffix];p=records['FD_p'+suffix]
        for sign,rec in [(-1,m),(1,p)]:
            target=np.array(c['positions_A']);target[i,j]+=sign*h
            require(np.allclose(target,rec['positions_A'],rtol=0,atol=1e-12),'FD displacement changed')
        fd=-(p['energy_eV']-m['energy_eV'])/(2*h)
        rows.append({'step_A':h,'parent_site_id':58,'atom_index':i,'axis':j,'analytic_force_eV_A':force,'finite_difference_force_eV_A':fd,'abs_error_eV_A':abs(fd-force)})
    difference=abs(rows[0]['finite_difference_force_eV_A']-rows[1]['finite_difference_force_eV_A'])
    require(all(x['abs_error_eV_A']<=SETTINGS['force_fd_error_max_eV_A'] for x in rows) and difference<=SETTINGS['force_fd_step_difference_max_eV_A'],'Force finite difference gate failed')
    require(abs(c['stress_GPa'][2][2])<=.05,'Fixed-cell finish entry axial stress rejected')
    return {'accepted':True,'repeat_delta_energy_eV':deltaE,'repeat_max_force_difference_eV_A':deltaF,'repeat_sigma_zz_difference_GPa':deltaS,'fd_rows':rows,'fd_step_difference_eV_A':difference,'TN_loose_to_tight_energy_difference_eV':c['energy_eV']-ref['energy_eV'],'TN_loose_to_tight_max_force_difference_eV_A':float(np.linalg.norm(np.asarray(c['forces_eV_A'])-ref['forces_eV_A'],axis=1).max())}

def diagnostic(args,prep):
    import fcntl
    g,checked,a,snap,cfg,d,meta,report=prep;root=g.base.validate_output_root(args.output_root)
    require(root==args.parent_runroot.resolve(),'Use the existing case lock/root')
    case=root/'R13_0V';case.mkdir(exist_ok=True,parents=True)
    with (case/'.execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not any(load(p).get('accepted') is True for p in case.glob('attempt*/result.json')),'Already accepted: refuse duplicate diagnosis')
        out=case/'diagnostics_003_electronic';out.mkdir(exist_ok=False)
        g.dump(out/'preflight.json',report);rho=field(d);trace=ElectronicTrace(out/'trace',len(a),cfg['GRID']['nr'],a.cell.array);records={};start=time.monotonic()
        try:
            with g.base.budget_guard(min(args.max_runtime_s,SETTINGS['diagnostic_seconds_max'])),trace.installed():
                try:records['LBFGS_reference'],_=scf(g,a,rho,cfg,out/'LBFGS_reference',trace,'LBFGS_reference','warm')
                except RuntimeError as exc:
                    if str(exc) not in ('Native electronic failure flag=1','Native electronic failure flag=2'):raise
                    records['LBFGS_reference']=load(out/'LBFGS_reference/result.json')
                tn=copy.deepcopy(cfg);tn['OPT']['method']='TN'
                records['TN_reference'],_=scf(g,a,rho,tn,out/'TN_reference',trace,'TN_reference','warm')
                tight=copy.deepcopy(tn);tight['OPT']['econv']=1e-11
                records['TN_tight_center'],center_rho=scf(g,a,rho,tight,out/'TN_tight_center',trace,'TN_tight_center','warm')
                i=int(np.where(a.arrays['parent_site_id']==58)[0][0]);records['TN_tight_center']['force_atom_index']=i
                g.dump(out/'TN_tight_center/result.json',records['TN_tight_center']);g.base.save_density(out/'center_density.npz',center_rho)
                records['TN_tight_HEG'],_=scf(g,a,make_heg(rho,len(a)),tight,out/'TN_tight_HEG',trace,'TN_tight_HEG','HEG')
                for h,suffix in zip(SETTINGS['force_fd_steps_A'],['1000','0500']):
                    for sign,tag in [(-1,'m'),(1,'p')]:
                        label='FD_'+tag+suffix;b=a.copy();b.positions[i,2]+=sign*h
                        records[label],_=scf(g,b,center_rho,tight,out/label,trace,label,'warm')
                result=evaluate_gate(records)
                require(trace.logical_count==8 and trace.physical_count<=14,'Unexpected SCF count')
                result.update(revision_manifest_sha256=report['revision_manifest_sha256'],parent_bundle_sha256=EXPECTED['bundle'],settings=SETTINGS,
                    center_density_sha256=sha(out/'center_density.npz'),record_sha256={k:sha(out/k/'result.json') for k in records},trace_summary=trace.summary(),elapsed_s=time.monotonic()-start,
                    scope='Checkpoint-specific self-consistent force spot gate; not a global surface/functional validation')
                g.dump(out/'gate_pass.json',result);g.dump(out/'diagnostic_result.json',result);return 0
        except BaseException as exc:
            g.dump(out/'diagnostic_result.json',{'accepted':False,'error':repr(exc),'traceback':traceback.format_exc(),'trace_summary':trace.summary(),'completed_records':list(records),'elapsed_s':time.monotonic()-start,'revision_manifest_sha256':report['revision_manifest_sha256'],'parent_bundle_sha256':EXPECTED['bundle']});return 1

def verify_gate(path,prep):
    g,checked,a,snap,cfg,d,meta,report=prep;path=path.resolve();gate=load(path)
    require(gate.get('accepted') is True and gate['revision_manifest_sha256']==report['revision_manifest_sha256'] and gate['parent_bundle_sha256']==EXPECTED['bundle'] and gate['settings']==SETTINGS,'Gate identity changed')
    records={}
    require(set(gate['record_sha256'])=={'LBFGS_reference','TN_reference','TN_tight_center','TN_tight_HEG','FD_m1000','FD_p1000','FD_m0500','FD_p0500'},'Incomplete gate records')
    for name,h in gate['record_sha256'].items():
        p=path.parent/name/'result.json';require(sha(p)==h,'Gate record changed '+name);records[name]=load(p)
    evaluate_gate(records)
    require(np.array_equal(records['TN_tight_center']['positions_A'],a.positions) and np.array_equal(records['TN_tight_center']['cell_A'],a.cell.array),'Gate used a different checkpoint geometry')
    require(records['TN_tight_center']['force_atom_index']==int(np.where(a.arrays['parent_site_id']==58)[0][0]),'Force parent index mismatch')
    dp=path.parent/'center_density.npz';require(sha(dp)==gate['center_density_sha256'],'Gate density changed')
    with np.load(dp,allow_pickle=False) as z:sd={k:z[k].copy() for k in z.files}
    rho=field(sd);validate_density(rho,len(a),cfg['GRID']['nr'],a.cell.array)
    return gate,rho

def run(args,prep):
    g,checked,a,snap,cfg,d,meta,report=prep
    require(args.output_root.resolve()==args.parent_runroot.resolve(),'Keep original case lock/root')
    require(args.gate.resolve()==args.parent_runroot.resolve()/'R13_0V/diagnostics_003_electronic/gate_pass.json','Wrong gate root')
    require(args.attempt=='attempt_003_TNsmallstep','New fixed attempt name required')
    gate,rho=verify_gate(args.gate,prep)
    c=copy.deepcopy(checked);c['protocol']['electronic'].update(method='TN',econv_Ha_per_atom=1e-11)
    c['protocol']['ionic'].update(optimizer='LBFGS_fixed_cell_finish',maxstep_A=.01,optimizer_policy=copy.deepcopy(POLICY))
    cfg=copy.deepcopy(cfg);cfg['OPT'].update(method='TN',econv=1e-11)
    c.update(optimizer_policy=POLICY,recovery_revision_manifest_sha256=report['revision_manifest_sha256'],density_initialization='Accepted TN tight center density from checkpoint-specific force gate',parent_checkpoint={'bundle_sha256':EXPECTED['bundle'],'completed_step':100,'parent_result_sha256':EXPECTED['parent_result']},electronic_retry_policy=SETTINGS['retry_policy'],electronic_gate_sha256=sha(args.gate),ionic_stage='fixed_cell_smallstep_finish',ionic_optimizer='LBFGS')
    c['initial_grid_spacing_A']=(a.cell.lengths()/c['protocol']['grid']).tolist()
    for name,p in {'electronic_recovery_runner.py':Path(__file__),'electronic_trace.py':HERE/'electronic_trace.py','recovery_runner.py':HERE/'recovery_base.py','staged_optimizer.py':HERE/'staged_optimizer.py','electronic_revision_manifest.json':HERE/'REVISION_MANIFEST.json','initial_checkpoint_bundle.npz':args.parent_attempt/'checkpoint_bundle.npz','parent_result.json':args.parent_attempt/'result.json','electronic_gate_pass.json':args.gate,'initial_density.npz':args.gate.parent/'center_density.npz'}.items():snap[name]=p;c['input_sha256'][name]=sha(p)
    base=module('recovery_base',HERE/'recovery_base.py');base.REPO=args.parent_snapshot.resolve()
    import dftpy.api.api4ase as api
    original=api.DFTpyCalculator
    class WarmCalculator(original):
        def __init__(self,*p,**kw):kw['rho']=rho.copy();super().__init__(*p,**kw)
    api.DFTpyCalculator=WarmCalculator
    attempt=args.output_root/'R13_0V'/args.attempt;trace=ElectronicTrace(attempt/'electronic_trace',len(a),cfg['GRID']['nr'],a.cell.array)
    try:
        with trace.installed():
            code=base.run(SimpleNamespace(output_root=args.output_root,case='R13_0V',attempt=args.attempt,campaign=args.parent_snapshot/'campaign',max_runtime_s=min(args.max_runtime_s,SETTINGS['ionic_seconds_max'])),c,a,snap,cfg)
        rp=attempt/'result.json';result=load(rp)
        result['all_logical_densities_used_converged']=result.pop('all_electronic_calls_converged',False)
        result.update(electronic_attempt_summary=trace.summary(),electronic_optimizer='TN',electronic_econv_Ha_per_atom=1e-11,ionic_optimizer='LBFGS',ionic_maxstep_A=.01,ionic_stage='fixed_cell_smallstep_finish',recovery_revision_manifest_sha256=report['revision_manifest_sha256'],electronic_gate_sha256=sha(args.gate),parent_checkpoint=c['parent_checkpoint'])
        if code==0:
            from ase.io import read
            final=read(attempt/'final.extxyz');gg=g.final_periodic_geometry(final)
            if not gg['accepted'] or not np.allclose(final.cell.array,d['cell_A'],atol=1e-8,rtol=0):result.update(accepted=False,status='final_periodic_geometry_rejected');code=1
            result['final_periodic_geometry_gate']=gg
        result['artifact_sha256']={str(p.relative_to(attempt)):sha(p) for p in sorted(attempt.rglob('*')) if p.is_file() and p.name not in ('result.json','result.json.tmp')}
        base.dump(rp,result);return code
    finally:api.DFTpyCalculator=original

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['verify','diagnose','run'])
    for name in ['parent-snapshot','parent-runroot','parent-attempt']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--output-root',type=Path);p.add_argument('--gate',type=Path);p.add_argument('--attempt',default='attempt_003_TNsmallstep');p.add_argument('--max-runtime-s',type=float,default=6600)
    args=p.parse_args();require(0<args.max_runtime_s<=SETTINGS['ionic_seconds_max'],'Invalid budget');prep=prepare(args)
    print(json.dumps(prep[-1],indent=2))
    if args.mode=='verify':return 0
    require(sys.platform=='linux' and args.output_root is not None,'Execution requires Linux/output root')
    if args.mode=='diagnose':return diagnostic(args,prep)
    require(args.gate is not None,'Gate required');return run(args,prep)

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:print('[REJECTED] '+repr(exc),file=sys.stderr);raise
