"""One bounded axial BFGSLineSearch continuation, reusing the frozen scientific runner."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['MPLBACKEND']='Agg';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,copy,hashlib,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from ase import Atoms
from ase.units import Bohr
from electronic_trace import ElectronicTrace,validate_density
from staged_optimizer import POLICY
import recovery_base as base

HERE=Path(__file__).resolve().parent
EXPECTED={'case':'R13_0V','original_manifest':'e2a78d9c360f5fbef8ec267a0565e2f74d17542e5641262071cf3bbb5fbe1674',
 'electronic_revision':'79009e0248878103f7b2ac86b390e03843100324866678008dc0e69881d8c223',
 'parent_result':'32bedd18ba54f078d9e7c17f0f8ae7cf73201608a90fe83da40d645853da36be',
 'axial_revision':'0c4cd1072d87fe1fbe3aff60bc0c45d72a10e29f5dc0096dca8278b3fd9ab01e',
 'parent_provenance':'5c4eed227bfc7addf4d0c17feaf1a59fc5c77b70081bfad48c55f15be408c696',
 'parent_bundle':'d30d37f9c33e7ef7be05a5ac653232f348774304636eb6691163f5b508f5297f',
 'electronic_gate':'c6d9465abddc8e38671eee7c623b0962ed2f430889aabe1f28b3c0eb2345baf2',
 'original_stress_gate':'b1ddc3cb06fed8e341eaf2197e9436c5841e25f6f34236a056ca70527374acd0',
 'base_runner':'dc36263b50496c3b03822cbc94e3b1d7fd153d38a78dc547fd2b717d0ffe5b72'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
def require(ok,msg):
    if not ok:raise ValueError(msg)
def mod(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def verify_revision():
    m=load(HERE/'REVISION_MANIFEST.json');require(m['expected']==EXPECTED and m['optimizer_policy']==POLICY,'Revision policy changed')
    for rel,h in m['files'].items():
        p=(HERE/rel).resolve();require(p.is_relative_to(HERE) and sha(p)==h,'Revision source changed '+rel)
    require(sha(HERE/'recovery_base.py')==EXPECTED['base_runner'],'Scientific core changed')
    return sha(HERE/'REVISION_MANIFEST.json')

def prepare(args):
    rh=verify_revision();parent=args.original_snapshot.resolve();previous=args.electronic_snapshot.resolve();pa=args.parent_attempt.resolve();root=args.runroot.resolve();axial=args.axial_snapshot.resolve()
    require(sha(axial/'REVISION_MANIFEST.json')==EXPECTED['axial_revision'],'Axial parent revision changed')
    for rel,h in load(axial/'REVISION_MANIFEST.json')['files'].items():require(sha(axial/rel)==h,'Axial parent file changed '+rel)
    require(sha(parent/'MANIFEST.json')==EXPECTED['original_manifest'],'Original snapshot identity changed')
    require(sha(previous/'REVISION_MANIFEST.json')==EXPECTED['electronic_revision'],'Electronic revision changed')
    for rel,h in load(previous/'REVISION_MANIFEST.json')['files'].items():require(sha(previous/rel)==h,'Electronic parent file changed '+rel)
    sys.path.insert(0,str(parent/'scripts'));g=mod('gated_nano',parent/'scripts/gated_nano.py')
    require(g.verify_package()==EXPECTED['original_manifest'],'Original package changed')
    gate=root/'gate_pass.json';require(sha(gate)==EXPECTED['original_stress_gate'],'Stress gate changed');g.verify_gate(gate,EXPECTED['original_manifest'],load(parent/'campaign/gate.json'))
    egate=root/'R13_0V/diagnostics_003_electronic/gate_pass.json';require(sha(egate)==EXPECTED['electronic_gate'],'Electronic gate changed')
    e=load(egate);require(e['accepted'] is True and e['revision_manifest_sha256']==EXPECTED['electronic_revision'],'Electronic gate invalid')
    prior=mod('prior_electronic_entry',previous/'recover_electronic.py');records={}
    for label,h in e['record_sha256'].items():
        p=egate.parent/label/'result.json';require(sha(p)==h,'Electronic gate result changed '+label);records[label]=load(p)
    prior.evaluate_gate(records)
    require(sha(pa/'result.json')==EXPECTED['parent_result'],'Interrupted parent changed');result=load(pa/'result.json')
    require(result['status']=='time_budget_exhausted' and result['accepted'] is False and result['error'].startswith("BudgetExpired('Wall-time budget reached;"),'Wrong parent status')
    require(sha(pa/'provenance.json')==EXPECTED['parent_provenance'],'Parent provenance changed')
    parent_provenance=load(pa/'provenance.json');require(parent_provenance['recovery_revision_manifest_sha256']==EXPECTED['axial_revision'],'Parent provenance revision mismatch')
    require(result['ionic_optimizer']=='BFGSLineSearch' and result['relaxation_mode']=='atoms_and_axial_zz_FrechetCellFilter_zero_pressure','Parent ionic method mismatch')
    require(result['recovery_revision_manifest_sha256']==EXPECTED['axial_revision'] and parent_provenance['electronic_gate_sha256']==EXPECTED['electronic_gate'],'Parent protocol mismatch')
    for name in ('checkpoint.json','checkpoint_bundle.npz','config.json','provenance.json'):
        require(sha(pa/name)==result['artifact_sha256'][name],'Parent artifact changed '+name)
    require(sha(pa/'checkpoint_bundle.npz')==EXPECTED['parent_bundle'],'Bundle identity changed')
    checked,_,snap,cfg=g.prep('R13_0V');p=checked['protocol']
    require(p['xc']=='LDA' and p['lambda_tf']==.9 and p['mu_vw']==.1 and p['grid']==[140,140,96],'Physical settings changed')
    require(p['cell']['mask']==[0,0,1,0,0,0] and p['cell']['stress_max_abs_GPa']==.05 and p['ionic']['fmax_eV_A']==p['ionic']['filter_fmax_eV_A']==.001,'Acceptance changed')
    cfg['OPT'].update(method='TN',econv=1e-11);oldcfg=load(pa/'config.json');compare=copy.deepcopy(cfg);compare['PATH']['pppath']=oldcfg['PATH']['pppath'];require(compare==oldcfg,'Electronic configuration changed')
    require(cfg['OPT']['ncheck']==3 and cfg['MATH']['reuse'] is True,'Electronic policy changed')
    with np.load(pa/'checkpoint_bundle.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    meta=json.loads(str(d['metadata_json']));cp=load(pa/'checkpoint.json');cell=d['cell_A'];rho=d['density_electrons_bohr3']
    require(meta['ionic_step']==cp['ionic_step']==12 and result['ionic_steps']==12,'Wrong completed/proposed step')
    require(cp['checkpoint_bundle_sha256']==EXPECTED['parent_bundle'],'Header/bundle mismatch')
    require(len(d['numbers'])==76 and np.all(d['numbers']==13) and d['positions_A'].shape==(76,3) and d['forces_eV_A'].shape==(76,3),'Wrong atoms')
    require(np.all(d['pbc']) and d['parent_site_id'].tolist()==checked['geometry_case']['remaining_parent_ids'],'PBC/parent IDs changed')
    require(d['grid'].tolist()==[140,140,96] and rho.shape==(140,140,96),'Grid changed')
    require(all(np.isfinite(v).all() for k,v in d.items() if k!='metadata_json') and np.all(rho>0),'Nonfinite/nonpositive bundle')
    require(np.max(abs(cell-np.diag(np.diag(cell))))<1e-8 and np.min(np.diag(cell))>=10,'Invalid periodic cell')
    require(np.allclose(d['lattice_bohr']*Bohr,cell,atol=1e-8,rtol=0) and np.allclose(d['origin_bohr'],0,atol=1e-12),'Density geometry mismatch')
    require(abs(rho.mean()*np.linalg.det(d['lattice_bohr'])-228)<1e-6,'Wrong electron number')
    for k in ('energy_eV','fmax_eV_A','filter_fmax_eV_A','cell_A','stress_GPa'):require(meta[k]==cp[k],'Metadata mismatch '+k)
    require(np.array_equal(cell,np.asarray(cp['cell_A'])) and abs(np.linalg.norm(d['forces_eV_A'],axis=1).max()-meta['fmax_eV_A'])<1e-12,'Force/cell incoherence')
    require(meta['fmax_eV_A']<.02 and meta['filter_fmax_eV_A']<.02,'Unexpected checkpoint state')
    atoms=Atoms(numbers=d['numbers'],positions=d['positions_A'],cell=cell,pbc=d['pbc']);atoms.set_array('parent_site_id',d['parent_site_id'])
    g.install();report={'accepted':True,'scope':'preflight only, no DFT','revision_manifest_sha256':rh,'parent_bundle_sha256':EXPECTED['parent_bundle'],'completed_ionic_step':12,'n_atoms':76,'electrons':228,'grid':[140,140,96],'cell_A':cell.tolist(),'optimizer_policy':POLICY,'force_target_eV_A':.001,'stress_target_GPa':.05}
    return g,checked,atoms,snap,cfg,d,meta,report

def execute(args,prep):
    g,checked,a,snap,cfg,d,meta,report=prep
    require(sys.platform=='linux' and 0<args.max_runtime_s<=21000,'Execution requires Linux and bounded budget')
    require(args.attempt=='attempt_005_axial_linesearch_continue','Unique fixed attempt required')
    require(args.parent_attempt.resolve()==args.runroot.resolve()/'R13_0V/attempt_004_axial_linesearch','Parent must be same case runroot')
    base.REPO=args.original_snapshot.resolve()
    c=copy.deepcopy(checked);c['scientific_scope']='Pristine R13 nanocolumn (0V), axial periodic relaxation; numerical transfer tests incomplete';c['protocol']['electronic'].update(method='TN',econv_Ha_per_atom=1e-11)
    c['protocol']['ionic'].update(optimizer='BFGSLineSearch_axial',maxstep_A=.02,max_steps=120,optimizer_policy=POLICY)
    c.update(optimizer_policy=POLICY,recovery_revision_manifest_sha256=report['revision_manifest_sha256'],ionic_stage='atoms_and_axial_zz_linesearch',ionic_optimizer='BFGSLineSearch',
        parent_checkpoint={'bundle_sha256':EXPECTED['parent_bundle'],'completed_step':12,'parent_result_sha256':EXPECTED['parent_result'],'parent_provenance_sha256':EXPECTED['parent_provenance'],'parent_revision_sha256':EXPECTED['axial_revision']},density_initialization='coherent completed trial step12 density; native DFTpy cell remap and charge normalization',optimizer_initialization='fresh inverse Hessian; no optimizer history was saved by parent trial',electronic_gate_sha256=EXPECTED['electronic_gate'],relaxation_mode='atoms_and_axial_zz_FrechetCellFilter_zero_pressure')
    c['initial_grid_spacing_A']=(a.cell.lengths()/c['protocol']['grid']).tolist()
    for name,p in {'axial_recovery_entry.py':Path(__file__),'electronic_trace.py':HERE/'electronic_trace.py','recovery_runner.py':HERE/'recovery_base.py','staged_optimizer.py':HERE/'staged_optimizer.py','axial_revision_manifest.json':HERE/'REVISION_MANIFEST.json','initial_checkpoint_bundle.npz':args.parent_attempt/'checkpoint_bundle.npz','parent_result.json':args.parent_attempt/'result.json','parent_provenance.json':args.parent_attempt/'provenance.json','parent_revision_manifest.json':args.axial_snapshot/'REVISION_MANIFEST.json','electronic_gate_pass.json':args.runroot/'R13_0V/diagnostics_003_electronic/gate_pass.json'}.items():snap[name]=p;c['input_sha256'][name]=sha(p)
    from dftpy.grid import DirectGrid
    from dftpy.field import DirectField
    import dftpy.api.api4ase as api
    rho=DirectField(grid=DirectGrid(lattice=d['lattice_bohr'],nr=d['grid'],origin=d['origin_bohr']),griddata_3d=d['density_electrons_bohr3'])
    original=api.DFTpyCalculator;holder={}
    class WarmCalculator(original):
        def __init__(self,*pos,**kw):kw['rho']=rho.copy();super().__init__(*pos,**kw);holder['calculator']=self
    def current_geometry():
        actual=holder['calculator'].atoms
        return {'cell_A':actual.cell.array.copy(),'pbc':actual.pbc.copy()}
    api.DFTpyCalculator=WarmCalculator;attempt=args.runroot/'R13_0V'/args.attempt
    trace=ElectronicTrace(attempt/'electronic_trace',76,[140,140,96],d['cell_A'],geometry_provider=current_geometry,max_logical_calls=360)
    # Core source stays byte-identical. Record the actual mode at every JSON write,
    # including the initial running result and interrupted results.
    original_dump=base.dump
    def truthful_dump(path,obj):
        if Path(path).name in ('result.json','provenance.json'):
            obj=dict(obj);obj['relaxation_mode']='atoms_and_axial_zz_FrechetCellFilter_zero_pressure';obj['ionic_optimizer']='BFGSLineSearch';obj['ionic_stage']='atoms_and_axial_zz_linesearch'
        return original_dump(path,obj)
    base.dump=truthful_dump
    try:
        with trace.installed():
            code=base.run(SimpleNamespace(output_root=args.runroot,case='R13_0V',attempt=args.attempt,campaign=args.original_snapshot/'campaign',max_runtime_s=args.max_runtime_s),c,a,snap,cfg)
        rp=attempt/'result.json';result=load(rp);result['all_logical_densities_used_converged']=result.pop('all_electronic_calls_converged',False)
        result.update(electronic_attempt_summary=trace.summary(),electronic_optimizer='TN',electronic_econv_Ha_per_atom=1e-11,recovery_revision_manifest_sha256=report['revision_manifest_sha256'],parent_checkpoint=c['parent_checkpoint'],ionic_line_search_maxstep=.02,maxstep_scope='ASE generalized-coordinate trial increment parameter, not a guaranteed whole-step atomic displacement cap',maximum_logical_scf=360,maximum_ionic_steps=120,optimizer_initialization=c['optimizer_initialization'])
        error=result.get('error','')
        if 'Repeated ionic cycle detected' in error:result['status']='stagnation_detected'
        if 'Logical electronic call budget exhausted' in error:result['status']='electronic_call_budget_exhausted'
        if code==0:
            from ase.io import read
            final=read(attempt/'final.extxyz');gg=g.final_periodic_geometry(final);result['final_periodic_geometry_gate']=gg
            if not gg['accepted']:result.update(accepted=False,status='final_periodic_geometry_rejected');code=1
        result['artifact_sha256']={str(p.relative_to(attempt)):sha(p) for p in sorted(attempt.rglob('*')) if p.is_file() and p.name not in ('result.json','result.json.tmp')}
        base.dump(rp,result);return code
    finally:api.DFTpyCalculator=original;base.dump=original_dump

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['verify','execute'])
    for n in ('original-snapshot','electronic-snapshot','axial-snapshot','runroot','parent-attempt'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--attempt',default='attempt_005_axial_linesearch_continue');p.add_argument('--max-runtime-s',type=float,default=21000)
    args=p.parse_args();prepared=prepare(args);print(json.dumps(prepared[-1],indent=2));return execute(args,prepared) if args.mode=='execute' else 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:print('[REJECTED] '+repr(exc),file=sys.stderr);raise
