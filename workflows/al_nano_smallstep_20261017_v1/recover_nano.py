"""One auditable entry point for a bounded R13_0V small-step LBFGS finish.

Only `execute` constructs a calculator. `verify` validates hashes/checkpoints.
Parent frozen scripts and gate remain their own immutable identity. This
revision only changes the ionic optimization path, not the electronic problem.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from ase import Atoms
from ase.io import read
from ase.units import Bohr
from staged_optimizer import POLICY

HERE = Path(__file__).resolve().parent
EXPECTED = {
    'case': 'R13_0V',
    'parent_manifest': 'e2a78d9c360f5fbef8ec267a0565e2f74d17542e5641262071cf3bbb5fbe1674',
    'parent_result': 'd3d3078939c8cefaf439854af3fc22b4aa2e7f7306c069f3648c075be8022226',
    'parent_gate': 'b1ddc3cb06fed8e341eaf2197e9436c5841e25f6f34236a056ca70527374acd0',
    'bundle': '7fb7e3a9b8419a10b84314a234448dc2cf1d175cec289fde79242cb9a4971254',
    'stress_candidate': '06c86e2aa20408042746f0d457765991ca91afe1ebf20737562108928b0acfeb',
    'parent_runner': '2870c1ff9136e43fb40c411bf045d26ba0fb8f10021d4c6b0a99cf64473b1253',
}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def require(condition, message):
    if not condition: raise ValueError(message)

def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m

def verify_revision():
    manifest=json.loads((HERE/'REVISION_MANIFEST.json').read_text())
    require(manifest['parent'] == EXPECTED, 'Recovery parent identity changed')
    require(manifest['optimizer_policy'] == POLICY, 'Recovery policy changed')
    for name,h in manifest['files'].items():
        p=(HERE/name).resolve()
        require(p.is_relative_to(HERE), 'Revision path escape')
        require(sha(p)==h, 'Revision hash mismatch: '+name)
    return sha(HERE/'REVISION_MANIFEST.json')

def check_physics(protocol):
    require(protocol['grid']==[140,140,96], 'R13 grid changed')
    require(protocol['xc']=='LDA' and protocol['lambda_tf']==.9 and protocol['mu_vw']==.1, 'R13 functional changed')
    require(protocol['pseudo']['sha256']=='c2fd4300aa39aa2466149a315e1c7cd87aac6615032db5ded8f237e9c40df803', 'PP changed')
    require(protocol['electronic']==dict(method='LBFGS', econv_Ha_per_atom=1e-9,
            ncheck=3,maxiter=1000,maxls=50,maxfun=100,densityini='heg',reuse=True), 'Electronic settings changed')
    require(protocol['ionic']==dict(optimizer='LBFGS',fmax_eV_A=.001,filter_fmax_eV_A=.001,max_steps=1000,maxstep_A=.15), 'Parent ionic settings changed')
    require(protocol['cell']['mask']==[0,0,1,0,0,0] and protocol['cell']['stress_max_abs_GPa']==.05, 'Cell gate changed')

def load_bundle(path, geometry, protocol, result):
    """Read one atomic NPZ; never combine separately replaced density/geometry."""
    with np.load(path, allow_pickle=False) as z:
        d={key:z[key].copy() for key in z.files}
    required={'metadata_json','positions_A','cell_A','numbers','pbc','parent_site_id',
              'forces_eV_A','density_electrons_bohr3','lattice_bohr','grid','origin_bohr'}
    require(required.issubset(d), 'Incomplete checkpoint bundle')
    meta=json.loads(str(d['metadata_json']))
    n=geometry['n_atoms']; cell=d['cell_A'];rho=d['density_electrons_bohr3']
    require(n==76 and len(d['numbers'])==n and np.all(d['numbers']==13), 'Wrong atoms')
    require(d['positions_A'].shape==(n,3) and d['forces_eV_A'].shape==(n,3), 'Bad positions/force shape')
    require(d['pbc'].shape==(3,) and np.all(d['pbc']), 'PBC changed')
    require(d['parent_site_id'].tolist()==geometry['remaining_parent_ids'], 'Parent site identity changed')
    require(d['grid'].tolist()==protocol['grid'] and list(rho.shape)==protocol['grid'], 'Density grid mismatch')
    require(cell.shape==(3,3) and np.linalg.det(cell)>0, 'Invalid cell')
    require(np.allclose(d['lattice_bohr']*Bohr,cell,atol=1e-8,rtol=0), 'Density/geometry lattice mismatch')
    require(d['origin_bohr'].shape==(3,) and np.allclose(d['origin_bohr'],0,atol=1e-12,rtol=0), 'Density origin changed')
    frozen=np.ones((3,3),dtype=bool);frozen[2,2]=False
    require(np.allclose(cell[frozen],np.asarray(geometry['cell_A'])[frozen],atol=1e-8,rtol=0), 'Transverse cell/tilts changed')
    require(np.min(np.diag(cell))>=10, 'Periodic distance below 10 A')
    for key in required-{'metadata_json'}:
        require(np.all(np.isfinite(d[key])), 'Nonfinite bundle: '+key)
    require(np.min(rho)>0, 'Stress candidate requires strictly positive density')
    integral=float(rho.sum()*np.linalg.det(d['lattice_bohr'])/rho.size)
    require(abs(integral-3*n)<=1e-6, 'Density electron integral mismatch')
    fmax=float(np.linalg.norm(d['forces_eV_A'],axis=1).max())
    require(abs(fmax-meta['fmax_eV_A'])<1e-10, 'Bundle force/metadata mismatch')
    require(np.allclose(cell,meta['cell_A'],atol=1e-10,rtol=0), 'Bundle metadata cell mismatch')
    require(meta['ionic_step']==result['ionic_steps']==1000, 'Unexpected checkpoint step')
    require(abs(meta['energy_eV']-result['energy_eV'])<1e-9 and abs(fmax-result['fmax_eV_A'])<1e-10, 'Parent result/checkpoint mismatch')
    require(np.allclose(cell,result['cell_A'],atol=1e-10,rtol=0), 'Parent result/cell mismatch')
    require(max(meta['fmax_eV_A'],meta['filter_fmax_eV_A'])<=.01, 'Not the approved low-force R13 recovery')
    require(abs(meta['stress_GPa'][2][2])<=protocol['cell']['stress_max_abs_GPa'], 'Parent axial stress exceeds finish entry gate')
    atoms=Atoms(numbers=d['numbers'],positions=d['positions_A'],cell=cell,pbc=d['pbc'])
    atoms.set_array('parent_site_id',d['parent_site_id'])
    return atoms,d,meta,integral

def prepare(args):
    revision_hash=verify_revision()
    parent=args.parent_snapshot.resolve(); pa=args.parent_attempt.resolve()
    require(pa.name=='attempt_001' and pa.parent.name==EXPECTED['case'], 'Wrong parent attempt')
    require(sha(parent/'MANIFEST.json')==EXPECTED['parent_manifest'], 'Parent manifest identity mismatch')
    sys.path.insert(0,str(parent/'scripts'))
    # Parent imports must retain their own base_runner and app module identity.
    gated=load_module('gated_nano',parent/'scripts/gated_nano.py')
    packhash=gated.verify_package()
    require(packhash==EXPECTED['parent_manifest'], 'Parent package mismatch')
    require(sha(pa/'result.json')==EXPECTED['parent_result'], 'Parent result changed')
    result=json.loads((pa/'result.json').read_text())
    require(result['accepted'] is False and result['status']=='relaxation_unconverged', 'Parent is accepted or wrong failure kind')
    require(result['all_electronic_calls_converged'] is True, 'Parent density not fully converged')
    for name in ['checkpoint.json','provenance.json','config.json','inputs/package_manifest.json',
                 'inputs/runner.py','inputs/stress_candidate.py','inputs/dftpy_cases.json']:
        require(sha(pa/name)==result['artifact_sha256'][name], 'Parent artifact mismatch: '+name)
    require(sha(parent/'scripts/base_runner.py')==EXPECTED['parent_runner'], 'Parent runner changed')
    require(sha(parent/'scripts/stress_candidate.py')==EXPECTED['stress_candidate'], 'Stress candidate changed')
    require(sha(pa/'checkpoint_bundle.npz')==EXPECTED['bundle']==result['artifact_sha256']['checkpoint_bundle.npz'], 'Checkpoint bundle hash mismatch')
    checkpoint=json.loads((pa/'checkpoint.json').read_text())
    require(checkpoint['checkpoint_bundle_sha256']==EXPECTED['bundle'], 'Checkpoint header mismatch')
    gate_path=args.parent_runroot.resolve()/'gate_pass.json'
    require(sha(gate_path)==EXPECTED['parent_gate'], 'Parent validation gate changed')
    settings=json.loads((parent/'campaign/gate.json').read_text())
    gated.verify_gate(gate_path,packhash,settings)
    checked,initial,snap,cfg=gated.prep(EXPECTED['case'])
    check_physics(checked['protocol'])
    require(checked['protocol']['stress_candidate']=='spectral Hessian of sqrt(rho), no smoothing, process-local only', 'Stress protocol changed')
    atoms,d,meta,integral=load_bundle(pa/'checkpoint_bundle.npz',checked['geometry_case'],checked['protocol'],result)
    old_prov=json.loads((pa/'provenance.json').read_text())
    require(old_prov['protocol']==checked['protocol'], 'Parent protocol provenance mismatch')
    # All SCF settings except snapshotted PP directory are byte-equivalent JSON values.
    old_cfg=json.loads((pa/'config.json').read_text());comp=copy.deepcopy(cfg)
    comp['PATH']['pppath']=old_cfg['PATH']['pppath']
    require(comp==old_cfg, 'Resolved electronic configuration changed')
    modified=copy.deepcopy(checked)
    modified['protocol']['ionic'].update(optimizer='LBFGS_fixed_cell_finish',maxstep_A=.01,optimizer_policy=copy.deepcopy(POLICY))
    modified['density_initialization']='coherent checkpoint_bundle warm density; electronic LBFGS unchanged'
    modified['ionic_optimizer']='LBFGS'
    modified['ionic_stage']='fixed_cell_smallstep_finish'
    modified['cell_filter_role']='final force/stress acceptance only; no cell moves during finish'
    modified['optimizer_policy_status']='historical optimizer family; current strict endpoint must still pass physical acceptance'
    modified['parent_checkpoint']={'bundle_sha256':EXPECTED['bundle'],'parent_result_sha256':EXPECTED['parent_result'],
            'parent_attempt':str(pa),'completed_step':meta['ionic_step'],'initial_energy_eV':meta['energy_eV'],
            'initial_fmax_eV_A':meta['fmax_eV_A'],'density_integral_electrons':integral,
            'optimizer_state':'new small-step LBFGS Hessian; prior ionic Hessian not reused'}
    modified['parent_package_manifest_sha256']=packhash
    modified['recovery_revision_manifest_sha256']=revision_hash
    modified['optimizer_policy']=copy.deepcopy(POLICY)
    modified['initial_grid_spacing_A']=(atoms.cell.lengths()/modified['protocol']['grid']).tolist()
    # Keep ideal parent geometry as provenance; frozen-transverse check is identical.
    for name,p in {'recovery_runner.py':HERE/'recovery_base.py','recover_nano.py':Path(__file__),
                   'staged_optimizer.py':HERE/'staged_optimizer.py','recovery_revision_manifest.json':HERE/'REVISION_MANIFEST.json',
                   'parent_result.json':pa/'result.json','parent_checkpoint.json':pa/'checkpoint.json',
                   'initial_checkpoint_bundle.npz':pa/'checkpoint_bundle.npz','validation_gate_pass.json':gate_path}.items():
        snap[name]=p;modified['input_sha256'][name]=sha(p)
    report={'accepted':True,'mode':'verify_no_calculation','case':EXPECTED['case'],
            'parent_package_manifest_sha256':packhash,'recovery_revision_manifest_sha256':revision_hash,
            'parent_checkpoint':modified['parent_checkpoint'],'ionic_optimizer':'LBFGS',
            'ionic_stage':'fixed_cell_smallstep_finish','maxstep_A':.01,
            'unchanged_electronic_optimizer':'LBFGS','grid':modified['protocol']['grid'],
            'force_target_eV_A':.001,'sigma_zz_target_GPa':.05,'cell_A':atoms.cell.array.tolist(),
            'bounded_max_steps':modified['protocol']['ionic']['max_steps'],
            'scientific_status':'candidate recovery; success requires inherited full acceptance after execution'}
    return gated,modified,atoms,snap,cfg,d,report

def execute(args,prepared):
    gated,checked,atoms,snap,cfg,d,report=prepared
    require(sys.platform=='linux', 'Execute requires Linux')
    require(args.output_root is not None, 'Explicit output-root required')
    require(args.attempt!='attempt_001', 'Recovery needs a distinct attempt name')
    # One case lock covers old and new attempts; do not run separate parallel recovery trees.
    require(args.output_root.resolve()==args.parent_runroot.resolve(), 'Reuse parent runroot lock')
    require(0<args.max_runtime_s<=checked['protocol']['runtime']['max_walltime_s'], 'Budget exceeds frozen bound')
    base=load_module('recovery_base',HERE/'recovery_base.py')
    # Scientific source identity and outside-source output check retain the
    # frozen parent; the separate recovery revision is recorded explicitly.
    base.REPO=args.parent_snapshot.resolve()
    from dftpy.grid import DirectGrid
    from dftpy.field import DirectField
    import dftpy.api.api4ase as api
    grid=DirectGrid(lattice=d['lattice_bohr'],nr=d['grid'],origin=d['origin_bohr'])
    rho=DirectField(grid=grid,griddata_3d=d['density_electrons_bohr3'])
    original=api.DFTpyCalculator
    class WarmCalculator(original):
        def __init__(self,*pos,**kw):
            kw['rho']=rho.copy();super().__init__(*pos,**kw)
    api.DFTpyCalculator=WarmCalculator
    gated.install()
    try:
        runargs=SimpleNamespace(output_root=args.output_root,case=EXPECTED['case'],attempt=args.attempt,
                campaign=args.parent_snapshot/'campaign',max_runtime_s=args.max_runtime_s)
        code=base.run(runargs,checked,atoms,snap,cfg)
        attempt=args.output_root/EXPECTED['case']/args.attempt
        rp=attempt/'result.json';result=json.loads(rp.read_text())
        result.update(ionic_optimizer='LBFGS',ionic_stage='fixed_cell_smallstep_finish',
                relaxation_mode='fixed_cell_atoms_after_axial_relaxation; original axial filter/stress acceptance retained',
                electronic_optimizer='LBFGS',optimizer_policy=POLICY,
                recovery_revision_manifest_sha256=report['recovery_revision_manifest_sha256'],
                parent_checkpoint=checked['parent_checkpoint'])
        if code==0:
            try:
                final=read(attempt/'final.extxyz')
                geometry_gate=gated.final_periodic_geometry(final)
                result['final_periodic_geometry_gate']=geometry_gate
                require(geometry_gate['accepted'],'Final periodic geometry rejected')
                require(np.allclose(final.cell.array,d['cell_A'],atol=1e-8,rtol=0),'Fixed-cell finish changed final cell')
            except Exception as exc:
                result.update(accepted=False,status='final_periodic_geometry_rejected',error=repr(exc));code=1
        base.dump(rp,result)
        return code
    finally:
        api.DFTpyCalculator=original

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['verify','execute'])
    p.add_argument('--parent-snapshot',type=Path,required=True)
    p.add_argument('--parent-runroot',type=Path,required=True)
    p.add_argument('--parent-attempt',type=Path,required=True)
    p.add_argument('--output-root',type=Path)
    p.add_argument('--attempt',default='attempt_002_smallstep')
    p.add_argument('--max-runtime-s',type=float,default=79200)
    args=p.parse_args();prepared=prepare(args)
    print(json.dumps(prepared[-1],indent=2))
    return execute(args,prepared) if args.mode=='execute' else 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print('[REJECTED] '+repr(exc),file=sys.stderr);raise
