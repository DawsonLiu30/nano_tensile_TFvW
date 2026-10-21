"""Reviewed k7-only entry point; parent scientific runner remains unchanged.

The failed k5 gate still blocks the six new-size relaxations. This entry point
only executes its two pre-existing conditional k7 SCFs after a new review.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)

def reviewed_case(m,index):
    require(index in (8,9),'Only the two reviewed R15 k7 SCFs may run')
    c=m['cases'][index]
    require(c['id']==f'nano_R15_{index-8}V_k7_SCF_20261015' and c['calculation']=='scf'
            and c['stage']=='conditional_k7' and c['kmesh']==[1,1,7], 'Wrong reviewed case')
    require(c['nat'] in (95,96) and c['nat']<=250,'Atom count exceeds reviewed scope')
    return c

def load_and_verify(parent,root):
    review=json.loads((HERE/'review_manifest.json').read_text())
    require(review['review_id']=='R15_k7_only_20261017_v1' and review['allowed_indices']==[8,9], 'Wrong review identity')
    for rel,h in review['entrypoint_sha256'].items():
        p=(HERE/rel).resolve();require(p.is_relative_to(HERE),'Review file path escape')
        require(sha(p)==h,'Reviewed entry point changed: '+rel)
    for rel,h in review['parent_file_sha256'].items():
        p=(parent/rel).resolve();require(p.is_relative_to(parent.resolve()),'Parent path escape')
        require(sha(p)==h,'Frozen parent changed: '+rel)
    sys.path.insert(0,str(parent/'scripts'))
    spec=importlib.util.spec_from_file_location('run_nano_gated_20261015',parent/'scripts/run_nano_gated_20261015.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    m=original.verify(parent)
    gatepath=root/'gate_status.json'
    require(sha(gatepath)==review['k5_evidence']['gate_status_sha256'],'k5 gate evidence changed')
    gate=json.loads(gatepath.read_text())
    require(gate['passed'] is False and gate['released_case_ids']==[],'Unexpected prior gate state')
    require(abs(gate['delta_E1_minus_E0_eV']-review['k5_evidence']['delta_E1_minus_E0_eV'])<1e-12,'k5 evidence value changed')
    for old,h in gate['stage1_hashes'].items():
        rel=Path(old).relative_to(review['parent_runroot'])
        require(sha(root/rel)==h,'Actual k5 evidence artifact changed: '+str(rel))
    for index in [8,9]:
        c=reviewed_case(m,index);prior=m['cases'][index-2]
        # Exact text comparison makes geometry, all controls and numerical
        # values identical. Only mesh and output-prefix identity differ.
        k5=(parent/prior['input']).read_text()
        expected=k5.replace(prior['prefix'],c['prefix']).replace('1 1 5 0 0 0','1 1 7 0 0 0')
        require(expected==(parent/c['input']).read_text(),'k7 differs from k5 beyond prefix and mesh')
        require(c['cell_A']==prior['cell_A'] and c['positions_A']==prior['positions_A'], 'k7 geometry changed')
        require(min(c['cell_A'][i][i] for i in range(3))>=10,'Periodic translation below 10 A')
    return original,m,review

def run(original,m,review,parent,root,index):
    import fcntl
    c=reviewed_case(m,index)
    require(sys.platform=='linux' and os.environ.get('SLURM_JOB_ID'),'Run requires Linux Slurm allocation')
    require(str(root.resolve())==review['parent_runroot'],'Production runroot must match reviewed parent')
    require(not any((p/'.git').exists() for p in [root,*root.parents]),'Output inside Git')
    require(sha(m['binary'])==m['binary_sha256'],'QE binary changed')
    limits=original.memory_limits();original.allocation_guard(c,os.environ,min(limits.values()))
    case_root=root/c['id'];case_root.mkdir(parents=True,exist_ok=True)
    with (case_root/'.execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        a=case_root/'attempt_001';a.mkdir(exist_ok=False)
        shutil.copyfile(parent/c['input'],a/'pw.in');(a/'pseudo').mkdir()
        shutil.copyfile(parent/m['pseudo_file'],a/'pseudo/Al_PAW_PBE.UPF')
        shutil.copyfile(parent/'qe_manifest.json',a/'qe_manifest.json')
        shutil.copyfile(HERE/'review_manifest.json',a/'k7_review_manifest.json')
        shutil.copyfile(__file__,a/'run_reviewed_k7.py')
        cmd=original.launch_command(m['binary'])
        original.dump(a/'provenance.json',{'case':c,'pseudo_sha256':m['pseudo_sha256'],
            'binary_sha256':m['binary_sha256'],'runner_sha256':sha(__file__),
            'parent_runner_sha256':sha(parent/'scripts/run_nano_gated_20261015.py'),
            'nano_inspector_sha256':sha(original.BASE.__file__),
            'review_manifest_sha256':sha(HERE/'review_manifest.json'),
            'parent_campaign_sha256':sha(parent/'qe_manifest.json'),'parent_snapshot':str(parent),
            'k5_gate_status_sha256':review['k5_evidence']['gate_status_sha256'],
            'scope':'k7 fixed-geometry SCF only; does not release size relaxations',
            'started_utc':datetime.now(timezone.utc).isoformat(),'slurm_job_id':os.environ['SLURM_JOB_ID'],
            'slurm_array_task_id':os.environ.get('SLURM_ARRAY_TASK_ID'),
            'cgroup_limits_bytes':limits,'command':cmd})
        with (a/'pw.out').open('w') as out,(a/'pw.stderr').open('w') as err:
            proc=subprocess.run(cmd,cwd=a,stdout=out,stderr=err)
        r=original.inspect(a,c,m,proc.returncode)
        r.update(returncode=proc.returncode,finished_utc=datetime.now(timezone.utc).isoformat(),
                 reviewed_k7_entrypoint_sha256=sha(__file__),review_manifest_sha256=sha(HERE/'review_manifest.json'))
        if r.get('accepted'):
            r['geometry_force_and_stress_targets_met']=bool(r['fmax_eV_A']<=.001 and
                r['max_force_component_eV_A']<=.0005 and abs(r['stress_QE_GPa'][2][2])<=.05)
            lattice=' '.join(format(v,'.15g') for row in r['cell_A'] for v in row)
            (a/'final.extxyz').write_text(str(r['n_atoms'])+'\n'+f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"\n'+'\n'.join('Al '+' '.join(format(v,'.15g') for v in row) for row in r['positions_A'])+'\n')
        original.dump(a/'result.json',r)
        print(json.dumps({'case':c['id'],'accepted':r['accepted'],'status':r['status'],'error':r.get('error')}))
        return 0 if r['accepted'] else 1

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['verify','run','inspect'])
    p.add_argument('--parent-snapshot',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--index',type=int,choices=[8,9])
    p.add_argument('--attempt',type=Path)
    a=p.parse_args();original,m,review=load_and_verify(a.parent_snapshot,a.root)
    if a.mode=='verify':
        print(json.dumps({'accepted':True,'mode':'verify_no_calculation','allowed_indices':[8,9],
                'cases':[m['cases'][i]['id'] for i in [8,9]],'parent_scripts_and_inputs':'unchanged',
                'review_manifest_sha256':sha(HERE/'review_manifest.json'),
                'no_new_size_relaxations_released':True},indent=2));return 0
    c=reviewed_case(m,a.index)
    if a.mode=='inspect':
        require(a.attempt is not None,'Attempt required')
        r=original.inspect(a.attempt,c,m);print(json.dumps(r,indent=2));return 0 if r['accepted'] else 1
    return run(original,m,review,a.parent_snapshot,a.root,a.index)

if __name__=='__main__':raise SystemExit(main())
