"""Resource-only recovery of four interrupted nano vc-relax jobs.

Scientific input bytes and the original final-state inspector are retained.
This runner makes allocation mistakes and lost-rank exits fail the job.
"""
from pathlib import Path
import argparse,importlib.util,json,os,shutil,subprocess,sys
from datetime import datetime,timezone

SPEC=importlib.util.spec_from_file_location('original_qe_runner',Path(__file__).with_name('run_luder_qe_20260930.py'))
BASE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(BASE)
sha=BASE.sha;dump=BASE.dump;require=BASE.require

def verify(campaign):
    m=BASE.verify(campaign)
    require([c['id'] for c in m['cases']]==['nano_R15_0V','nano_R15_1V','nano_R20_0V','nano_R20_1V'],'Unexpected recovery case set')
    for c in m['cases']:
        require(c['calculation']=='vc-relax' and c['kind']=='nanocolumn_pilot_axial_vcrelax','Recovery must be axial nano relaxation')
        require(c['input_sha256']==c['original_input_sha256'],'Scientific input changed')
        require(c['resources']['ntasks']==29 and c['resources']['nodes']==1,'MPI layout changed')
        require(c['resources']['memory_GiB'] == (128 if 'R15' in c['id'] else 256),'Wrong size-specific allocation')
    return m

def launch_command(binary):
    return ['srun','--mpi=pmi2','--kill-on-bad-exit=1','--oom-kill-step=1','--wait=60','-n','29',binary,'-in','pw.in']

def allocation_guard(c,env,limit_bytes):
    require(int(env.get('SLURM_NTASKS','0'))==29,'Need exactly 29 MPI ranks')
    require(int(env.get('SLURM_JOB_NUM_NODES','0'))==1,'Need one node')
    require(int(env.get('SLURM_MEM_PER_NODE','0'))>=c['resources']['memory_GiB']*1024,'Slurm memory below recovery request')
    require(limit_bytes>=c['resources']['memory_GiB']*2**30,'Enforced cgroup memory below recovery request')

def memory_limits():
    found={}
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        no,controllers,rel=line.split(':',2)
        if 'memory' in controllers.split(','):
            root=Path('/sys/fs/cgroup/memory');p=root/rel.lstrip('/');name='memory.limit_in_bytes'
        elif no=='0':
            root=Path('/sys/fs/cgroup');p=root/rel.lstrip('/');name='memory.max'
        else:continue
        while p.is_relative_to(root):
            q=p/name
            if q.exists():
                value=q.read_text().strip()
                if value!='max':found[str(q)]=int(value)
            if p==root:break
            p=p.parent
    require(bool(found),'Cannot verify enforced memory allocation')
    return found

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['verify','run','inspect']);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--index',type=int);p.add_argument('--root',type=Path);p.add_argument('--attempt',type=Path)
    args=p.parse_args();m=verify(args.campaign)
    if args.mode=='verify':print(json.dumps({'verified':4,'original_input_bytes_preserved':True}));return 0
    c=m['cases'][args.index]
    if args.mode=='inspect':
        r=BASE.inspect(args.attempt,c);print(json.dumps(r,indent=2));return 0 if r['accepted'] else 1
    require(sys.platform=='linux' and os.environ.get('SLURM_JOB_ID'),'Run requires a Slurm Linux allocation')
    require(sha(m['binary'])==m['binary_sha256'],'QE binary changed')
    limits=memory_limits();allocation_guard(c,os.environ,min(limits.values()))
    root=args.root.resolve();require(not any((x/'.git').exists() for x in [root,*root.parents]),'Output inside Git repository')
    a=root/c['id']/'attempt_001';a.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(args.campaign/c['input'],a/'pw.in');(a/'pseudo').mkdir();shutil.copyfile(args.campaign/m['pseudo_file'],a/'pseudo/Al_PAW_PBE.UPF')
    require(sha(a/'pw.in')==c['input_sha256'],'Input changed during copy')
    shutil.copyfile(args.campaign/'qe_manifest.json',a/'qe_manifest.json')
    cmd=launch_command(m['binary'])
    dump(a/'provenance.json',{'case':c,'pseudo_sha256':m['pseudo_sha256'],'binary_sha256':m['binary_sha256'],'runner_sha256':sha(__file__),'inspector_sha256':sha(Path(BASE.__file__)),'slurm_job_id':os.environ['SLURM_JOB_ID'],'slurm_array_task_id':os.environ.get('SLURM_ARRAY_TASK_ID'),'started_utc':datetime.now(timezone.utc).isoformat(),'cgroup_limits_bytes':limits,'selected_slurm_environment':{k:os.environ.get(k) for k in ['SLURM_NTASKS','SLURM_JOB_NUM_NODES','SLURM_MEM_PER_NODE','SLURM_JOB_NODELIST']},'command':cmd,'restart_policy':'Fresh identical initial input; interrupted jobs had no complete force/ionic step; old scratch untouched.'})
    with (a/'pw.out').open('w') as out,(a/'pw.stderr').open('w') as err:
        proc=subprocess.run(cmd,cwd=a,stdout=out,stderr=err)
    r=BASE.inspect(a,c);r['returncode']=proc.returncode
    if proc.returncode:r.update(accepted=False,status='process_failed')
    r['finished_utc']=datetime.now(timezone.utc).isoformat();dump(a/'result.json',r)
    print(json.dumps({'case':c['id'],'accepted':r['accepted'],'status':r['status'],'error':r.get('error')}))
    return 0 if r['accepted'] else 1

if __name__=='__main__':sys.exit(main())
