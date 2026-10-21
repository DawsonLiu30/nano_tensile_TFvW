"""Two R20 continuations: copy and verify saved state, then resume identical physics."""
from pathlib import Path
import argparse, importlib.util, json, os, shutil, subprocess, sys, hashlib
from datetime import datetime, timezone

spec=importlib.util.spec_from_file_location('resource_guard',Path(__file__).with_name('run_nano_qe_memory_recovery_20261001.py'))
GUARD=importlib.util.module_from_spec(spec);spec.loader.exec_module(GUARD)
BASE=GUARD.BASE;require=BASE.require;dump=BASE.dump
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def verify(campaign):
 m=BASE.verify(campaign)
 require([c['id'] for c in m['cases']]==['nano_R20_0V','nano_R20_1V'],'Unexpected restart cases')
 for c in m['cases']:
  text=(campaign/c['input']).read_text()
  parent=(campaign/c['parent_input']).read_text()
  restored=text.replace("restart_mode = 'restart'","restart_mode = 'from_scratch'").replace('max_seconds = 255600','max_seconds = 169200')
  require(restored==parent,'Only restart_mode and time budget may change')
  require(sha(campaign/c['parent_input'])==c['parent_input_sha256'],'Parent input hash changed')
  require(c['resources']=={'nodes':1,'ntasks':29,'memory_GiB':256},'Unexpected resources')
 return m
def validate_parent(c):
 p=Path(c['parent_attempt'])
 for rel,h in c['parent_evidence_sha256'].items():require(sha(p/rel)==h,'Parent evidence changed: '+rel)
 text=(p/'pw.out').read_text()
 require('Maximum CPU time exceeded' in text and 'JOB DONE.' in text,'Parent did not stop cleanly')
 r=json.loads((p/'result.json').read_text())
 require(r['returncode']==0 and not r['accepted'],'Parent is not the stopped unaccepted attempt')
 files={f.relative_to(p/'scratch').as_posix():f for f in (p/'scratch').rglob('*') if f.is_file()}
 require(all(not f.is_symlink() for f in files.values()),'Unexpected scratch symlink')
 require(len([n for n in files if n.startswith(c['prefix']+'.wfc')])==29,'Missing distributed wavefunctions')
 require(len([n for n in files if n.startswith(c['prefix']+'.mix')])==29,'Missing distributed mixing files')
 for n in [c['prefix']+'.bfgs',c['prefix']+'.update',c['prefix']+'.restart_scf',c['prefix']+'.save/charge-density.dat',c['prefix']+'.save/data-file-schema.xml']:
  require(n in files and files[n].stat().st_size>0,'Missing restart file '+n)
 return p,files
def copy_restart(c,attempt):
 parent,files=validate_parent(c)
 dest=attempt/'scratch';require(not dest.exists(),'Scratch already exists')
 before={n:(p.stat().st_size,p.stat().st_mtime_ns) for n,p in files.items()}
 dest.mkdir()
 subprocess.run(['cp','-a','--reflink=auto',str(parent/'scratch')+'/.',str(dest)],check=True)
 records={}
 for n,p in files.items():
  h=sha(p);q=dest/n
  require(q.is_file() and q.stat().st_size==before[n][0] and sha(q)==h,'Restart copy mismatch '+n)
  require((p.stat().st_size,p.stat().st_mtime_ns)==before[n],'Parent changed while copying '+n)
  records[n]={'sha256':h,'bytes':before[n][0]}
 dump(attempt/'restart_copy_manifest.json',{'source':str(parent/'scratch'),'files':records,'bytes':sum(v['bytes'] for v in records.values())})
 return parent
def inspect(attempt,c):
 r=BASE.inspect(attempt,c)
 try:
  p=json.loads((attempt/'provenance.json').read_text())
  require(p['restart_mode']=='restart' and p['copied_restart_verified'],'Restart provenance incomplete')
  require(sha(attempt/'restart_copy_manifest.json')==p['restart_copy_manifest_sha256'],'Restart manifest changed')
 except Exception as exc:r.update(accepted=False,status='restart_provenance_failed',error=repr(exc))
 return r
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['verify','run','inspect']);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--index',type=int);p.add_argument('--root',type=Path);p.add_argument('--attempt',type=Path)
 args=p.parse_args();m=verify(args.campaign)
 if args.mode=='verify':print(json.dumps({'verified':2,'only_restart_and_time_changed':True}));return 0
 c=m['cases'][args.index]
 if args.mode=='inspect':
  r=inspect(args.attempt,c);print(json.dumps(r,indent=2));return 0 if r['accepted'] else 1
 require(sys.platform=='linux' and os.environ.get('SLURM_JOB_ID'),'Requires Slurm')
 require(sha(m['binary'])==m['binary_sha256'],'QE binary changed')
 limits=GUARD.memory_limits();GUARD.allocation_guard(c,os.environ,min(limits.values()))
 root=args.root.resolve();require(not any((x/'.git').exists() for x in [root,*root.parents]),'Run output inside Git')
 attempt=root/c['id']/'attempt_001';attempt.mkdir(parents=True,exist_ok=False)
 shutil.copy2(args.campaign/c['input'],attempt/'pw.in');(attempt/'pseudo').mkdir()
 shutil.copy2(args.campaign/m['pseudo_file'],attempt/'pseudo/Al_PAW_PBE.UPF')
 shutil.copy2(args.campaign/'qe_manifest.json',attempt/'qe_manifest.json')
 dump(attempt/'staging_status.json',{'phase':'copying_parent_restart','started_utc':datetime.now(timezone.utc).isoformat(),'job':os.environ['SLURM_JOB_ID']})
 try:
  parent=copy_restart(c,attempt)
  cmd=GUARD.launch_command(m['binary'])
  dump(attempt/'provenance.json',{'case':c,'pseudo_sha256':m['pseudo_sha256'],'binary_sha256':m['binary_sha256'],
   'runner_sha256':sha(__file__),'inspector_sha256':sha(Path(BASE.__file__)),'slurm_job_id':os.environ['SLURM_JOB_ID'],
   'slurm_array_task_id':os.environ.get('SLURM_ARRAY_TASK_ID'),'started_utc':datetime.now(timezone.utc).isoformat(),
   'cgroup_limits_bytes':limits,'command':cmd,'parent_attempt':str(parent),'restart_mode':'restart',
   'copied_restart_verified':True,'restart_copy_manifest_sha256':sha(attempt/'restart_copy_manifest.json')})
  dump(attempt/'staging_status.json',{'phase':'QE_launched','utc':datetime.now(timezone.utc).isoformat()})
  with (attempt/'pw.out').open('w') as out,(attempt/'pw.stderr').open('w') as err:
   proc=subprocess.run(cmd,cwd=attempt,stdout=out,stderr=err)
  r=inspect(attempt,c);r['returncode']=proc.returncode
  if proc.returncode:r.update(accepted=False,status='process_failed')
 except Exception as exc:r={'case':c['id'],'accepted':False,'status':'restart_staging_or_launch_failed','error':repr(exc),'returncode':None}
 r['finished_utc']=datetime.now(timezone.utc).isoformat();dump(attempt/'result.json',r)
 print(json.dumps({k:r.get(k) for k in ['case','accepted','status','returncode','error']}));return 0 if r['accepted'] else 1
if __name__=='__main__':sys.exit(main())
