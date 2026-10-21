"""Bounded restart/diagnostic adapter. Frozen scientific inspectors are reused."""
from pathlib import Path
import argparse, json, os, re, shutil, sys, subprocess, hashlib
from datetime import datetime, timezone
import run_bulk_k4_validation_20261015 as BULK
import run_nano_qe_memory_recovery_20261001 as RESOURCE
import restart_adapter as RESTART
import k5_diagnostics as K5

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def require(x,msg):
 if not x:raise ValueError(msg)
def verify(pkg):
 sm=json.loads((pkg/'SOURCE_MANIFEST.json').read_text())
 for rel,h in sm['files'].items():require(sha(pkg/rel)==h,'Changed package file '+rel)
 m=json.loads((pkg/'campaign.json').read_text())
 old=BULK.verify(pkg/'frozen_bulk')
 require([c['id'] for c in m['cases']]==[c['id'] for c in old['cases']],'Five-state inventory changed')
 require(m['binary_sha256']==old['binary_sha256'],'Binary identity changed')
 for i,c in enumerate(m['cases']):
  require(c['prefix']==old['cases'][i]['prefix'],'Parent prefix changed')
  if i>=2:
   inp=(pkg/c['restart_input']).read_text();parent=(pkg/'frozen_bulk'/old['cases'][i]['input']).read_text()
   parsed=BULK.BASE.parse_input(inp)
   require(BULK.BASE.unquote(parsed['blocks']['electrons']['startingpot'])=='file' and BULK.BASE.unquote(parsed['blocks']['electrons']['startingwfc'])=='file','Interrupted restart must read saved electronic state')
   restored=inp.replace("restart_mode = 'restart'","restart_mode = 'from_scratch'").replace("startingpot = 'file'","startingpot = 'atomic'").replace("startingwfc = 'file'","startingwfc = 'atomic+random'")
   require(restored==parent,'Restart physics/input changed')
 return m,old
def check_parent_evidence(c):
 p=Path(c['parent_attempt'])
 for rel,h in c['parent_evidence_sha256'].items():require(sha(p/rel)==h,'Parent changed '+rel)
def resumed_inspect(a,c,oldcase,oldmanifest,rc):
 require(rc==0,'MPI nonzero return')
 expected=c['restart_input_sha256'];require(sha(a/'pw.in')==expected,'Runtime restart input changed')
 require(sha(a/'pseudo/Al_PAW_PBE.UPF')==oldmanifest['pseudo_sha256'],'Runtime PP changed')
 r=BULK.endpoint((a/'pw.out').read_text(errors='replace'),(a/'pw.stderr').read_text(errors='replace'),BULK.BASE.parse_input((a/'pw.in').read_text()),a/'scratch'/(c['prefix']+'.save')/'data-file-schema.xml',rc)
 require(r['accepted'],'Strict full-cell relaxation endpoint failed')
 prov=json.loads((a/'provenance.json').read_text())
 require(prov['copied_restart_verified'] and sha(a/'restart_copy_manifest.json')==prov['restart_copy_manifest_sha256'],'Restart copy record changed')
 r.update(case=c['id'],status='accepted_bulk_k4_restarted_relaxation',returncode=rc,
  hashes={'pw.in':sha(a/'pw.in'),'pw.out':sha(a/'pw.out'),'xml':sha(a/'scratch'/(c['prefix']+'.save')/'data-file-schema.xml'),'PP':sha(a/'pseudo/Al_PAW_PBE.UPF')})
 return r
def accepted_parent(pkg,root,m,old,i):
 c=m['cases'][i]
 if i<2:
  check_parent_evidence(c);a=Path(c['parent_attempt']);r=BULK.inspect(a,old['cases'][i],old)
  require(r.get('accepted'),'Original k4 reference no longer accepted')
 else:
  a=root/c['id']/'k4_resume';saved=json.loads((a/'result.json').read_text())
  require(saved.get('accepted') and saved.get('returncode')==0,'Resumed k4 parent unaccepted')
  r=resumed_inspect(a,c,old['cases'][i],old,0)
 return a,r
def save_relaxed(a,r):
 dump(a/'final_geometry.json',r['final_geometry']);dump(a/'trajectory.json',r.get('trajectory',[]))
 lattice=' '.join(format(v,'.14g') for row in r['cell_A'] for v in row)
 (a/'final.extxyz').write_text(str(r['n_atoms'])+'\n'+f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"\n'+'\n'.join('Al '+' '.join(format(v,'.14g') for v in row) for row in r['positions_A'])+'\n')
def run(pkg,root,i,mode):
 import fcntl
 m,old=verify(pkg);c=m['cases'][i]
 require(mode!='resume' or i>=2,'Do not recompute accepted 0V/1V k4')
 require(sys.platform=='linux' and os.environ.get('SLURM_JOB_ID'),'Slurm allocation required')
 require(sha(Path(m['binary']))==m['binary_sha256'],'QE binary changed')
 require(not any((p/'.git').exists() for p in [root,*root.parents]),'Outputs inside Git')
 case_root=root/c['id'];case_root.mkdir(parents=True,exist_ok=True)
 lock=(case_root/('.'+mode+'.lock')).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 resources={'resources':{'nodes':1,'ntasks':29,'memory_GiB':64 if mode=='resume' else 128}}
 limits=RESOURCE.memory_limits();RESOURCE.allocation_guard(resources,os.environ,min(limits.values()))
 a=case_root/('k4_resume' if mode=='resume' else 'k5_check');require(not a.exists(),'Attempt already exists; refusing duplicate')
 if mode=='scf':parent,ref=accepted_parent(pkg,root,m,old,i)
 else:check_parent_evidence(c)
 a.mkdir();(a/'pseudo').mkdir();shutil.copyfile(pkg/'frozen_bulk'/old['pseudo_file'],a/'pseudo/Al_PAW_PBE.UPF')
 try:
  prov={'case':c['id'],'mode':mode,'runner_sha256':sha(__file__),'source_manifest_sha256':sha(pkg/'SOURCE_MANIFEST.json'),
   'pseudo_sha256':old['pseudo_sha256'],'binary_sha256':m['binary_sha256'],'slurm_job_id':os.environ['SLURM_JOB_ID'],
   'array_index':i,'started_utc':datetime.now(timezone.utc).isoformat(),'cgroup_limits_bytes':limits}
  if mode=='resume':
   shutil.copyfile(pkg/c['restart_input'],a/'pw.in')
   dump(a/'staging_status.json',{'phase':'copying_verified_parent_checkpoint','job':os.environ['SLURM_JOB_ID']})
   parent=RESTART.copy_restart(c,a)
   prov.update(parent_attempt=str(parent),restart_mode='restart',copied_restart_verified=True,restart_copy_manifest_sha256=sha(a/'restart_copy_manifest.json'))
  else:
   text,scfcase=K5.make_input((parent/'pw.in').read_text(),ref,c['id'])
   (a/'pw.in').write_text(text);scfcase['input_sha256']=sha(a/'pw.in');dump(a/'case.json',scfcase)
   dump(a/'k4_reference.json',ref)
   prov.update(parent_attempt=str(parent),parent_evidence_sha256={rel:sha(parent/rel) for rel in ['pw.in','pw.out','result.json','scratch/'+c['prefix']+'.save/data-file-schema.xml']},scope='fixed k4 geometry k5 diagnostic, not a relaxed endpoint')
  dump(a/'provenance.json',prov)
  cmd=RESOURCE.launch_command(m['binary']);dump(a/'staging_status.json',{'phase':'QE_launched','utc':datetime.now(timezone.utc).isoformat(),'command':cmd})
  with (a/'pw.out').open('w') as out,(a/'pw.stderr').open('w') as err:p=subprocess.run(cmd,cwd=a,stdout=out,stderr=err)
  if mode=='resume':r=resumed_inspect(a,c,old['cases'][i],old,p.returncode)
  else:r=K5.inspect(a,scfcase,p.returncode)
  r['returncode']=p.returncode
  if mode=='resume' and r.get('accepted'):save_relaxed(a,r)
 except Exception as exc:r={'case':c['id'],'accepted':False,'mode':mode,'status':'staging_execution_or_acceptance_failed','error':repr(exc),'returncode':locals().get('p').returncode if 'p' in locals() else None}
 r['finished_utc']=datetime.now(timezone.utc).isoformat();dump(a/'result.json',r)
 print(json.dumps({k:r.get(k) for k in ['case','accepted','status','error']}))
 return 0 if r.get('accepted') else 1
def evaluate(pkg,root):
 m,old=verify(pkg);b4={};b5={}
 for i,c in enumerate(m['cases']):
  a,ref=accepted_parent(pkg,root,m,old,i);b4[c['id']]=ref
  diag_root=Path(m['reference_diagnostic_root']) if i<2 else root
  a=diag_root/c['id']/'k5_check';saved=json.loads((a/'result.json').read_text());require(saved.get('returncode')==0,'k5 exited nonzero')
  sc=json.loads((a/'case.json').read_text());r=K5.inspect(a,sc,0)
  require(r.get('accepted'),'k5 SCF not accepted');b5[c['id']]=r
 result=K5.evaluate(b4,b5);dump(root/'k4_k5_sensitivity.json',result);return result
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['verify','resume','scf','evaluate']);p.add_argument('--package',type=Path,required=True);p.add_argument('--root',type=Path);p.add_argument('--index',type=int);a=p.parse_args()
 if a.mode=='verify':verify(a.package);print('Frozen inputs and adapter verified');return 0
 if a.mode=='evaluate':
  try:r=evaluate(a.package,a.root);print(json.dumps(r));return 0
  except Exception as exc:dump(a.root/'k4_k5_sensitivity.json',{'complete':False,'error':repr(exc)});return 1
 return run(a.package,a.root.resolve(),a.index,a.mode)
if __name__=='__main__':sys.exit(main())
