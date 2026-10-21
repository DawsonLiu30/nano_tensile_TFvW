"""Two reviewed k9 input overlays; reuse frozen k7 entry and original inspector.

No parent input, parser, inspector, gate, installed program or finished result
is modified. This entry cannot launch new-size relaxations or later meshes.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,hashlib,importlib.util,json,os,shutil,subprocess,sys
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def select(m,index):
 require(index in (0,1),'Only R15 0V/1V k9 indices0/1 are reviewed')
 c=m['cases'][index]
 require(c['id']==f'nano_R15_{index}V_k9_SCF_20261017' and c['kmesh']==[1,1,9]
         and c['stage']=='reviewed_k9' and c['calculation']=='scf','Wrong reviewed case')
 return c
def locate_k7(root,case,v):
 a=root/case/'attempt_001'
 if not a.exists():a=root/f'QE_R15_{v}V_K7'
 require(a.exists(),'k7 evidence missing');return a
def verify_overlay(original,parent_manifest,overlay,parent):
 require(len(overlay['cases'])==2,'Overlay must contain exactly two cases')
 for v in [0,1]:
  c=select(overlay,v);old=parent_manifest['cases'][8+v]
  expected=copy.deepcopy(old);cid=f'nano_R15_{v}V_k9_SCF_20261017'
  expected.update(id=cid,prefix=cid,stage='reviewed_k9',kmesh=[1,1,9],
                  input=f'inputs/{cid}/pw.in',initial_structure=f'inputs/{cid}/initial.extxyz')
  text=(parent/old['input']).read_text().replace(old['prefix'],cid).replace('1 1 7 0 0 0','1 1 9 0 0 0')
  actual=HERE/c['input'];require(actual.resolve().is_relative_to(HERE),'Overlay input escapes')
  require(actual.read_text()==text,'Overlay differs beyond kmesh and prefix')
  expected['input_sha256']=sha(actual)
  require(c==expected,'Unexpected overlay metadata/geometry/resources change')
  require(sha(HERE/c['initial_structure'])==sha(parent/old['initial_structure'])==c['initial_structure_sha256'],'Initial structure changed')
  require(c['nat'] in (95,96) and c['nat']<=250,'Wrong atom count')
  require(min(c['cell_A'][i][i] for i in range(3))>=10 and c['pbc']==[True]*3,'PBC or image translation changed')
 return overlay
def load_and_verify(parent,root,k7_entry,k7_root):
 review=json.loads((HERE/'overlay_manifest.json').read_text())
 require(review['review_id']=='R15_k9_only_20261017_v1' and review['allowed_indices']==[0,1],'Wrong review')
 for rel,h in review['entrypoint_sha256'].items():
  path=(HERE/rel).resolve();require(path.is_relative_to(HERE) and sha(path)==h,'Overlay file changed: '+rel)
 require(sha(k7_entry)==review['parent_k7_entry_sha256'],'Frozen k7 entry changed')
 require(sha(k7_entry.with_name('review_manifest.json'))==review['parent_k7_review_sha256'],'Parent k7 review changed')
 k7=load('frozen_k7_entry',k7_entry)
 original,parent_manifest,old_review=k7.load_and_verify(parent,root)
 require(sha(parent/'qe_manifest.json')==review['parent_campaign_sha256'],'Parent campaign changed')
 verify_overlay(original,parent_manifest,review,parent)
 pairs={5:[],7:[]}
 for v in [0,1]:
  c=parent_manifest['cases'][8+v];a=locate_k7(k7_root,c['id'],v)
  for rel,h in review['k7_stageproof'][str(v)]['sha256'].items():require(sha(a/rel)==h,'k7 evidence changed: '+str(a/rel))
  r=json.loads((a/'result.json').read_text());require(r['accepted'] is True and r['returncode']==0 and r['status']=='accepted_scf_diagnostic','Parent k7 SCF not accepted')
  require(r['case']==c['id'] and r['final_scf_accuracy_Ry']<=1e-10,'Wrong parent endpoint')
  pairs[7].append(r['energy_eV'])
  c5=parent_manifest['cases'][6+v];r5=json.loads((root/c5['id']/'attempt_001/result.json').read_text());pairs[5].append(r5['energy_eV'])
 delta=(pairs[7][1]-pairs[7][0])-(pairs[5][1]-pairs[5][0])
 require(abs(delta-review['k5_to_k7_pair_change_eV'])<1e-12,'Stageproof energy difference changed')
 require(abs(delta)>.005,'Review expected unresolved k5-to-k7 paired sensitivity')
 return original,parent_manifest,review
def run(original,parent_manifest,m,parent,root,index):
 import fcntl
 c=select(m,index)
 require(sys.platform=='linux' and os.environ.get('SLURM_JOB_ID'),'Run requires Linux Slurm allocation')
 require(str(root.resolve())==m['parent_runroot'],'Use reviewed unique case runroot')
 require(not any((p/'.git').exists() for p in [root,*root.parents]),'Output inside Git')
 require(sha(parent_manifest['binary'])==parent_manifest['binary_sha256'],'QE binary changed')
 limits=original.memory_limits();original.allocation_guard(c,os.environ,min(limits.values()))
 case_root=root/c['id'];case_root.mkdir(parents=True,exist_ok=True)
 with (case_root/'.execution.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  a=case_root/'attempt_001';a.mkdir(exist_ok=False)
  shutil.copyfile(HERE/c['input'],a/'pw.in');(a/'pseudo').mkdir()
  shutil.copyfile(parent/parent_manifest['pseudo_file'],a/'pseudo/Al_PAW_PBE.UPF')
  require(sha(a/'pw.in')==c['input_sha256'] and sha(a/'pseudo/Al_PAW_PBE.UPF')==parent_manifest['pseudo_sha256'],'Runtime copy changed')
  shutil.copyfile(HERE/'overlay_manifest.json',a/'k9_overlay_manifest.json')
  shutil.copyfile(parent/'qe_manifest.json',a/'parent_qe_manifest.json');shutil.copyfile(__file__,a/'run_reviewed_k9.py')
  cmd=original.launch_command(parent_manifest['binary'])
  original.dump(a/'provenance.json',{'case':c,'pseudo_sha256':parent_manifest['pseudo_sha256'],
       'binary_sha256':parent_manifest['binary_sha256'],'runner_sha256':sha(__file__),
       'parent_runner_sha256':sha(original.__file__),'nano_inspector_sha256':sha(original.BASE.__file__),
       'overlay_manifest_sha256':sha(HERE/'overlay_manifest.json'),'parent_campaign_sha256':sha(parent/'qe_manifest.json'),
       'parent_k7_entry_sha256':m['parent_k7_entry_sha256'],'parent_k7_review_sha256':m['parent_k7_review_sha256'],
       'k7_stageproof':m['k7_stageproof'],'scope':'R15 k9 fixed-geometry diagnostic only; no size-case release',
       'started_utc':datetime.now(timezone.utc).isoformat(),'slurm_job_id':os.environ['SLURM_JOB_ID'],
       'slurm_array_task_id':os.environ.get('SLURM_ARRAY_TASK_ID'),'cgroup_limits_bytes':limits,'command':cmd})
  with (a/'pw.out').open('w') as out,(a/'pw.stderr').open('w') as err:proc=subprocess.run(cmd,cwd=a,stdout=out,stderr=err)
  r=original.inspect(a,c,parent_manifest,proc.returncode)
  r.update(returncode=proc.returncode,finished_utc=datetime.now(timezone.utc).isoformat(),
           overlay_manifest_sha256=sha(HERE/'overlay_manifest.json'))
  if r.get('accepted'):
   r['geometry_force_and_stress_targets_met']=bool(r['fmax_eV_A']<=.001 and r['max_force_component_eV_A']<=.0005 and abs(r['stress_QE_GPa'][2][2])<=.05)
   lattice=' '.join(format(v,'.15g') for row in r['cell_A'] for v in row)
   (a/'final.extxyz').write_text(str(r['n_atoms'])+'\n'+f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"\n'+'\n'.join('Al '+' '.join(format(v,'.15g') for v in row) for row in r['positions_A'])+'\n')
  original.dump(a/'result.json',r);print(json.dumps({'case':c['id'],'accepted':r['accepted'],'status':r['status'],'error':r.get('error')}))
  return 0 if r['accepted'] else 1
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['verify','run','inspect'])
 p.add_argument('--parent-snapshot',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
 p.add_argument('--k7-entry',type=Path,required=True);p.add_argument('--k7-evidence-root',type=Path)
 p.add_argument('--index',type=int,choices=[0,1]);p.add_argument('--attempt',type=Path)
 a=p.parse_args();original,parent_manifest,m=load_and_verify(a.parent_snapshot,a.root,a.k7_entry,a.k7_evidence_root or a.root)
 if a.mode=='verify':
  print(json.dumps({'accepted':True,'mode':'verify_no_calculation','cases':[c['id'] for c in m['cases']],
       'parent_runner_and_inspector':'unchanged','overlay_manifest_sha256':sha(HERE/'overlay_manifest.json'),
       'no_size_cases_released':True},indent=2));return 0
 c=select(m,a.index)
 if a.mode=='inspect':
  require(a.attempt is not None,'Attempt required');r=original.inspect(a.attempt,c,parent_manifest)
  print(json.dumps(r,indent=2));return 0 if r['accepted'] else 1
 return run(original,parent_manifest,m,a.parent_snapshot,a.root,a.index)
if __name__=='__main__':raise SystemExit(main())
