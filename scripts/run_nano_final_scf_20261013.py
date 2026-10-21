"""Complete the interrupted final SCF of an already converged R20 geometry."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, re, shutil, subprocess, sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
spec=importlib.util.spec_from_file_location('guard',Path(__file__).with_name('run_nano_qe_memory_recovery_20261001.py'))
GUARD=importlib.util.module_from_spec(spec);spec.loader.exec_module(GUARD)
BASE=GUARD.BASE;require=BASE.require;dump=BASE.dump
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def expected_input(parent,geometry):
 text=parent.replace("calculation = 'vc-relax'","calculation = 'scf'").replace('nstep = 1000','nstep = 1')
 text=text.replace("startingpot = 'atomic'","startingpot = 'file'").replace("startingwfc = 'atomic+random'","startingwfc = 'file'")
 text=re.sub(r'&IONS\b.*?^/\s*\n','',text,flags=re.S|re.M)
 text=re.sub(r'&CELL\b.*?^/\s*\n','',text,flags=re.S|re.M)
 return re.sub(r'CELL_PARAMETERS[^\n]*\n.*?(?=K_POINTS)',geometry,text,flags=re.S)

def verify(campaign):
 m=BASE.verify(campaign)
 require(len(m['cases'])==1 and m['cases'][0]['id']=='nano_R20_1V_final_scf','Wrong continuation case')
 c=m['cases'][0]
 require(c['calculation']=='scf' and c['nat']==175,'Wrong calculation or atom count')
 parent=campaign/c['parent_input'];geom=campaign/'final_geometry.card'
 require(sha(parent)==c['parent_input_sha256'],'Parent input changed')
 require(sha(geom)==c['final_geometry_sha256'],'Converged geometry card changed')
 require((campaign/c['input']).read_text()==expected_input(parent.read_text(),geom.read_text()),'Unexpected input change beyond final-SCF continuation')
 require(c['resources']=={'nodes':1,'ntasks':29,'memory_GiB':256},'Changed resource layout')
 require("restart_mode = 'restart'" in (campaign/c['input']).read_text(),'Continuation must read restart state')
 return m

def geometry(ast):
 b=BASE.BOHR
 return ([[float(v)*b for v in ast.find('cell/'+k).text.split()] for k in ['a1','a2','a3']],
         [[float(v)*b for v in x.text.split()] for x in ast.findall('atomic_positions/atom')])

def validate_parent(c):
 p=Path(c['parent_attempt'])
 for rel,h in c['parent_evidence_sha256'].items():require(sha(p/rel)==h,'Parent evidence changed: '+rel)
 text=(p/'pw.out').read_text();marker='Final scf calculation at the relaxed structure'
 require('bfgs converged' in text.lower() and marker in text,'No completed parent geometry optimization')
 require(text.lower().index('bfgs converged')<text.index(marker)<text.index('Maximum CPU time exceeded'),'Stop was not in final SCF')
 require('JOB DONE.' in text and 'Calculation stopped in scf loop' in text,'Parent not cleanly stopped')
 r=json.loads((p/'result.json').read_text());require(r['returncode']==0 and not r['accepted'],'Parent state differs')
 root=ET.parse(p/'scratch'/(c['prefix']+'.save/data-file-schema.xml')).getroot()
 require(root.findtext('output/convergence_info/scf_conv/convergence_achieved')=='false','Parent already electronically converged')
 for ast in [root.find('output/atomic_structure'),root.findall('step')[-1].find('atomic_structure')]:
  cell,pos=geometry(ast)
  require(len(pos)==c['nat'],'Parent atom count differs')
  require(max(abs(a-b) for row,ref in zip(cell,c['cell_A']) for a,b in zip(row,ref))<2e-7,'Parent final cell differs')
  require(max(abs(a-b) for row,ref in zip(pos,c['positions_A']) for a,b in zip(row,ref))<2e-7,'Parent final positions differ')
 files={f.relative_to(p/'scratch').as_posix():f for f in (p/'scratch').rglob('*') if f.is_file()}
 require(all(not f.is_symlink() for f in files.values()),'Unexpected restart symlink')
 for suffix in ['wfc','mix','restart_scf','restart_k']:
  require(len([n for n in files if re.fullmatch(re.escape(c['prefix'])+r'\.'+suffix+r'\d*',n)])==29,'Missing distributed '+suffix)
 for n in [c['prefix']+'.save/charge-density.dat',c['prefix']+'.save/data-file-schema.xml',c['prefix']+'.save/paw.txt']:
  require(n in files and files[n].stat().st_size>0,'Missing saved '+n)
 return p,files

def copy_restart(c,attempt):
 parent,files=validate_parent(c);dest=attempt/'scratch'
 require(not dest.exists(),'Destination scratch exists')
 before={n:(p.stat().st_size,p.stat().st_mtime_ns) for n,p in files.items()};dest.mkdir()
 subprocess.run(['cp','-a','--reflink=auto',str(parent/'scratch')+'/.',str(dest)],check=True)
 records={}
 for n,p in files.items():
  h=sha(p);q=dest/n
  require(q.is_file() and q.stat().st_size==before[n][0] and sha(q)==h,'Copied restart mismatch: '+n)
  require((p.stat().st_size,p.stat().st_mtime_ns)==before[n],'Parent changed during copy')
  records[n]={'sha256':h,'bytes':before[n][0]}
 dump(attempt/'restart_copy_manifest.json',{'source':str(parent/'scratch'),'files':records,'bytes':sum(r['bytes'] for r in records.values())})

def accept_final_forces(r):
 require(r['fmax_eV_A']<=.001 and r['max_force_component_eV_A']<=.0005,'Final atomic force gate failed')
 require(abs(r['stress_QE_GPa'][2][2])<=.05,'Final axial stress gate failed')

def inspect(attempt,c):
 r=BASE.inspect(attempt,c)
 if not r['accepted']:return r
 try:
  validate_parent(c)
  p=json.loads((attempt/'provenance.json').read_text())
  require(p['copied_restart_verified'],'Restart copy unverified')
  require(sha(attempt/'restart_copy_manifest.json')==p['restart_copy_manifest_sha256'],'Restart manifest changed')
  require('Maximum CPU time exceeded' not in (attempt/'pw.out').read_text(),'Final SCF hit time limit')
  accept_final_forces(r)
  r.update(status='accepted_axial_relaxation_with_separate_final_scf',parent_geometry_converged=True,parent_attempt=c['parent_attempt'],scope='Joined parent axial vc-relax and converged final-cell SCF; nano sampling convergence remains unvalidated')
 except Exception as exc:r.update(accepted=False,status='final_scf_not_accepted',error=repr(exc))
 return r

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['verify','run','inspect']);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--root',type=Path);p.add_argument('--attempt',type=Path)
 args=p.parse_args();m=verify(args.campaign);c=m['cases'][0]
 if args.mode=='verify':print(json.dumps({'verified':1,'same_science':True,'final_geometry_fixed':True}));return 0
 if args.mode=='inspect':
  r=inspect(args.attempt,c);print(json.dumps(r,indent=2));return 0 if r['accepted'] else 1
 require(sys.platform=='linux' and os.environ.get('SLURM_JOB_ID'),'Requires Slurm')
 require(sha(m['binary'])==m['binary_sha256'],'QE executable changed')
 limits=GUARD.memory_limits();GUARD.allocation_guard(c,os.environ,min(limits.values()))
 root=args.root.resolve();require(not any((x/'.git').exists() for x in [root,*root.parents]),'Output inside Git')
 a=root/c['id']/'attempt_001';a.mkdir(parents=True,exist_ok=False)
 shutil.copy2(args.campaign/c['input'],a/'pw.in');(a/'pseudo').mkdir()
 shutil.copy2(args.campaign/m['pseudo_file'],a/'pseudo/Al_PAW_PBE.UPF');shutil.copy2(args.campaign/'qe_manifest.json',a/'qe_manifest.json')
 dump(a/'staging_status.json',{'phase':'copying_and_hashing_parent','job':os.environ['SLURM_JOB_ID']})
 try:
  copy_restart(c,a);cmd=GUARD.launch_command(m['binary'])
  dump(a/'provenance.json',{'case':c,'pseudo_sha256':m['pseudo_sha256'],'binary_sha256':m['binary_sha256'],'runner_sha256':sha(__file__),'inspector_sha256':sha(BASE.__file__),'slurm_job_id':os.environ['SLURM_JOB_ID'],'started_utc':datetime.now(timezone.utc).isoformat(),'command':cmd,'cgroup_limits_bytes':limits,'copied_restart_verified':True,'restart_copy_manifest_sha256':sha(a/'restart_copy_manifest.json')})
  dump(a/'staging_status.json',{'phase':'QE_launched','utc':datetime.now(timezone.utc).isoformat()})
  with (a/'pw.out').open('w') as out,(a/'pw.stderr').open('w') as err:proc=subprocess.run(cmd,cwd=a,stdout=out,stderr=err)
  r=inspect(a,c);r['returncode']=proc.returncode
  if proc.returncode:r.update(accepted=False,status='process_failed')
 except Exception as exc:r={'case':c['id'],'accepted':False,'status':'staging_or_execution_failed','error':repr(exc)}
 r['finished_utc']=datetime.now(timezone.utc).isoformat();dump(a/'result.json',r)
 print(json.dumps({k:r.get(k) for k in ['case','accepted','status','error']}));return 0 if r['accepted'] else 1
if __name__=='__main__':sys.exit(main())
