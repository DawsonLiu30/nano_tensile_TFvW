"""Five-state validation runner: reuses established bulk full-stress parser.

No scheduling is performed. Each new run requires a Slurm allocation and uses
a new attempt directory. All gates apply to the rebuilt final-cell SCF.
"""
from pathlib import Path
import argparse,hashlib,json,math,os,re,shutil,subprocess,sys
import xml.etree.ElementTree as ET
from datetime import datetime,timezone
import run_bulk_vacancy_qe_vcrelax_20260925 as BASE
from run_nano_qe_memory_recovery_20261001 import memory_limits,allocation_guard,launch_command
HA=27.211386245988;BOHR=.5291772105638411;GPA=29421.02648438959
PP='450c947cbbf30c33a5a77685bbe9a270947816f8a93837c02c749d0ee4e3ab4a'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def verify(campaign):
 m=json.loads((campaign/'qe_manifest.json').read_text());require(m['campaign']=='al_bulk_k4_validation_20261015','Wrong campaign')
 require([c['id'] for c in m['cases']]==['0V','1V','2V_1NN_D110','2V_2NN_D100','2V_5NN_D310'],'Wrong five-state inventory')
 require(sha(campaign/m['pseudo_file'])==m['pseudo_sha256']==PP,'PP mismatch')
 require(sha(campaign/'protocol.json')==m['protocol_sha256'],'Protocol changed')
 for c in m['cases']:
  p=campaign/c['input'];require(p.resolve().is_relative_to(campaign.resolve()),'Input escaped campaign');require(sha(p)==c['input_sha256'],'Input hash mismatch')
  x=BASE.parse_input(p.read_text());b=x['blocks'];num=BASE.number;uq=BASE.unquote
  require(x['nat']==c['nat'] and x['cell_A']==c['cell_A'] and max(abs(a-d) for r,s in zip(x['positions_A'],c['positions_A']) for a,d in zip(r,s))<1e-10,'Initial geometry changed')
  required={'control':{'nstep':200,'etot_conv_thr':1e-6,'forc_conv_thr':1.94469038114889e-5,'max_seconds':255600},'system':{'ecutwfc':58.798915481,'ecutrho':470.3913238484,'nbnd':200,'degauss':.02,'nspin':1,'tot_charge':0},'electrons':{'conv_thr':1e-10,'electron_maxstep':300,'mixing_beta':.2,'mixing_ndim':4,'diago_david_ndim':2},'ions':{'trust_radius_min':1e-6},'cell':{'press':0,'press_conv_thr':.5}}
  for block,kv in required.items():
   for k,v in kv.items():require(num(b[block].get(k))==v,'Changed setting '+block+'.'+k)
  require(x['cards']['K_POINTS']['rows']==[['4','4','4','0','0','0']],'Wrong validation mesh')
  require(set(b['system'])=={'ibrav','nat','ntyp','nbnd','ecutwfc','ecutrho','occupations','smearing','degauss','nspin','tot_charge','nosym','noinv'},'Unexpected SYSTEM setting')
  require(set(b['electrons'])=={'conv_thr','electron_maxstep','scf_must_converge','mixing_mode','mixing_beta','mixing_ndim','diagonalization','diago_david_ndim','startingpot','startingwfc'},'Unexpected ELECTRONS setting')
  require(uq(b['system']['smearing'])=='mp' and b['system']['nosym']=='.true.' and b['system']['noinv']=='.true.','Smearing/symmetry changed')
  require(uq(b['electrons']['mixing_mode'])=='plain' and uq(b['electrons']['diagonalization'])=='david','Electronic solver changed')
  require(uq(b['control']['pseudo_dir'])=='./pseudo' and uq(b['control']['outdir'])=='./scratch','PP/scratch path changed')
  require(uq(b['control']['prefix'])==c['prefix'] and uq(b['control']['restart_mode'])=='from_scratch','Invalid fresh attempt')
  require(uq(b['electrons']['startingpot'])=='atomic' and uq(b['electrons']['startingwfc'])=='atomic+random','Fresh electronic state changed')
  require(c['resources']=={'nodes':1,'ntasks':29,'memory_GiB':64,'slurm_hours':72,'array_concurrency':5},'Resource request changed')
 return m
def endpoint(text,err,parsed,xml_path,returncode=0):
 """Established stdout gates plus strict, full-precision XML agreement."""
 old=BASE.parse_output(text,parsed,returncode=returncode,stderr=err,force_target_eV_A=.001)
 require(old['usable'],'Existing full-stress bulk parser rejected: '+str(old['reasons']+old['data_quality_reasons']))
 o=ET.parse(xml_path).getroot().find('output');require(o is not None,'Missing XML output')
 require(o.findtext('convergence_info/scf_conv/convergence_achieved')=='true','SCF XML not converged')
 scf=2*float(o.findtext('convergence_info/scf_conv/scf_error'));require(scf<=1e-10,'Final SCF accuracy')
 require(o.findtext('convergence_info/opt_conv/convergence_achieved')=='true','Final optimizer XML not converged')
 a=o.find('atomic_structure');n=int(a.attrib['nat']);require(n==parsed['nat'],'nat mismatch')
 require(abs(float(o.findtext('band_structure/nelec'))-3*n)<1e-8,'Wrong electron count')
 cell=[[float(v)*BOHR for v in a.find('cell/'+k).text.split()] for k in ['a1','a2','a3']]
 pos=[[float(v)*BOHR for v in z.text.split()] for z in a.findall('atomic_positions/atom')]
 ff=[float(v)*HA/BOHR for v in o.findtext('forces').split()];ss=[float(v)*GPA for v in o.findtext('stress').split()]
 en=float(o.findtext('total_energy/etot'))*HA
 require(len(ff)==3*n and len(ss)==9 and len(pos)==n and all(math.isfinite(v) for v in ff+ss+[en]),'Invalid endpoint arrays')
 fm=max(math.sqrt(sum(v*v for v in ff[i:i+3])) for i in range(0,len(ff),3));fc=max(map(abs,ff));sm=max(map(abs,ss))
 require(fm<=.001,'Force norm >0.001 eV/A');require(fc<=.0005,'Force component >0.0005 eV/A');require(sm<=.05,'Full stress component >0.05 GPa')
 fg=old['final_geometry'];require(max(abs(v-w) for r,s in zip(cell,fg['cell_A']) for v,w in zip(r,s))<2e-7,'Final XML/stdout cell mismatch')
 require(max(abs(v-w) for r,s in zip(pos,fg['positions_A']) for v,w in zip(r,s))<2e-7,'Final XML/stdout positions mismatch')
 require(abs(en-old['energy_eV'])<2e-6,'Final XML/stdout energy mismatch')
 require(max(abs(v-w) for i,r in enumerate(old['forces_eV_A']) for j,w in enumerate(r) for v in [ff[3*i+j]])<3e-7,'Final XML/stdout total forces mismatch')
 require(max(abs(ss[3*i+j]-old['stress']['kbar_from_Ry_bohr3'][i][j]/10) for i in range(3) for j in range(3))<.00008,'Final XML/stdout stress mismatch')
 return {'accepted':True,'status':'accepted_bulk_k4_relaxation_validation','energy_eV':en,'n_atoms':n,'fmax_eV_A':fm,'max_force_component_eV_A':fc,'stress_max_abs_GPa':sm,'stress_QE_GPa':[ss[i:i+3] for i in range(0,9,3)],'final_scf_accuracy_Ry':scf,'cell_A':cell,'positions_A':pos,'forces_eV_A':[ff[i:i+3] for i in range(0,len(ff),3)],'final_scf_verified':True,'full_stress_target_pass':True,'final_geometry':fg,'trajectory':old['trajectory'],'limitations':'k4 relaxed validation endpoint; k-point, smearing and finite-size convergence not certified'}
def inspect(a,c,m,returncode=0):
 result={'case':c['id'],'accepted':False,'status':'not_accepted'}
 try:
  require(sha(a/'pw.in')==c['input_sha256'],'Runtime input mismatch');require(sha(a/'pseudo/Al_PAW_PBE.UPF')==m['pseudo_sha256']==PP,'Runtime PP mismatch')
  p=BASE.parse_input((a/'pw.in').read_text());xml=a/'scratch'/(c['prefix']+'.save')/'data-file-schema.xml'
  result.update(endpoint((a/'pw.out').read_text(errors='replace'),(a/'pw.stderr').read_text(errors='replace') if (a/'pw.stderr').exists() else '',p,xml,returncode))
  result['hashes']={k:sha(v) for k,v in {'pw.in':a/'pw.in','pw.out':a/'pw.out','xml':xml,'PP':a/'pseudo/Al_PAW_PBE.UPF'}.items()}
 except Exception as exc:result['error']=repr(exc)
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['verify','inspect','run']);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--index',type=int);p.add_argument('--attempt',type=Path);p.add_argument('--root',type=Path);args=p.parse_args();m=verify(args.campaign)
 if args.mode=='verify':print(json.dumps({'verified':5,'scope':m['validation_scope']}));return 0
 c=m['cases'][args.index]
 if args.mode=='inspect':
  r=inspect(args.attempt,c,m);print(json.dumps(r,indent=2));return 0 if r['accepted'] else 1
 require(sys.platform=='linux' and os.environ.get('SLURM_JOB_ID'),'Need Linux Slurm allocation');require(sha(m['binary'])==m['binary_sha256'],'QE binary changed')
 limits=memory_limits();allocation_guard(c,os.environ,min(limits.values()))
 root=args.root.resolve();require(not any((x/'.git').exists() for x in [root,*root.parents]),'Run output inside Git')
 a=root/c['id']/'attempt_001';a.mkdir(parents=True,exist_ok=False);shutil.copyfile(args.campaign/c['input'],a/'pw.in');(a/'pseudo').mkdir();shutil.copyfile(args.campaign/m['pseudo_file'],a/'pseudo/Al_PAW_PBE.UPF');shutil.copyfile(args.campaign/'qe_manifest.json',a/'qe_manifest.json')
 cmd=launch_command(m['binary']);dump(a/'provenance.json',{'case':c,'pseudo_sha256':m['pseudo_sha256'],'binary_sha256':m['binary_sha256'],'runner_sha256':sha(__file__),'bulk_parser_sha256':sha(BASE.__file__),'started_utc':datetime.now(timezone.utc).isoformat(),'slurm_job_id':os.environ['SLURM_JOB_ID'],'slurm_array_task_id':os.environ.get('SLURM_ARRAY_TASK_ID'),'cgroup_limits_bytes':limits,'command':cmd,'scope':m['validation_scope']})
 with (a/'pw.out').open('w') as out,(a/'pw.stderr').open('w') as err:proc=subprocess.run(cmd,cwd=a,stdout=out,stderr=err)
 r=inspect(a,c,m,proc.returncode);r.update(returncode=proc.returncode,finished_utc=datetime.now(timezone.utc).isoformat());dump(a/'result.json',r)
 if r['accepted']:
  dump(a/'final_geometry.json',r['final_geometry']);dump(a/'trajectory.json',r.pop('trajectory'))
  lattice=' '.join(format(v,'.14g') for row in r['cell_A'] for v in row)
  (a/'final.extxyz').write_text(str(r['n_atoms'])+'\n'+f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"\n'+'\n'.join('Al '+' '.join(format(v,'.14g') for v in row) for row in r['positions_A'])+'\n')
 print(json.dumps({'case':c['id'],'accepted':r['accepted'],'error':r.get('error')}));return 0 if r['accepted'] else 1
if __name__=='__main__':sys.exit(main())
