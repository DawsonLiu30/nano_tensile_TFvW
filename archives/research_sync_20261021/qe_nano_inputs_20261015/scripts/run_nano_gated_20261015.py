"""Three-stage nano execution, with a fail-closed representative k-point gate.

Reuses the accepted nano inspector and allocation guard; never calls sbatch.
"""
from pathlib import Path
import argparse,hashlib,json,math,os,re,shutil,subprocess,sys
from datetime import datetime,timezone
import xml.etree.ElementTree as ET
import run_luder_qe_20260930 as BASE
import run_bulk_vacancy_qe_vcrelax_20260925 as PARSER
from run_nano_qe_memory_recovery_20261001 import memory_limits,allocation_guard,launch_command
PP='450c947cbbf30c33a5a77685bbe9a270947816f8a93837c02c749d0ee4e3ab4a'
sha=BASE.sha;require=BASE.require;dump=BASE.dump
IDS=[f'nano_{r}_{v}V_k3_20261015' for r in ['R11','R13','R17'] for v in [0,1]]+[f'nano_R15_{v}V_k{k}_SCF_20261015' for k in [5,7] for v in [0,1]]
def parse_input(text):
 """Reuse established syntax tokens, with nano-specific semantics checked below."""
 clean=PARSER.strip_comments(text);blocks={};spans=[]
 for match in PARSER.NAMELIST.finditer(clean):
  name=match.group(1).lower();require(name not in blocks,'Duplicate namelist')
  b={}
  for item in PARSER.ASSIGNMENT.finditer(match.group(2)):
   key=item.group(1).lower();require(key not in b,'Duplicate namelist setting');b[key]=item.group(2).strip()
  require(not PARSER.ASSIGNMENT.sub('',match.group(2)).replace(',','').strip(),'Unsupported namelist syntax')
  blocks[name]=b;spans.append(match.span())
 nat=int(PARSER.number(blocks['system']['nat']))
 for start,end in reversed(spans):clean=clean[:start]+clean[end:]
 lines=[l.strip() for l in clean.splitlines() if l.strip()];cards={};i=0
 while i<len(lines):
  header=lines[i];name=header.split()[0].upper();require(name in ['ATOMIC_SPECIES','CELL_PARAMETERS','ATOMIC_POSITIONS','K_POINTS'] and name not in cards,'Unsupported/duplicate card')
  n={'ATOMIC_SPECIES':1,'CELL_PARAMETERS':3,'ATOMIC_POSITIONS':nat,'K_POINTS':1}[name]
  cards[name]={'units':header[len(name):].strip().strip('{}()').lower(),'rows':[l.split() for l in lines[i+1:i+n+1]]};require(len(cards[name]['rows'])==n,'Missing card rows');i+=n+1
 require(len(cards)==4,'Missing input cards')
 require(cards['ATOMIC_SPECIES']['rows']==[['Al','26.9815385','Al_PAW_PBE.UPF']],'Wrong atom species/PP card')
 require(cards['CELL_PARAMETERS']['units']=='angstrom','Cell units changed')
 cell=[[PARSER.number(v) for v in r] for r in cards['CELL_PARAMETERS']['rows']];require(all(len(r)==3 for r in cell) and PARSER.cell_volume(cell)>0,'Invalid cell')
 rows=cards['ATOMIC_POSITIONS']['rows'];require(all(len(r)==4 and r[0]=='Al' for r in rows),'Wrong/constraint position card')
 require(cards['ATOMIC_POSITIONS']['units'] in ['angstrom','crystal'],'Unsupported position units')
 pos=PARSER.convert_positions([[PARSER.number(v) for v in r[1:]] for r in rows],cards['ATOMIC_POSITIONS']['units'],cell)
 require(len({tuple(r) for r in pos})==nat,'Duplicate atomic positions')
 require(cards['K_POINTS']['units']=='automatic','Wrong k-point card')
 return {'blocks':blocks,'nat':nat,'cell_A':cell,'positions_A':pos,'cards':cards}
def value(v):
 if v in ['.true.','.false.']:return v=='.true.'
 if v.startswith("'"):return PARSER.unquote(v)
 return PARSER.number(v)
def verify(campaign):
 m=BASE.verify(campaign);require(m['campaign']=='al_nano_five_radii_gated_20261015','Unexpected campaign')
 require([c['id'] for c in m['cases']]==IDS,'Unexpected case inventory')
 require(m['pseudo_sha256']==PP,'Wrong PAW/PBE pseudo')
 require(sha(campaign/'protocol.json')==m['protocol_sha256'],'Protocol hash mismatch')
 require(sha(campaign/m['k3_baseline_file'])==m['k3_baseline_sha256'],'k3 reference index changed')
 baseline=json.loads((campaign/m['k3_baseline_file']).read_text())
 for b in baseline.values():
  p=campaign/b['reference'];require(sha(p)==b['sha256'],'Accepted k3 source changed')
  r=json.loads(p.read_text());require(r['accepted'] and r['energy_eV']==b['energy_eV'] and r['cell_A']==b['cell_A'] and r['positions_A']==b['positions_A'],'k3 reference fields changed')
 for p,h in m['calibration_sources'].items():require(sha(campaign/p)==h,'Calibration changed')
 for i,c in enumerate(m['cases']):
  require(c['pbc']==[True]*3,'PBC changed')
  require(sha(campaign/c['initial_structure'])==c['initial_structure_sha256'],'Initial structure changed')
  x=parse_input((campaign/c['input']).read_text());b={n:{k:value(v) for k,v in z.items()} for n,z in x['blocks'].items()}
  expected_stage='size_relaxation' if i<6 else ('representative_k5' if i<8 else 'conditional_k7')
  k=3 if i<6 else (5 if i<8 else 7)
  calc='vc-relax' if i<6 else 'scf'
  require(c['stage']==expected_stage and c['calculation']==calc and c['kmesh']==[1,1,k],'Stage, calculation or mesh changed')
  require(c['nat']==([52,51,76,75,128,127,96,95,96,95][i]),'Wrong atom count')
  require(x['nat']==c['nat'] and max(abs(v-w) for r,s in zip(x['cell_A'],c['cell_A']) for v,w in zip(r,s))<1e-11,'Input cell or nat changed')
  require(len(x['positions_A'])==c['nat'] and max(abs(v-w) for r,s in zip(x['positions_A'],c['positions_A']) for v,w in zip(r,s))<1e-10,'Input coordinates changed')
  require(min(c['cell_A'][j][j] for j in range(3))>=10,'Periodic vacancy-image translation below 10 A')
  require(max(abs(c['cell_A'][j][q]) for j in range(3) for q in range(3) if j!=q)<1e-10,'Unexpected tilted box')
  if i<6:
   require(min(c['geometry_metadata']['projected_atom_center_image_gaps_A'])>=10,'Initial transverse vacuum reduced')
   require(c['geometry_metadata']['prospective_vacancy_parent_CN']==12,'Vacancy parent site is not CN12')
   require(c['nbnd']==math.ceil(1.9*c['nat']),'Band count changed')
  else:
   z=baseline[str(c['vacancy_count'])]
   require(c['cell_A']==z['cell_A'] and c['positions_A']==z['positions_A'],'k test not at accepted k3 geometry')
   require(c['nbnd']==(183 if c['vacancy_count']==0 else 181),'R15 band count changed')
  common={'control':{'calculation':calc,'nstep':1000 if i<6 else 1,'etot_conv_thr':1e-6,'forc_conv_thr':1.94469038114889e-5,'max_seconds':255600,'tstress':True,'tprnfor':True,'prefix':c['prefix'],'pseudo_dir':'./pseudo','outdir':'./scratch','restart_mode':'from_scratch','disk_io':'low','verbosity':'high'},'system':{'ibrav':0,'nat':c['nat'],'ntyp':1,'nbnd':c['nbnd'],'ecutwfc':58.798915481,'ecutrho':470.3913238484,'occupations':'smearing','smearing':'mp','degauss':.02,'nspin':1,'tot_charge':0,'nosym':True,'noinv':True},'electrons':{'conv_thr':1e-10,'electron_maxstep':300,'scf_must_converge':True,'mixing_mode':'plain','mixing_beta':.2,'mixing_ndim':4,'diagonalization':'david','diago_david_ndim':2,'startingpot':'atomic','startingwfc':'atomic+random'}}
  if i<6:common.update(ions={'ion_dynamics':'bfgs','trust_radius_min':1e-6},cell={'cell_dynamics':'bfgs','cell_dofree':'z','press':0,'press_conv_thr':.5})
  require(b==common,'Unexpected namelist change: '+c['id'])
  require(x['cards']['K_POINTS']['rows']==[['1','1',str(k),'0','0','0']],'Mesh/offset changed')
  mem=128 if i<4 else (192 if i<8 else 256)
  require(c['resources']=={'nodes':1,'ntasks':29,'memory_GiB':mem,'slurm_hours':72},'Resource request changed')
 return m
def inspect(a,c,m,returncode=0):
 r={'case':c['id'],'accepted':False,'status':'not_accepted'}
 try:
  require(sha(a/'pseudo/Al_PAW_PBE.UPF')==m['pseudo_sha256']==PP,'Runtime PP changed')
  require(returncode==0,'MPI process returned nonzero')
  err=(a/'pw.stderr').read_text(errors='replace') if (a/'pw.stderr').exists() else ''
  require(not re.search(r'oom-kill|out of memory|killed by signal|MPI_ABORT|BAD TERMINATION',err,re.I),'MPI/memory failure')
  r=BASE.inspect(a,c);require(r['accepted'],'Original nano inspector: '+str(r.get('error')))
  xi=ET.parse(a/'scratch'/(c['prefix']+'.save')/'data-file-schema.xml').getroot().find('input')
  require(xi.findtext('control_variables/calculation')==c['calculation'],'XML calculation mismatch')
  require(xi.findtext('control_variables/prefix')==c['prefix'],'XML prefix mismatch')
  require(xi.findtext('dft/functional')=='PBE','XML XC mismatch')
  require(xi.findtext('atomic_species/species/pseudo_file')=='Al_PAW_PBE.UPF','XML PP mismatch')
  require(int(xi.findtext('bands/nbnd'))==c['nbnd'],'XML band count mismatch')
  require(xi.findtext('bands/smearing')=='mp' and abs(2*float(xi.find('bands/smearing').attrib['degauss'])-.02)<1e-12,'XML smearing mismatch')
  for path,target in [('basis/ecutwfc',58.798915481),('basis/ecutrho',470.3913238484),('electron_control/conv_thr',1e-10)]:require(abs(2*float(xi.findtext(path))-target)<=max(1e-20,target*1e-12),'XML numerical setting mismatch: '+path)
  kp=xi.find('k_points_IBZ/monkhorst_pack');require([int(kp.attrib[k]) for k in ['nk1','nk2','nk3','k1','k2','k3']]==c['kmesh']+[0,0,0],'XML k mesh mismatch')
  ai=xi.find('atomic_structure');require(int(ai.attrib['nat'])==c['nat'],'XML input nat mismatch')
  icell=[[float(v)*BASE.BOHR for v in ai.find('cell/'+k).text.split()] for k in ['a1','a2','a3']]
  ipos=[[float(v)*BASE.BOHR for v in z.text.split()] for z in ai.findall('atomic_positions/atom')]
  require(len(ipos)==c['nat'] and max(abs(v-w) for rr,ss in zip(icell,c['cell_A']) for v,w in zip(rr,ss))<2e-7 and max(abs(v-w) for rr,ss in zip(ipos,c['positions_A']) for v,w in zip(rr,ss))<2e-7,'XML initial geometry mismatch')
  require(r['final_scf_accuracy_Ry']<=1e-10,'Strict final SCF target failed')
  require(min(r['cell_A'][j][j] for j in range(3))>=10,'Final vacancy-image translation below 10 A')
  if c['calculation']=='vc-relax':
   require(r['fmax_eV_A']<=.001 and r['max_force_component_eV_A']<=.0005,'Final force gate failed')
   require(abs(r['stress_QE_GPa'][2][2])<=.05,'Final axial stress failed')
  r['scientific_scope']='representative fixed-geometry diagnostic' if c['calculation']=='scf' else 'relaxed size endpoint; representative R15 k check passed, all-size convergence not certified'
  r['output_sha256']={'pw.in':sha(a/'pw.in'),'pw.out':sha(a/'pw.out'),'xml':sha(a/'scratch'/(c['prefix']+'.save')/'data-file-schema.xml')}
 except Exception as exc:r.update(accepted=False,status='not_accepted',error=str(exc))
 return r
def evaluate_gate(results,baseline):
 checks=[]
 def ck(name,passed,actual,limit=None):checks.append({'criterion':name,'pass':bool(passed),'actual':actual,'limit':limit})
 for v in [0,1]:
  r=results[v];ck(f'{v}V final SCF diagnostic accepted',r.get('accepted') is True,r.get('status'))
  ck(f'{v}V final SCF accuracy',r.get('final_scf_accuracy_Ry',math.inf)<=1e-10,r.get('final_scf_accuracy_Ry'),1e-10)
  ck(f'{v}V force norm',r.get('fmax_eV_A',math.inf)<=.001,r.get('fmax_eV_A'),.001)
  ck(f'{v}V force component',r.get('max_force_component_eV_A',math.inf)<=.0005,r.get('max_force_component_eV_A'),.0005)
  zz=r.get('stress_QE_GPa',[[None]*3]*3)[2][2]
  ck(f'{v}V axial stress',zz is not None and abs(zz)<=.05,zz,.05)
 delta=None
 if all(r.get('accepted') for r in results):
  delta=(results[1]['energy_eV']-results[0]['energy_eV'])-(baseline['1']['energy_eV']-baseline['0']['energy_eV'])
 ck('Change in E1-E0 at fixed k3 geometries',delta is not None and abs(delta)<=.005,delta,.005)
 return {'passed':all(x['pass'] for x in checks),'checks':checks,'delta_E1_minus_E0_eV':delta,'delta_E1_minus_E0_meV':None if delta is None else 1000*delta,'definition':'[(E1-E0)_k5 - (E1-E0)_k3]; fixed k3 endpoint geometries; same chemical potential cancels','scope':'5 meV operational threshold for representative R15 only; does not certify all radii, smearing, vacuum or axial-length convergence'}
def gate(campaign,root,release):
 m=verify(campaign);baseline=json.loads((campaign/m['k3_baseline_file']).read_text());res=[];hashes={}
 # A stale release cannot survive a re-evaluation failure.
 require(not release.exists(),'Gate release already exists; use a new run root instead of overwriting')
 for c in m['cases'][6:8]:
  a=root/c['id']/'attempt_001';r=inspect(a,c,m)
  result_file=a/'result.json'
  if result_file.exists():
   saved=json.loads(result_file.read_text());require(saved.get('returncode')==0,'Stage1 did not exit cleanly')
  else:r.update(accepted=False,status='missing_runner_result')
  res.append(r)
  for p in [a/'pw.in',a/'pw.out',a/'result.json',a/'scratch'/(c['prefix']+'.save')/'data-file-schema.xml']:
   if p.exists():hashes[str(p)]=sha(p)
 g=evaluate_gate(res,baseline);g.update(campaign_sha256=sha(campaign/'qe_manifest.json'),created_utc=datetime.now(timezone.utc).isoformat(),stage1_hashes=hashes,baseline_index_sha256=m['k3_baseline_sha256'],released_case_ids=IDS[:6] if g['passed'] else [],status='passed_release_six_size_relaxations' if g['passed'] else 'failed_downstream_blocked')
 dump(root/'gate_status.json',g)
 if g['passed']:dump(release,g)
 else:(root/'GATE_FAILED.txt').write_text('No new-size relaxations may start.\n'+'\n'.join(str(c) for c in g['checks'] if not c['pass'])+'\nReview k5 sensitivity; no automatic k7 or denser re-relaxation is submitted.\n')
 return g
def release_guard(campaign,root,release,c):
 require(release is not None and release.exists(),'Missing successful stage2 release')
 g=json.loads(release.read_text());require(g.get('passed') is True and all(x['pass'] for x in g['checks']),'Gate did not pass')
 require(g['campaign_sha256']==sha(campaign/'qe_manifest.json'),'Gate belongs to another campaign')
 require(c['id'] in g['released_case_ids'] and g['released_case_ids']==IDS[:6],'Case not released')
 for p,h in g['stage1_hashes'].items():require(sha(p)==h,'Stage1 evidence changed after gate')
 require(g['baseline_index_sha256']==sha(campaign/'reference/k3_baseline.json'),'Gate baseline changed')
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['verify','inspect','run','gate']);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--index',type=int);p.add_argument('--root',type=Path);p.add_argument('--attempt',type=Path);p.add_argument('--release',type=Path);args=p.parse_args()
 if args.mode=='gate':
  try:r=gate(args.campaign,args.root,args.release)
  except Exception as exc:
   r={'passed':False,'status':'failed_downstream_blocked','error':str(exc)};dump(args.root/'gate_status.json',r)
  print(json.dumps(r));return 0 if r['passed'] else 1
 m=verify(args.campaign)
 if args.mode=='verify':print(json.dumps({'verified':len(m['cases']),'stage1_indices':[6,7],'gated_relaxation_indices':list(range(6)),'conditional_not_scheduled':[8,9]}));return 0
 c=m['cases'][args.index]
 if args.mode=='inspect':
  r=inspect(args.attempt,c,m);print(json.dumps(r,indent=2));return 0 if r['accepted'] else 1
 require(sys.platform=='linux' and os.environ.get('SLURM_JOB_ID'),'Run requires Linux Slurm allocation')
 require(sha(m['binary'])==m['binary_sha256'],'QE binary changed')
 require(c['stage']!='conditional_k7','k7 is prepared only; new reviewed campaign needed before execution')
 root=args.root.resolve();require(not any((x/'.git').exists() for x in [root,*root.parents]),'Run root inside Git')
 if c['stage']=='size_relaxation':release_guard(args.campaign,root,args.release,c)
 limits=memory_limits();allocation_guard(c,os.environ,min(limits.values()))
 a=root/c['id']/'attempt_001';a.mkdir(parents=True,exist_ok=False);shutil.copyfile(args.campaign/c['input'],a/'pw.in');(a/'pseudo').mkdir();shutil.copyfile(args.campaign/m['pseudo_file'],a/'pseudo/Al_PAW_PBE.UPF');shutil.copyfile(args.campaign/'qe_manifest.json',a/'qe_manifest.json')
 cmd=launch_command(m['binary']);dump(a/'provenance.json',{'case':c,'pseudo_sha256':m['pseudo_sha256'],'binary_sha256':m['binary_sha256'],'runner_sha256':sha(__file__),'nano_inspector_sha256':sha(BASE.__file__),'started_utc':datetime.now(timezone.utc).isoformat(),'slurm_job_id':os.environ['SLURM_JOB_ID'],'slurm_array_task_id':os.environ.get('SLURM_ARRAY_TASK_ID'),'cgroup_limits_bytes':limits,'command':cmd,'gate_release_sha256':sha(args.release) if c['stage']=='size_relaxation' else None})
 with (a/'pw.out').open('w') as out,(a/'pw.stderr').open('w') as err:proc=subprocess.run(cmd,cwd=a,stdout=out,stderr=err)
 r=inspect(a,c,m,proc.returncode);r.update(returncode=proc.returncode,finished_utc=datetime.now(timezone.utc).isoformat());dump(a/'result.json',r)
 if r['accepted']:
  lattice=' '.join(format(v,'.15g') for row in r['cell_A'] for v in row)
  (a/'final.extxyz').write_text(str(r['n_atoms'])+'\n'+f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"\n'+'\n'.join('Al '+' '.join(format(v,'.15g') for v in row) for row in r['positions_A'])+'\n')
 print(json.dumps({'case':c['id'],'accepted':r['accepted'],'status':r['status'],'error':r.get('error')}));return 0 if r['accepted'] else 1
if __name__=='__main__':sys.exit(main())
