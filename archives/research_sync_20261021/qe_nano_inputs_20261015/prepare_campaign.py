"""Prepare new nano sizes and a representative axial-k gate. No DFT/submission."""
from pathlib import Path
import hashlib,json,math,re,shutil,sys,gzip
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]
REPO=Path('C:/OFDFT/nano_tensile_TFvW')
DATA=Path('C:/OFDFT/NAS_SHARE/Al_QE_DFTpy_DATA')
sys.path.insert(0,str(ROOT/'scripts'))
import run_bulk_vacancy_qe_vcrelax_20260925 as PARSER
from run_nano_gated_20261015 import parse_input
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
sources=[]
def record(p,purpose):
 sources.append({'path':str(p).replace('\\','/'),'sha256':sha(p),'bytes':p.stat().st_size,'purpose':purpose})
def fval(v):
 if isinstance(v,bool):return '.true.' if v else '.false.'
 if isinstance(v,str):return "'"+v+"'"
 return str(v)
old=REPO/'campaigns/al_nano_qe_memory_recovery_20261001'
record(old/'qe_manifest.json','verified predecessor protocol, binary and PP')
om=json.loads((old/'qe_manifest.json').read_text())
template=parse_input((old/'qe/nano_R15_0V/pw.in').read_text())['blocks']
record(old/'qe/nano_R15_0V/pw.in','scientific namelists predecessor')
def convert(v):
 if v in ['.true.','.false.']:return v=='.true.'
 if v.startswith("'"):return PARSER.unquote(v)
 return PARSER.number(v)
common={k:{a:convert(v) for a,v in b.items()} for k,b in template.items()}
common['control'].update(max_seconds=255600,nstep=1000)
for b,k in [('control','prefix'),('system','nat'),('system','nbnd')]:common[b].pop(k)
pp=ROOT/'pseudo/Al_PAW_PBE.UPF';pp.parent.mkdir(exist_ok=True);shutil.copyfile(old/'pseudo/Al_PAW_PBE.UPF',pp)
record(old/'pseudo/Al_PAW_PBE.UPF','unchanged PBE PAW')
for name in ['CONVERGENCE_BASELINES.md','convergence_baselines_20261011.json']:
 src=REPO/'docs'/name;record(src,'required calibration');shutil.copyfile(src,ROOT/'reference'/name)
cases=[];baseline={};qa=[]
def make(caseid,nat,cell,pos,radius,vac,k,stage,nbnd,source,geom=None):
 blocks=json.loads(json.dumps(common));calc='vc-relax' if stage=='size_relaxation' else 'scf'
 blocks['control'].update(calculation=calc,prefix=caseid,nstep=1000 if calc=='vc-relax' else 1)
 blocks['system'].update(nat=nat,nbnd=nbnd)
 if calc=='scf':blocks.pop('ions');blocks.pop('cell')
 cdir=ROOT/'inputs'/caseid;cdir.mkdir(parents=True,exist_ok=True)
 text=''
 for name,block in blocks.items():text+='&'+name.upper()+'\n'+''.join(' '+name+' = '+fval(v)+',\n' for name,v in block.items())+'/\n'
 text+='ATOMIC_SPECIES\nAl 26.9815385 Al_PAW_PBE.UPF\nCELL_PARAMETERS angstrom\n'+'\n'.join(' '.join(format(x,'.15g') for x in r) for r in cell)+'\nATOMIC_POSITIONS angstrom\n'+'\n'.join('Al '+' '.join(format(x,'.15g') for x in r) for r in pos)+f'\nK_POINTS automatic\n1 1 {k} 0 0 0\n'
 (cdir/'pw.in').write_text(text)
 lattice=' '.join(format(x,'.15g') for r in cell for x in r)
 (cdir/'initial.extxyz').write_text(str(nat)+'\n'+f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"\n'+'\n'.join('Al '+' '.join(format(x,'.15g') for x in r) for r in pos)+'\n')
 mem=128 if radius in ['R11','R13'] else (192 if k<7 else 256)
 case={'id':caseid,'kind':'nanocolumn_axial_vcrelax' if calc=='vc-relax' else 'nanocolumn_fixed_geometry_axial_k_diagnostic','calculation':calc,'stage':stage,'radius_label':radius,'vacancy_count':vac,'nat':nat,'nbnd':nbnd,'prefix':caseid,'input':str((cdir/'pw.in').relative_to(ROOT)).replace('\\','/'),'input_sha256':sha(cdir/'pw.in'),'initial_structure':str((cdir/'initial.extxyz').relative_to(ROOT)).replace('\\','/'),'initial_structure_sha256':sha(cdir/'initial.extxyz'),'cell_A':cell,'positions_A':pos,'pbc':[True]*3,'kmesh':[1,1,k],'koffset':[0,0,0],'resources':{'nodes':1,'ntasks':29,'memory_GiB':mem,'slurm_hours':72},'source':source}
 if geom:case['geometry_metadata']=geom
 cases.append(case)
 x=parse_input(text);diff=max(abs(a-b) for r,s in zip(x['positions_A'],pos) for a,b in zip(r,s))
 assert x['nat']==nat and diff<1e-10
 assert cell[2][2]>=10 and min(cell[0][0],cell[1][1])>=10
 qa.append({'case':caseid,'atom_count':nat,'parsed_position_max_difference_A':diff,'axial_period_A':cell[2][2],'PBC':True,'pass':True})
for radius in ['R11','R13','R17']:
 for vac in [0,1]:
  d=WORK/'luder_response_20261015/geometry_audit/candidate_structures/QE'/f'{radius}_{vac}V'
  g=json.loads((d/'geometry.json').read_text());lines=(d/'structure.extxyz').read_text().splitlines();pos=[[float(v) for v in l.split()[1:4]] for l in lines[2:]]
  assert len(pos)==g['N_atoms'] and g['pbc']==[True]*3 and min(g['projected_atom_center_image_gaps_A'])>=10
  for name in ['geometry.json','structure.extxyz']:record(d/name,'frozen candidate geometry')
  make(f'nano_{radius}_{vac}V_k3_20261015',len(pos),g['cell_A'],pos,radius,vac,3,'size_relaxation',math.ceil(1.9*len(pos)),{'geometry_sha256':sha(d/'geometry.json'),'structure_sha256':sha(d/'structure.extxyz')},g)
for k in [5,7]:
 for vac in [0,1]:
  src=DATA/'QE/03_Nanocolumn_R15_relaxed'/f'nano_R15_{vac}V'
  r=json.loads((src/'result.json').read_text());assert r['accepted'] and r['fmax_eV_A']<=.001
  out=ROOT/'reference'/f'R15_{vac}V_k3_result.json';shutil.copyfile(src/'result.json',out)
  if k==5:
   for name in ['result.json','pw.in','pw.out',f'scratch/nano_R15_{vac}V.save/data-file-schema.xml']:record(src/name,'accepted R15 k3 baseline for same-geometry k5 test')
   baseline[str(vac)]={'case':f'nano_R15_{vac}V','energy_eV':r['energy_eV'],'cell_A':r['cell_A'],'positions_A':r['positions_A'],'nat':r['n_atoms'],'reference':str(out.relative_to(ROOT)).replace('\\','/'),'sha256':sha(out)}
  nbnd=183 if vac==0 else 181
  make(f'nano_R15_{vac}V_k{k}_SCF_20261015',r['n_atoms'],r['cell_A'],r['positions_A'],'R15',vac,k,'representative_k5' if k==5 else 'conditional_k7',nbnd,{'accepted_k3_result_sha256':sha(out),'input_geometry':'accepted full precision XML result; fixed existing final geometry'})
protocol={'campaign':'al_nano_five_radii_gated_20261015','status':'prepared_not_submitted','baseline_ids':['QE 800 eV cutoff retained','primitive 40^3/80^3 EOS supported in its own cell','nano k3 transfer checked here, not previously certified'],'common_namelists':common,'acceptance':{'force_norm_eV_A':.001,'force_component_eV_A':.0005,'axial_stress_GPa':.05,'SCF_accuracy_Ry':1e-10,'require_BFGS_and_final_cell_SCF_for_relaxation':True,'fixed_transverse_cell':True,'minimum_vacancy_image_translation_A':10},'k_gate':{'stage1_cases':[6,7],'stage2':'gate --run-root <root> --release <root>/gate_release.json','stage3_small_indices':[0,1,2,3],'stage3_medium_indices':[4,5],'maximum_change_E1_minus_E0_eV':.005,'delta_definition':'[(E1-E0)_k5 - (E1-E0)_k3] at each existing k3 equilibrium geometry; same bulk mu cancels','gate_requires_both_k5_final_SCF_and_force_and_axial_stress':True,'component_threshold_eV_A':.0005,'failure_action':'Fail gate job (nonzero); dependent six relaxations do not start. No automatic k7/re-relax/submission. Review the residual force and energy sensitivity.','conditional_k7_indices':[8,9],'k7_status':'prepared_only_not_scheduled','scope':'Representative R15 check at old final geometries; 5 meV is this work practical decision threshold, not a universal convergence law. Does not certify all radii, smearing, vacuum or axial-length convergence.'},'resources':{'R11_R13_GiB':128,'R17_GiB':192,'R15_k5_GiB':192,'R15_k7_conditional_GiB':256,'ranks':29,'walltime_hours':72,'QE_internal_seconds':255600,'gate_memory_GiB':1,'gate_walltime_minutes':20,'memory_note':'R15 observed QE estimate >72.60/72.10 GB and successful 128 GiB allocations; R20 >156.45/155.76 GB and measured ~177.96 GiB/256 GiB. R17 allocation is a size-interpolated conservative request, not a measured peak. Denser-k R15 given extra headroom; QE estimates are lower bounds.'},'checkpoint_policy':'Retain full scratch and .bfgs under unique attempt_001. Internal max_seconds stops 1h before allocation wall. No automatic restart; subsequent continuation must verify density/wfc/BFGS consistency and explicitly use startingpot/wfc=file.'}
dump(ROOT/'protocol.json',protocol)
dump(ROOT/'reference/k3_baseline.json',baseline)
manifest={'campaign':protocol['campaign'],'cases':cases,'pseudo_file':'pseudo/Al_PAW_PBE.UPF','pseudo_sha256':om['pseudo_sha256'],'binary':om['binary'],'binary_sha256':om['binary_sha256'],'protocol_sha256':sha(ROOT/'protocol.json'),'k3_baseline_file':'reference/k3_baseline.json','k3_baseline_sha256':sha(ROOT/'reference/k3_baseline.json'),'scope':'6 new size endpoints gated by 2 representative SCFs; 2 k7 files are conditional and not scheduled','calibration_sources':{'reference/CONVERGENCE_BASELINES.md':sha(ROOT/'reference/CONVERGENCE_BASELINES.md'),'reference/convergence_baselines_20261011.json':sha(ROOT/'reference/convergence_baselines_20261011.json')}}
dump(ROOT/'qe_manifest.json',manifest);dump(ROOT/'source_index.json',sources);dump(ROOT/'preparation_QA.json',{'passed':len(qa),'failed':0,'checks':qa})
fixture=ROOT/'tests/fixture';fixture.mkdir(exist_ok=True);fm=[]
for vac in [0,1]:
 src=DATA/'QE/03_Nanocolumn_R15_relaxed'/f'nano_R15_{vac}V'
 for name in ['pw.in','pw.out',f'scratch/nano_R15_{vac}V.save/data-file-schema.xml']:
  p=src/name;dest=fixture/(f'R15_{vac}V_'+p.name+'.gz');dest.write_bytes(gzip.compress(p.read_bytes(),mtime=0))
  fm.append({'file':dest.name,'uncompressed_sha256':sha(p),'compressed_sha256':sha(dest),'source':str(p).replace('\\','/'),'scope':'historical k3 test fixture only; never a new k5 calculation'})
dump(fixture/'manifest.json',fm)
print(json.dumps({'cases':len(cases),'ready_stage1':[c['id'] for c in cases[6:8]],'gated_relaxations':6,'k7_prepared_only':2}))
