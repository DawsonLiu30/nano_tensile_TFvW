"""Create local reviewed payload inventory; no network or scheduling."""
from pathlib import Path
import csv,hashlib,json
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
m=json.loads((R/'qe_manifest.json').read_text());sources=json.loads((R/'source_index.json').read_text())
checks=[]
for s in sources:
 p=Path(s['path']);checks.append({'path':s['path'],'unchanged':sha(p)==s['sha256']})
dump(R/'source_unchanged_QA.json',{'checked':len(checks),'all_unchanged':all(x['unchanged'] for x in checks),'checks':checks})
rows=[]
for i,c in enumerate(m['cases']):
 rows.append({'index':i,'case':c['id'],'stage':c['stage'],'n_atoms':c['nat'],'kz':c['kmesh'][2],'Lz_A':c['cell_A'][2][2],'Lx_A':c['cell_A'][0][0],'nbnd':c['nbnd'],'memory_GiB':c['resources']['memory_GiB'],'ranks':29,'walltime_hours':72,'QE_seconds':255600,'scheduled_status':'conditional_not_scheduled' if i>=8 else ('ready_for_stage1' if i>=6 else 'requires_afterok_gate'),'input_sha256':c['input_sha256']})
with (R/'case_inventory.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
memory={'historical_printed_QE_total_dynamic_RAM_lower_bound_GB':{'R15_0V':72.60,'R15_1V':72.10,'R20_0V':156.45,'R20_1V':155.76},'historical_successful_allocation_GiB':{'R15':128,'R20':256},'historical_R20_cgroup_peak_GiB':177.96,'historical_sources':['C:/OFDFT/nano_tensile_TFvW/docs/QE_NANO_MEMORY_RECOVERY_20261001_ZH.md','C:/OFDFT/nano_tensile_TFvW/docs/QE_NANO_PROGRESS_20261012.md'],'proposed_GiB':{'R11_0V_1V':128,'R13_0V_1V':128,'R17_0V_1V':192,'R15_k5_0V_1V':192},'reasoning':'R11/R13 are smaller than the successful R15 benchmark; retain its proven 128 GiB request rather than return to failed 64 GiB. R17 lies between 96 and176 sites; use192 GiB. R15k5 uses192GiB to allow extra k-point storage. None is a measured peak or guarantee for new inputs. Enforced cgroup and Slurm memory checks precede launching all29 ranks.','not_a_measurement':'R17 lower-bound interpolation of the two historical pristine estimates is106.14 GB; this is planning only, not QE output for R17.'}
dump(R/'memory_planning.json',memory)
payload=[]
for p in sorted(R.rglob('*')):
 if not p.is_file() or '__pycache__' in p.parts or p.name=='deployment_manifest.json':continue
 payload.append({'path':p.relative_to(R).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
dump(R/'deployment_manifest.json',{'campaign':m['campaign'],'status':'prepared_not_submitted','payload_count':len(payload),'total_bytes':sum(x['bytes'] for x in payload),'files':payload})
print(json.dumps({'payload_count':len(payload),'bytes':sum(x['bytes'] for x in payload),'sources_unchanged':all(x['unchanged'] for x in checks)}))
