"""Generate only k9 input overlays from the audited frozen k7 input bytes."""
from pathlib import Path
import argparse,copy,json,shutil
from run_reviewed_k9 import sha
HERE=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-snapshot',type=Path,required=True)
 p.add_argument('--k7-entry',type=Path,required=True);p.add_argument('--archive-root',type=Path,required=True);a=p.parse_args()
 parent=a.parent_snapshot;archive=a.archive_root;m=json.loads((parent/'qe_manifest.json').read_text());oldreview=json.loads(a.k7_entry.with_name('review_manifest.json').read_text())
 cases=[];proof={};energies={5:[],7:[]};changed=[]
 for v in [0,1]:
  old=m['cases'][8+v];c=copy.deepcopy(old);cid=f'nano_R15_{v}V_k9_SCF_20261017';d=HERE/'inputs'/cid;d.mkdir(parents=True,exist_ok=True)
  text=(parent/old['input']).read_text().replace(old['prefix'],cid).replace('1 1 7 0 0 0','1 1 9 0 0 0')
  (d/'pw.in').write_text(text,encoding='utf-8',newline='\n');shutil.copyfile(parent/old['initial_structure'],d/'initial.extxyz')
  c.update(id=cid,prefix=cid,stage='reviewed_k9',kmesh=[1,1,9],input=f'inputs/{cid}/pw.in',
           input_sha256=sha(d/'pw.in'),initial_structure=f'inputs/{cid}/initial.extxyz')
  cases.append(c);changed.append({'case':cid,'parent_k7_input_sha256':sha(parent/old['input']),
    'new_input_sha256':c['input_sha256'],'changes':['prefix identity','automatic axial k mesh 7 to9'],
    'same_geometry':True,'same_namelists_except_prefix':True,'unchanged_initial_extxyz_sha256':c['initial_structure_sha256']})
  raw=archive/'FOLLOWUP_RESULTS'/f'QE_R15_{v}V_K7';r=json.loads((raw/'result.json').read_text())
  names=['pw.in','pw.out','pw.stderr','result.json','provenance.json','k7_review_manifest.json','run_reviewed_k7.py',f"scratch/{old['prefix']}.save/data-file-schema.xml"]
  proof[str(v)]={'case':old['id'],'sha256':{name:sha(raw/name) for name in names},
    'energy_eV':r['energy_eV'],'Fmax_eV_A':r['fmax_eV_A'],'sigma_zz_GPa':r['stress_QE_GPa'][2][2],
    'SCF_accuracy_Ry':r['final_scf_accuracy_Ry'],'status':r['status'],'geometry_targets_met':False}
  energies[7].append(r['energy_eV']);c5=m['cases'][6+v]
  energies[5].append(json.loads((archive/'RAW/QE_NANO'/c5['id']/'attempt_001/result.json').read_text())['energy_eV'])
 delta=(energies[7][1]-energies[7][0])-(energies[5][1]-energies[5][0])
 review={'review_id':'R15_k9_only_20261017_v1','status':'prepared_not_submitted','allowed_indices':[0,1],
  'decision':'User requests investigation and resubmission of unpassed work. k7 SCFs finished, but k5-to-k7 sampling sensitivity remains; perform only two k9 SCFs before selecting a re-relaxation mesh.',
  'parent_runroot':oldreview['parent_runroot'],'parent_snapshot':oldreview['parent_snapshot'],
  'parent_k7_snapshot':'/work/dawson666/src/campaign_snapshots/al_nano_R15_k7_review_20261017_v1',
  'parent_k7_entry_sha256':sha(a.k7_entry),'parent_k7_review_sha256':sha(a.k7_entry.with_name('review_manifest.json')),
  'parent_campaign_sha256':sha(parent/'qe_manifest.json'),'cases':cases,'k7_stageproof':proof,
  'k5_to_k7_pair_change_eV':delta,'operational_pair_sensitivity_target_eV':.005,
  'input_overlay_changes':changed,'parent_sources':'Original 10/15 full52-file snapshot plus 10/17 registered k7 entry: all hashes verified via the original k7 verifier; neither is changed.',
  'physics':'PAW/PBE original PP,800eV,ecutrho8x,MP.02Ry,SCF1e-10Ry,nosym/noinv,identical historicalk3 geometry,from_scratch; only axial mesh7→9 and prefix change.',
  'resources':{'ranks':29,'nodes':1,'memory_GiB':256,'Slurm_hours':72,'QE_internal_seconds':255600,'array':'0-1%2','basis':'Retain the successfully used k7 allocation; not a measured k9 peak-memory guarantee.'},
  'acceptance_scope':'Original inspector checks actual mesh9, PP/ecut/SCF/identity/fixed geometry. SCF acceptance remains distinct from geometry force and axial stress targets. No new-size case release, automatic higher k, re-relaxation or threshold change.',
  'entrypoint_sha256':{name:sha(HERE/name) for name in ['run_reviewed_k9.py','submit_k9.sbatch','prepare_overlay.py','test_k9_overlay.py']}}
 (HERE/'overlay_manifest.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8',newline='\n')
 print(json.dumps({'overlay_manifest_sha256':sha(HERE/'overlay_manifest.json'),'cases':[c['id'] for c in cases],'k5_to_k7_pair_change_meV':delta*1000},indent=2))
if __name__=='__main__':main()
