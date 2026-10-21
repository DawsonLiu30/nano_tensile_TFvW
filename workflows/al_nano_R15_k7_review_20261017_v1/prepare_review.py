"""Prepare review evidence from frozen local sources; does not submit or edit them."""
import argparse
import json
import re
from pathlib import Path
from run_reviewed_k7 import sha

HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-snapshot',type=Path,required=True)
    p.add_argument('--raw-root',type=Path,required=True);a=p.parse_args()
    source=json.loads((a.parent_snapshot/'remote_payload_manifest.json').read_text())
    parent_hashes={rel:r['sha256'] for rel,r in source.items()}
    for rel,h in parent_hashes.items():
        if sha(a.parent_snapshot/rel)!=h:raise ValueError('Frozen local source mismatch: '+rel)
    gpath=a.raw_root/'gate_status.json';gate=json.loads(gpath.read_text())
    m=json.loads((a.parent_snapshot/'qe_manifest.json').read_text())
    estimates=[]
    for i in [6,7]:
        c=m['cases'][i];out=a.raw_root/c['id']/'attempt_001/pw.out';text=out.read_text()
        estimates.append({'case':c['id'],'pw_out_sha256':sha(out),
            'printed_total_dynamic_RAM_lower_bound_GB':float(re.search(r'Estimated total dynamical RAM\s*>\s*([\d.]+)',text).group(1)),
            'allocation_GiB':192,'actual_completed_diagnostic':True})
    review={
      'review_id':'R15_k7_only_20261017_v1','status':'reviewed_preparation_not_submitted',
      'decision':'After the user requested failed work to be addressed, review the failed k5 transfer gate with only two denser k7 SCFs at the same accepted k3 geometries. Do not release or resubmit the six new-size relaxations.',
      'allowed_indices':[8,9],
      'parent_snapshot':'/work/dawson666/src/campaign_snapshots/al_nano_five_radii_qe_20261015_20261015T123901Z',
      'parent_runroot':'/work/dawson666/qe_cases/qe_runs/AL_NANO_FIVE_RADII_QE_20261015',
      'parent_campaign_sha256':sha(a.parent_snapshot/'qe_manifest.json'),
      'parent_file_sha256':parent_hashes,
      'entrypoint_sha256':{name:sha(HERE/name) for name in ['run_reviewed_k7.py','submit_k7.sbatch','test_reviewed_k7.py','prepare_review.py']},
      'k5_evidence':{'gate_status_sha256':sha(gpath),'gate_status':gate['status'],
          'delta_E1_minus_E0_eV':gate['delta_E1_minus_E0_eV'],
          'gate_limit_eV':.005,'gate_passed':False,'released_new_size_cases':[],
          'reason':'k5 diagnostic SCFs completed; energy difference, forces and axial stresses did not pass the transfer gate.'},
      'scientific_scope':'Fixed-geometry sampling sensitivity only. k7 SCF acceptance is electronic/data acceptance; it is not geometry convergence or proof of k convergence. Compare k3/k5/k7 paired energies, force vectors and axial stress. Re-relax setting still requires review.',
      'input_changes_vs_k5':['K_POINTS: 1 1 5 -> 1 1 7, offset 0 0 0 unchanged','prefix/case/output identity uses the already prepared k7 name'],
      'unchanged':['96/95 atoms','k3 final cell and coordinates','PAW/PBE PP','800 eV/ecutrho eightfold','MP .02 Ry','SCF 1e-10 Ry','71 h internal stop','29 MPI ranks','from_scratch startingpot/startingwfc'],
      'resources':{'nodes':1,'ranks':29,'memory_GiB':256,'Slurm_hours':72,'QE_max_seconds':255600,'array':'8-9%2',
          'reason':'Retain the resource request already frozen for conditional k7 cases; no new memory assumption added. k5 successful 192GiB and printed lower bounds give context, not a maximum bound or a guarantee.',
          'k5_printed_evidence':estimates},
      'old_guard':'The original runner explicitly blocks conditional_k7 execution. It stays byte-for-byte unchanged. This separately reviewed, hash-pinned entry point permits only the two pre-existing k7 cases.',
      'protect_old_work':'Original snapshot, k3/k5 results and failed gate unchanged. Unique new k7 case/attempt_001 plus case lock refuse duplicates. No gate release is written.',
    }
    (HERE/'review_manifest.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'review_manifest_sha256':sha(HERE/'review_manifest.json'),'parent_files':len(parent_hashes),'cases':[m['cases'][i]['id'] for i in [8,9]]},indent=2))
if __name__=='__main__':main()
