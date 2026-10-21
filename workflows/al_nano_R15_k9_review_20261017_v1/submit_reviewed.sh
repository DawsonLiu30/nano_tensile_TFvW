set -eu
python3 - <<'PY'
import datetime,fcntl,json,re,subprocess
from pathlib import Path
root=Path('/work/dawson666/src/campaign_snapshots/al_nano_R15_k9_review_20261017_v1')
parent='/work/dawson666/src/campaign_snapshots/al_nano_five_radii_qe_20261015_20261015T123901Z'
k7='/work/dawson666/src/campaign_snapshots/al_nano_R15_k7_review_20261017_v1'
run=Path('/work/dawson666/qe_cases/qe_runs/AL_NANO_FIVE_RADII_QE_20261015')
receipt=root/'submission_receipt.json'
def dump(obj):
 tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(receipt)
def cmd(a):
 p=subprocess.run(a,capture_output=True,text=True,timeout=120)
 return {'command':a,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
with (root/'.submission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if receipt.exists():
  print(receipt.read_text());raise SystemExit('Existing receipt: no resubmission')
 pre=json.loads((root/'preflight_receipt.json').read_text())
 if pre['tests']['returncode'] or pre['verify']['returncode']:raise ValueError('Preflight failed')
 for v in [0,1]:
  if (run/f'nano_R15_{v}V_k9_SCF_20261017/attempt_001').exists():raise ValueError('Attempt exists')
 args=['sbatch','--parsable','--export=ALL,REVIEW_SNAPSHOT='+str(root)+',PARENT_SNAPSHOT='+parent+',K7_REVIEW_SNAPSHOT='+k7+',QE_RUN_ROOT='+str(run),
       '--output='+str(run/'slurm-R15-k9-%A_%a.out'),'--error='+str(run/'slurm-R15-k9-%A_%a.err'),str(root/'submit_k9.sbatch')]
 r={'scope':'Two R15 k9 fixed-geometry SCF diagnostics; new-size relaxations remain pending sampling validation',
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'snapshot':str(root),'runroot':str(run),
    'parent_snapshot':parent,'k7_review_snapshot':k7,'indices':[0,1],'preflight_passed':True,
    'test_only':cmd(args[:1]+['--test-only']+args[1:])}
 if r['test_only']['returncode']:
  r['status']='test_only_rejected';dump(r);print(json.dumps(r,indent=2));raise SystemExit(1)
 r['status']='submission_in_progress';dump(r)
 r['submission']=cmd(args)
 if r['submission']['returncode']:
  r['status']='submission_rejected';dump(r);print(json.dumps(r,indent=2));raise SystemExit(1)
 token=r['submission']['stdout'].strip().split(';')[0]
 if not re.fullmatch(r'\d+',token):
  r['status']='submission_output_needs_manual_check';dump(r);print(json.dumps(r,indent=2));raise SystemExit(2)
 r['job_id']=token;r['status']='submitted';dump(r)
 r['job_details']=cmd(['scontrol','show','job',token]);r['queue_snapshot']=cmd(['squeue','-h','-j',token,'-o','%i|%T|%M|%l|%R'])
 dump(r);print(json.dumps(r,indent=2))
PY
