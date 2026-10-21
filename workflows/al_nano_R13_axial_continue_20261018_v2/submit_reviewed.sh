set -eu
python3 - <<'PY'
import datetime,fcntl,hashlib,json,re,subprocess
from pathlib import Path
root=Path('/work/dawson666/src/campaign_snapshots/al_nano_R13_axial_continue_20261018_v2')
run=Path('/work/dawson666/dftpy_project/relax/AL_NANO_FIVE_RADII_CORRECTED_20261015')
receipt=root/'submission_receipt.json'
def dump(r):
    p=receipt.with_suffix('.tmp');p.write_text(json.dumps(r,indent=2));p.replace(receipt)
def cmd(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=120)
    return {'command':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
with (root/'.submission.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if receipt.exists():
        print(receipt.read_text());raise SystemExit('Existing receipt: no duplicate submission')
    assert b'\r' not in (root/'continue.sbatch').read_bytes(), 'SBATCH requires LF'
    pre=json.loads((root/'preflight_receipt.json').read_text())
    if any(pre[k]['returncode'] for k in ['tests','api_smoke','verify']) or any(x['returncode'] for x in pre['shell_checks'].values()):raise ValueError('Preflight failed')
    for rel,h in pre['payload_hashes'].items():
        if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=h:raise ValueError('Changed payload: '+rel)
    if (run/'R13_0V'/'attempt_005_axial_linesearch_continue').exists():raise ValueError('Output already exists')
    for p in (run/'R13_0V').glob('attempt*/result.json'):
        if json.loads(p.read_text()).get('accepted') is True:raise ValueError('Case already accepted')
    active=cmd(['squeue','-h','-u','dawson666','-o','%i|%j|%T'])
    if active['returncode']:raise ValueError('Cannot inspect queue')
    if any('R13' in line for line in active['stdout'].splitlines()):raise ValueError('An R13 job is still active')
    r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'One bounded R13_0V continuation after completed trial step12; 6h maximum, fresh Hessian, line-search trial increment .02; no automatic follow-up',
       'snapshot':str(root),'runroot':str(run),'revision_manifest_sha256':pre['payload_hashes']['REVISION_MANIFEST.json'],'preflight_passed':True,
       'force_target_eV_A':.001,'filter_force_target_eV_A':.001,'sigma_zz_target_GPa':.05,'automatic_monitoring':False,
       'queue_before_submission':active}
    r['test_only']=cmd(['sbatch','--test-only','--parsable',str(root/'continue.sbatch')])
    if r['test_only']['returncode']:
        r['status']='test_only_rejected';dump(r);print(json.dumps(r,indent=2));raise SystemExit(1)
    r['status']='submission_in_progress';dump(r)
    r['submission']=cmd(['sbatch','--parsable',str(root/'continue.sbatch')]);dump(r)
    if r['submission']['returncode']:
        r['status']='submission_rejected';dump(r);print(json.dumps(r,indent=2));raise SystemExit(1)
    token=r['submission']['stdout'].strip().split(';')[0]
    if not re.fullmatch(r'\d+',token):
        r['status']='unknown_outcome_check_before_retry';dump(r);print(json.dumps(r,indent=2));raise SystemExit(2)
    r['job_id']=token;r['status']='continuation_submitted';dump(r)
    r['job_details']=cmd(['scontrol','show','job',token])
    r['queue_snapshot']=cmd(['squeue','-h','-j',token,'-o','%i|%T|%M|%l|%R'])
    dump(r);print(json.dumps(r,indent=2))
PY
