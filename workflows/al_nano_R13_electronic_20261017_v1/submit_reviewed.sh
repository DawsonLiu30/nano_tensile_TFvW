set -eu
python3 - <<'PY'
import datetime,fcntl,hashlib,json,re,subprocess
from pathlib import Path
root=Path('/work/dawson666/src/campaign_snapshots/al_nano_R13_electronic_20261017_v1')
run=Path('/work/dawson666/dftpy_project/relax/AL_NANO_FIVE_RADII_CORRECTED_20261015')
receipt=root/'submission_receipt.json'
def dump(r):
    p=receipt.with_suffix('.tmp');p.write_text(json.dumps(r,indent=2));p.replace(receipt)
def cmd(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=120)
    return {'command':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
def submit(r,key,args):
    r['status']=key+'_submission_in_progress';dump(r)
    r[key]=cmd(args);dump(r)
    if r[key]['returncode']:
        r['status']=key+'_submission_rejected';dump(r);print(json.dumps(r,indent=2));raise SystemExit(1)
    token=r[key]['stdout'].strip().split(';')[0]
    if not re.fullmatch(r'\d+',token):
        r['status']=key+'_unknown_outcome_check_before_retry';dump(r);print(json.dumps(r,indent=2));raise SystemExit(2)
    r[key+'_job_id']=token;r['status']=key+'_submitted';dump(r)
    return token
with (root/'.submission.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if receipt.exists():
        print(receipt.read_text());raise SystemExit('Existing receipt: no duplicate submission')
    pre=json.loads((root/'preflight_receipt.json').read_text())
    if pre['tests']['returncode'] or pre['verify']['returncode'] or any(x['returncode'] for x in pre['shell_checks'].values()):raise ValueError('Preflight failed')
    for rel,h in pre['payload_hashes'].items():
        if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=h:raise ValueError('Changed payload: '+rel)
    for name in ['diagnostics_003_electronic','attempt_003_TNsmallstep']:
        if (run/'R13_0V'/name).exists():raise ValueError('Output exists: '+name)
    for p in (run/'R13_0V').glob('attempt*/result.json'):
        if json.loads(p.read_text()).get('accepted') is True:raise ValueError('Case already accepted')
    r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'R13_0V bounded electronic/force diagnostic, followed only on scientific success by same-checkpoint smallstep ionic completion',
       'snapshot':str(root),'runroot':str(run),'revision_manifest_sha256':pre['payload_hashes']['REVISION_MANIFEST.json'],'preflight_passed':True,
       'force_target_eV_A':.001,'filter_force_target_eV_A':.001,'sigma_zz_target_GPa':.05,'automatic_monitoring':False}
    for key,script in [('diagnostic_test_only','diagnose.sbatch'),('continuation_test_only','continue.sbatch')]:
        r[key]=cmd(['sbatch','--test-only','--parsable',str(root/script)])
        if r[key]['returncode']:
            r['status']='test_only_rejected';dump(r);print(json.dumps(r,indent=2));raise SystemExit(1)
    gate=submit(r,'diagnostic',['sbatch','--parsable',str(root/'diagnose.sbatch')])
    r['required_dependency']='afterok:'+gate;dump(r)
    continuation=submit(r,'continuation',['sbatch','--parsable','--dependency=afterok:'+gate,'--kill-on-invalid-dep=yes',str(root/'continue.sbatch')])
    r['status']='both_submitted';dump(r)
    r['job_details']={key:cmd(['scontrol','show','job',jid]) for key,jid in [('diagnostic',gate),('continuation',continuation)]}
    r['queue_snapshot']=cmd(['squeue','-h','-j',gate+','+continuation,'-o','%i|%T|%M|%l|%R'])
    dump(r);print(json.dumps(r,indent=2))
PY
