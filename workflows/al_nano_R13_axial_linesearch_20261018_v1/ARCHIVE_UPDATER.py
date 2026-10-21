"""Archive the submitted bounded R13 trial and update additive local research memory.

Preparation is inert. Execution requires an explicit receipt hash after parent approval.
No remote command or calculation is executed by this script.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, re, shutil

WORK=Path(__file__).resolve().parent
SOURCE=WORK/'r13_linesearch_recovery_20261018'
REPO=Path('C:/OFDFT/nano_tensile_TFvW')
PACKAGE=Path('C:/OFDFT/NAS_SHARE/LUDER_RESULTS_20261018')
DESTINATIONS=[REPO/'workflows/al_nano_R13_axial_linesearch_20261018_v1',PACKAGE/'RECOVERY_R13_AXIAL_LINESEARCH']
MARKER='R13_AXIAL_LINESEARCH_TRIAL_20261018'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def encoded(obj):return (json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
def dump(p,obj):Path(p).write_bytes(encoded(obj))
def require(ok,why):
    if not ok:raise ValueError(why)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--approved-receipt-sha256')
    parser.add_argument('--local-qa',type=Path)
    parser.add_argument('--independent-review',type=Path)
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps({'status':'prepared_not_executed','source':str(SOURCE),'destinations':[str(p) for p in DESTINATIONS],
                          'required':'Parent approval after receipt and QA; pass --execute --approved-receipt-sha256 HASH --local-qa PATH'},indent=2));return
    require(args.approved_receipt_sha256 and args.local_qa,'Explicit receipt hash and local QA path are required')
    receipt_path=SOURCE/'submission_receipt.json'
    require(receipt_path.exists() and sha(receipt_path)==args.approved_receipt_sha256,'Receipt absent/changed since approval')
    receipt=read(receipt_path);manifest=read(SOURCE/'REVISION_MANIFEST.json');remote=read(SOURCE/'remote_preflight.json')
    require(receipt.get('status')=='trial_submitted','No actual successful trial submission')
    job=str(receipt['job_id']);require(re.fullmatch(r'\d+',job),'Invalid job ID')
    require(receipt['submission']['returncode']==0 and receipt['submission']['stdout'].strip().split(';')[0]==job,'Submission response mismatch')
    revision_sha=sha(SOURCE/'REVISION_MANIFEST.json')
    require(receipt['revision_manifest_sha256']==revision_sha,'Submitted revision changed')
    require(receipt['force_target_eV_A']==receipt['filter_force_target_eV_A']==.001 and receipt['sigma_zz_target_GPa']==.05,'Acceptance thresholds changed')
    require(receipt['automatic_monitoring'] is False,'Unexpected monitoring')
    for name,expected in manifest['files'].items():
        require(sha(SOURCE/name)==expected,'Frozen source changed: '+name)
        require(remote['payload_hashes'][name]==expected,'Remote deployment mismatch: '+name)
    require(remote['payload_hashes']['REVISION_MANIFEST.json']==revision_sha,'Remote manifest mismatch')
    for key in ['tests','api_smoke','verify']:
        require(remote[key]['returncode']==0,'Failed remote check: '+key)
    require(all(v['returncode']==0 for v in remote['shell_checks'].values()),'Shell syntax failed')
    require(remote['no_DFT_executed'] is True,'Preflight scope changed')
    remote_tests=json.loads(remote['tests']['stdout']);verify=json.loads(remote['verify']['stdout'])
    require(remote_tests.get('passed')==remote_tests.get('tests') and remote_tests.get('tests',0)>0,'Remote program tests not all passed')
    require(verify.get('accepted') is True and verify['parent_bundle_sha256']==manifest['expected']['parent_bundle'],'Actual checkpoint preflight mismatch')
    local=read(args.local_qa)
    local_total=local.get('tests',local.get('total'));local_passed=local.get('passed')
    require(isinstance(local_total,int) and local_total>0 and local_passed==local_total,'Local program QA not all passed')
    details=receipt.get('job_details',{}).get('stdout','')
    require('Partition=ctest' in details and 'TimeLimit=02:00:00' in details and 'mem=24G' in details,'Unexpected actual Slurm resources')
    require('NumTasks=1' in details,'Unexpected task count')
    require('trial.sbatch' in receipt['submission']['command'][-1],'Wrong submitted entry')
    require(receipt.get('preflight_passed') is True,'Receipt lacks passed preflight')

    # Reference existing canonical checkpoint/input material; do not duplicate density data.
    parent=PACKAGE/'POST_STOP/R13_0V'
    refmap={
        'parent_result':parent/'result.json',
        'parent_bundle':parent/'checkpoint_bundle.npz',
        'electronic_gate':PACKAGE/'DFTPY/R13_ELECTRONIC_GATE/gate_pass.json',
        'electronic_revision':REPO/'workflows/al_nano_R13_electronic_20261017_v1/REVISION_MANIFEST.json',
        'original_manifest':Path('C:/OFDFT/NAS_SHARE/LUDER_NAS_UPDATE_20261015/EXECUTION_20261015/03_DFTpy_輸入與程式/MANIFEST.json'),
    }
    refs=[]
    for key,path in refmap.items():
        require(sha(path)==manifest['expected'][key],'Canonical reference changed: '+key)
        refs.append({'kind':key,'path':path.as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
    for name in ['checkpoint.json','config.json','provenance.json','initial.extxyz']:
        path=parent/name;refs.append({'kind':'parent_'+name,'path':path.as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})

    files=[]
    for path in sorted(SOURCE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix in ('.pyc','.tmp'):continue
        if path.name=='submission_receipt.json':require(path.stat().st_size>0,'Empty receipt')
        files.append((path,path.relative_to(SOURCE)))
    if args.local_qa.resolve() not in [p.resolve() for p,_ in files]:files.append((args.local_qa,Path('QA/local_program_checks.json')))
    if args.independent_review:
        for path in sorted(args.independent_review.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
                files.append((path,Path('INDEPENDENT_REVIEW')/path.relative_to(args.independent_review)))
    files.append((Path(__file__).resolve(),Path('ARCHIVE_UPDATER.py')))
    rows=[]
    for source,relative in files:
        digest=sha(source)
        for dest in DESTINATIONS:
            target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():require(sha(target)==digest,'Archive destination differs: '+str(target))
            else:shutil.copy2(source,target)
            require(sha(target)==digest,'Copy hash mismatch')
        require(sha(source)==digest,'Source changed during copy')
        rows.append({'path':relative.as_posix(),'source':source.resolve().as_posix(),'bytes':source.stat().st_size,'sha256':digest})
    for dest in DESTINATIONS:dump(dest/'SOURCE_REFERENCES.json',refs)

    statuspath=PACKAGE/'CURRENT_STATUS.json';registry_path=REPO/'docs/validated_workflows.json'
    state=read(statuspath);registry=read(registry_path)
    require(state['OF_accepted_endpoints']==9 and state['accepted_OF_pairs_unchanged']==['R11','R15','R17','R20'],'Canonical OF status changed; review before updating')
    require(state['R13_relaxation']['job_id']=='2253354' and state['R13_relaxation']['accepted'] is False,'Stop history changed')
    immutable_stop=json.dumps(state['R13_relaxation'],sort_keys=True)
    targets=[REPO/'AGENTS.md',REPO/'docs/VALIDATED_WORKFLOWS.md',registry_path,statuspath]
    for n,path in enumerate(targets):
        for dest in DESTINATIONS:
            backup=dest/'MEMORY_BEFORE'/f'{n:02d}_{path.name}';backup.parent.mkdir(exist_ok=True)
            if not backup.exists():shutil.copy2(path,backup)
    evidence={'revision':manifest.get('revision','al_nano_R13_axial_linesearch_20261018_v1'),'job_id':job,
              'status':'submitted_bounded_trial_physical_result_pending','submission_utc':receipt['utc'],
              'submission_snapshot':receipt.get('queue_snapshot'),'optimizer':'BFGSLineSearch',
              'cell_freedom':'atoms and z-only FrechetCellFilter; x/y fixed','electronic_optimizer':'TN',
              'electronic_econv_Ha_per_atom':1e-11,'force_target_eV_A':.001,'filter_force_target_eV_A':.001,'sigma_zz_target_GPa':.05,
              'maximum_ionic_steps':60,'maximum_logical_scf':120,'internal_runtime_limit_s':6600,
              'slurm_resources':{'partition':'ctest','walltime':'02:00:00','requested_memory_GiB':24,'tasks':1},
              'parent_completed_ionic_step':483,'parent_checkpoint_sha256':manifest['expected']['parent_bundle'],
              'revision_manifest_sha256':revision_sha,'submission_receipt_sha256':sha(receipt_path),
              'program_tests':{'local_passed':local_passed,'remote_passed':remote_tests['passed'],'physical_success':False},
              'physical_accepted':False,'automatic_continuation':False,'automatic_monitoring':False,
              'local_workflow':DESTINATIONS[0].as_posix(),'local_archive':DESTINATIONS[1].as_posix(),
              'scope':'Single bounded R13 trial; no new accepted endpoint or formation energy; original scientific core and physical model retained.'}
    state['R13_axial_trial']=evidence;state['updated_utc']=datetime.now(timezone.utc).isoformat()
    if job not in [str(x) for x in state['new_submissions']]:state['new_submissions'].append(job)
    require(json.dumps(state['R13_relaxation'],sort_keys=True)==immutable_stop,'Stop history must remain unchanged')
    registry['updated']='2026-10-18'
    registry['r13_axial_linesearch_reviewed_entrypoint']={'entry':(DESTINATIONS[0]/'recover_axial.py').as_posix(),
                                                       'manifest':(DESTINATIONS[0]/'REVISION_MANIFEST.json').as_posix(),'evidence':evidence}
    dump(statuspath,state);dump(registry_path,registry)
    note=f'''<!-- BEGIN {MARKER} -->
## 2026-10-18 R13 軸向 line-search 短試已提交（最新）

- 實際作業 **{job}**，ctest／24GiB／2h，僅R13_0V一組；回執時間 `{receipt['utc']}`。執行至多60個離子步、120次logical電子呼叫及6600秒，沒有自動長續算或定時監控。
- 使用停止後step483的一致checkpoint，bundle SHA `{manifest['expected']['parent_bundle']}`。恢復原子的移動與z軸晶胞自由度，以ASE BFGSLineSearch配合z-only FrechetCellFilter做有限短試；電子仍TN／1e−11Ha/atom，LDA/local、TF=.9/vW=.1與140×140×96網格保留。原子力與filter力均≤.001eV/Å、|σzz|≤.05GPa不變。
- 這是新的ionic候選：本機{local_passed}項及遠端{remote_tests['passed']}項程式檢查、現役ASE API smoke與真實checkpoint preflight通過，尚無新物理端點。原科學核心沿用；新revision manifest SHA `{revision_sha}`。不可宣稱已解決收斂或新增R13形成能。
- 永久流程 `{DESTINATIONS[0].as_posix()}`；來源、輸入引用、QA與正式回執在 `{DESTINATIONS[1].as_posix()}`。舊2253354主動停止紀錄與四組OF配對／九端點維持。QE k9資料與六個未放行尺寸狀態保持；本次沒有另送QE、上傳NAS或寄信。
<!-- END {MARKER} -->

'''
    for path in targets[:2]:
        old=path.read_text(encoding='utf-8-sig')
        if f'<!-- BEGIN {MARKER} -->' in old:
            require(f'**{job}**' in old.split(f'<!-- END {MARKER} -->')[0],'Different trial already recorded')
        else:path.write_text(note+old,encoding='utf-8',newline='\n')
    # Reference and backup records belong to the archive manifest as well.
    for p in sorted(DESTINATIONS[0].rglob('*')):
        rel=p.relative_to(DESTINATIONS[0])
        if not p.is_file() or rel.as_posix() in {r['path'] for r in rows} or p.name=='ARCHIVE_MANIFEST.json':continue
        require(sha(p)==sha(DESTINATIONS[1]/rel),'Auxiliary archive copy mismatch')
        rows.append({'path':rel.as_posix(),'source':'generated archive evidence','bytes':p.stat().st_size,'sha256':sha(p)})
    archive={'archive_kind':'submitted_program_tested_bounded_trial','archived_utc':datetime.now(timezone.utc).isoformat(),
             'submitted':True,'job_id':job,'physical_accepted':False,'revision_manifest_sha256':revision_sha,
             'submission_receipt_sha256':sha(receipt_path),'source_directory':SOURCE.as_posix(),
             'scope':'Program tests and deployment/checkpoint preflight passed. Bounded physical trial submitted; physical outcome not yet assessed.',
             'file_count':len(rows),'copy_hash_mismatches':0,'files':rows,'evidence':evidence}
    for dest in DESTINATIONS:dump(dest/'ARCHIVE_MANIFEST.json',archive)
    for row in rows:
        for dest in DESTINATIONS:require(sha(dest/row['path'])==row['sha256'],'Final archive mismatch')
    package_rows=[]
    for path in sorted(PACKAGE.rglob('*')):
        if path.is_file() and path!=PACKAGE/'PACKAGE_MANIFEST.json' and '__pycache__' not in path.parts:
            package_rows.append({'path':path.relative_to(PACKAGE).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
    dump(PACKAGE/'PACKAGE_MANIFEST.json',package_rows)
    require(all(sha(PACKAGE/r['path'])==r['sha256'] for r in package_rows),'Package verification failed')
    print(json.dumps({'status':'archived_memory_updated','job_id':job,'archive_files':len(rows),'package_files':len(package_rows),
                      'frozen_files_preserved':len(manifest['files'])+1,'copy_hash_mismatches':0,
                      'archive_manifest_sha256':sha(DESTINATIONS[0]/'ARCHIVE_MANIFEST.json'),'canonical_OF_pairs':4,'canonical_OF_endpoints':9},indent=2))

if __name__=='__main__':main()
