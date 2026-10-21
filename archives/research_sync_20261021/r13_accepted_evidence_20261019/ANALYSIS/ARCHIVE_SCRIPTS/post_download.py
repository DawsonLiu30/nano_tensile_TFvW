import hashlib,json,subprocess,tarfile
from pathlib import Path
W=Path(__file__).resolve().parent
D=Path('C:/OFDFT/NAS_SHARE/LUDER_RESULTS_20261019/R13_AXIAL_CONTINUATION_RESULTS')
remote=r'''set -eu
python3 - <<'PY'
import datetime,hashlib,io,json,sys,tarfile
from pathlib import Path
of=Path('/work/dawson666/dftpy_project/relax/AL_NANO_FIVE_RADII_CORRECTED_20261015/R13_0V')
qe=Path('/work/dawson666/qe_cases/qe_runs/AL_NANO_FIVE_RADII_QE_20261015')
roots={'R13_0V':of/'attempt_005_axial_linesearch_continue'}
manifest=[];references=[];started=datetime.datetime.now(datetime.timezone.utc).isoformat()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
    def add(name,data):
        item=tarfile.TarInfo(name);item.size=len(data);archive.addfile(item,io.BytesIO(data))
    for label,root in roots.items():
        for p in sorted(root.rglob('*')):
            if not p.is_file():continue
            rel=p.relative_to(root)
            if 'scratch' in rel.parts and p.name!='data-file-schema.xml':continue
            if p.name=='.execution.lock' or '__pycache__' in rel.parts:continue
            omit=(p.suffix.lower() in ('.upf','.recpot') or (p.suffix=='.npz' and (p.name not in ['center_density.npz','checkpoint_bundle.npz'] or 'inputs' in rel.parts)))
            before=p.stat();data=p.read_bytes();after=p.stat()
            row={'source':str(p),'archive':label+'/'+rel.as_posix(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                 'mtime_before':before.st_mtime,'mtime_after':after.st_mtime,'stable_during_read':before.st_mtime_ns==after.st_mtime_ns and before.st_size==after.st_size,
                 'live_job_snapshot':'RUNNING_SNAPSHOT' in label}
            if omit:row['downloaded']=False;references.append(row)
            else:manifest.append(row);add(row['archive'],data)
    for root,patterns in [(of,['axial-continue-2255218.*'])]:
        for pattern in patterns:
            for p in root.glob(pattern):
                data=p.read_bytes();row={'source':str(p),'archive':'ACCOUNTING/'+p.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()};manifest.append(row);add(row['archive'],data)
    for name,obj in [('REMOTE_FILE_MANIFEST.json',manifest),('REMOTE_REFERENCES.json',references),
                     ('DOWNLOAD_SCOPE.json',{'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       'scope':'Final saved files of completed job2255218. Runner reports converged step46; independent endpoint and structural audit pending.'})]:
        add(name,json.dumps(obj,indent=2).encode())
PY
'''
(W/'post_download.sh').write_text(remote,encoding='utf8',newline='\n')
temp=W/'transfer.tar.gz'
with temp.open('wb') as f:
    p=subprocess.run(['wsl','python3','/mnt/c/Users/m1331/Documents/Codex/2026-09-24/g/work/remote_exec.py','/mnt/c/Users/m1331/Documents/Codex/2026-09-24/g/work/r13_continue_results_20261019/post_download.sh'],stdout=f,stderr=subprocess.PIPE)
(W/'post_download.stderr').write_bytes(p.stderr)
if p.returncode:raise SystemExit(p.stderr.decode(errors='replace'))
with tarfile.open(temp,'r:gz') as archive:
    for member in archive:
        target=(D/member.name).resolve()
        if not member.isfile() or not target.is_relative_to(D.resolve()):raise ValueError('Unsafe member')
        data=archive.extractfile(member).read()
        if target.exists() and target.read_bytes()!=data:raise ValueError('Refusing overwrite '+str(target))
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
manifest=json.loads((D/'REMOTE_FILE_MANIFEST.json').read_text())
for row in manifest:
    p=D/row['archive'];assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
(D/'ACCOUNTING/probe.json').write_bytes((W/'probe.json').read_bytes())
temp.unlink()
print(json.dumps({'files':len(manifest),'bytes':sum(x['bytes'] for x in manifest),'all_remote_hashes_match':True,'changed_while_reading':[x['archive'] for x in manifest if x.get('stable_during_read') is False],'destination':str(D)},indent=2))
