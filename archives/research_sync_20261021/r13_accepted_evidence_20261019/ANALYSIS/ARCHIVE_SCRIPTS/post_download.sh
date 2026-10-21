set -eu
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
