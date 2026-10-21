"""Deploy the reviewed immutable revision and run only non-DFT preflight."""
import base64, hashlib, io, json, subprocess, tarfile
from pathlib import Path

W = Path(__file__).resolve().parent
m = json.loads((W / 'REVISION_MANIFEST.json').read_text())
names = list(m['files']) + ['REVISION_MANIFEST.json']
files = {name: (W / name).read_bytes() for name in names}
manifest = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
assert all(manifest[k] == v for k, v in m['files'].items())
assert {'continue.sbatch', 'recover_axial.py', 'api_smoke.py'} <= files.keys()
assert b'\r' not in files['continue.sbatch'] and files['continue.sbatch'].startswith(b'#!/bin/bash\n'), 'SBATCH requires LF'
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode='w:gz') as t:
    for name, data in files.items():
        info = tarfile.TarInfo(name)
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
remote = r'''set -eu
python3 - <<'PY'
import base64,hashlib,io,json,os,subprocess,tarfile
from pathlib import Path
root=Path('/work/dawson666/src/campaign_snapshots/al_nano_R13_axial_continue_20261018_v2')
root.mkdir(parents=True,exist_ok=True)
manifest=__MANIFEST__
with tarfile.open(fileobj=io.BytesIO(base64.b64decode('__PAYLOAD__')),mode='r:gz') as t:
    for member in t:
        path=(root/member.name).resolve()
        if not path.is_relative_to(root) or not member.isfile():raise ValueError('Unsafe member')
        data=t.extractfile(member).read()
        if path.exists() and path.read_bytes()!=data:raise ValueError('Refuse snapshot overwrite: '+str(path))
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():path.write_bytes(data)
for rel,h in manifest.items():
    if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=h:raise ValueError(rel)
assert b'\r' not in (root/'continue.sbatch').read_bytes(), 'SBATCH requires LF'
runtime_sources=json.loads((root/'runtime_sources/manifest.json').read_text())
for item in runtime_sources:
    if hashlib.sha256(Path(item['source']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Installed runtime source changed: '+item['module'])
py='/home/dawson666/miniconda3/envs/dftpy-env/bin/python'
parent='/work/dawson666/src/campaign_snapshots/al_nano_five_radii_of_20261015_20261015T124450Z'
run='/work/dawson666/dftpy_project/relax/AL_NANO_FIVE_RADII_CORRECTED_20261015'
electronic='/work/dawson666/src/campaign_snapshots/al_nano_R13_electronic_20261017_v1'
axial='/work/dawson666/src/campaign_snapshots/al_nano_R13_axial_linesearch_20261018_v1'
common=['--original-snapshot',parent,'--electronic-snapshot',electronic,'--axial-snapshot',axial,'--runroot',run,'--parent-attempt',run+'/R13_0V/attempt_004_axial_linesearch']
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
def cmd(args):
    p=subprocess.run(args,cwd=root,env=env,capture_output=True,text=True,timeout=120)
    return dict(command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
tests=cmd([py,'-B','test_recovery.py'])
api_smoke=cmd([py,'-B','api_smoke.py'])
verify=cmd([py,'-B','recover_axial.py','verify']+common)
shell_checks={'continue.sbatch':cmd(['bash','-n',str(root/'continue.sbatch')])}
r={'snapshot':str(root),'payload_hashes':manifest,'runtime_sources_verified':runtime_sources,'tests':tests,'api_smoke':api_smoke,'verify':verify,'shell_checks':shell_checks,'no_DFT_executed':True}
(root/'preflight_receipt.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))
if tests['returncode'] or api_smoke['returncode'] or verify['returncode'] or any(x['returncode'] for x in shell_checks.values()):raise SystemExit(1)
PY
'''.replace('__MANIFEST__', repr(manifest)).replace('__PAYLOAD__', base64.b64encode(buf.getvalue()).decode())
(W / 'deploy_review.sh').write_text(remote, encoding='utf8', newline='\n')
p = subprocess.run(['wsl', 'python3', '/mnt/c/Users/m1331/Documents/Codex/2026-09-24/g/work/remote_exec.py', '/mnt/c/Users/m1331/Documents/Codex/2026-09-24/g/work/r13_axial_continue_20261018_v2/deploy_review.sh'], capture_output=True)
(W / 'remote_preflight.json').write_bytes(p.stdout)
(W / 'remote_preflight.stderr').write_bytes(p.stderr)
print('returncode', p.returncode)
print(p.stdout.decode(errors='replace'))
print(p.stderr.decode(errors='replace'))
raise SystemExit(p.returncode)
