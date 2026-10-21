import base64,hashlib,io,json,subprocess,tarfile
from pathlib import Path
W=Path(__file__).resolve().parent
names=['run_reviewed_k9.py','submit_k9.sbatch','overlay_manifest.json','test_k9_overlay.py','prepare_overlay.py']
files={name:(W/name).read_bytes() for name in names}
for p in (W/'inputs').rglob('*'):
 if p.is_file():files[p.relative_to(W).as_posix()]=p.read_bytes()
assert hashlib.sha256(files['overlay_manifest.json']).hexdigest()=='fc36ca93d6fc4a25047f2c838b3f3330c013ba92045e75a35d1eddd538a5053a'
manifest={k:hashlib.sha256(v).hexdigest() for k,v in files.items()}
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as t:
 for name,data in files.items():
  info=tarfile.TarInfo(name);info.size=len(data);t.addfile(info,io.BytesIO(data))
remote=r'''set -eu
python3 - <<'PY'
import base64,hashlib,io,json,os,subprocess,tarfile
from pathlib import Path
root=Path('/work/dawson666/src/campaign_snapshots/al_nano_R15_k9_review_20261017_v1')
root.mkdir(parents=True,exist_ok=True)
manifest=__MANIFEST__
with tarfile.open(fileobj=io.BytesIO(base64.b64decode('__PAYLOAD__')),mode='r:gz') as t:
 for m in t:
  p=(root/m.name).resolve()
  if not p.is_relative_to(root) or not m.isfile():raise ValueError('Unsafe member')
  data=t.extractfile(m).read()
  if p.exists() and p.read_bytes()!=data:raise ValueError('Refusing snapshot overwrite: '+str(p))
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
for rel,h in manifest.items():
 if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=h:raise ValueError(rel)
py='/home/dawson666/miniconda3/bin/python3'
parent='/work/dawson666/src/campaign_snapshots/al_nano_five_radii_qe_20261015_20261015T123901Z'
run='/work/dawson666/qe_cases/qe_runs/AL_NANO_FIVE_RADII_QE_20261015'
k7='/work/dawson666/src/campaign_snapshots/al_nano_R15_k7_review_20261017_v1/run_reviewed_k7.py'
common=['--parent-snapshot',parent,'--root',run,'--k7-entry',k7]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
def cmd(args):
 p=subprocess.run(args,cwd=root,env=env,capture_output=True,text=True,timeout=120)
 return dict(command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
tests=cmd([py,'-B','test_k9_overlay.py']+common)
verify=cmd([py,'-B','run_reviewed_k9.py','verify']+common)
r={'snapshot':str(root),'payload_hashes':manifest,'tests':tests,'verify':verify,'no_DFT_executed':True}
(root/'preflight_receipt.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))
if tests['returncode'] or verify['returncode']:raise SystemExit(1)
PY
'''.replace('__MANIFEST__',repr(manifest)).replace('__PAYLOAD__',base64.b64encode(buf.getvalue()).decode())
(W/'deploy_review.sh').write_text(remote,encoding='utf8',newline='\n')
p=subprocess.run(['wsl','python3','/mnt/c/Users/m1331/Documents/Codex/2026-09-24/g/work/remote_exec.py','/mnt/c/Users/m1331/Documents/Codex/2026-09-24/g/work/qe_R15_k9_recovery_20261017/deploy_review.sh'],capture_output=True)
(W/'remote_preflight.json').write_bytes(p.stdout);(W/'remote_preflight.stderr').write_bytes(p.stderr)
print('returncode',p.returncode)
if p.stdout:print(p.stdout.decode(errors='replace'))
if p.stderr:print(p.stderr.decode(errors='replace'))
raise SystemExit(p.returncode)
