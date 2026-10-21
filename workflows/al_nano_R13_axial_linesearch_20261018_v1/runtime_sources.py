import io,json,subprocess,tarfile,hashlib
from pathlib import Path
W=Path(__file__).resolve().parent;D=W/'runtime_sources';D.mkdir(exist_ok=True)
script='''set -eu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/home/dawson666/miniconda3/envs/dftpy-env/bin/python -B - <<'PY'
import importlib,inspect,hashlib,tarfile,sys,io,json
from pathlib import Path
mods=['ase.optimize.bfgslinesearch','ase.optimize.optimize','ase.filters','ase.utils.linesearch','dftpy.utils']
manifest=[]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for name in mods:
  m=importlib.import_module(name);p=Path(inspect.getsourcefile(m));data=p.read_bytes();arc=name+'.py'
  manifest.append({'module':name,'source':str(p),'archive':arc,'sha256':hashlib.sha256(data).hexdigest()})
  info=tarfile.TarInfo(arc);info.size=len(data);t.addfile(info,io.BytesIO(data))
 data=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('manifest.json');info.size=len(data);t.addfile(info,io.BytesIO(data))
PY
'''
(W/'runtime_sources.sh').write_text(script,encoding='utf8',newline='\n')
p=subprocess.run(['wsl','python3','/mnt/c/Users/m1331/Documents/Codex/2026-09-24/g/work/remote_exec.py','/mnt/c/Users/m1331/Documents/Codex/2026-09-24/g/work/r13_linesearch_recovery_20261018/runtime_sources.sh'],capture_output=True)
if p.returncode:raise SystemExit(p.stderr.decode(errors='replace'))
with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:gz') as t:
 for m in t:
  q=(D/m.name).resolve();assert q.is_relative_to(D.resolve()) and m.isfile();data=t.extractfile(m).read()
  if q.exists():assert q.read_bytes()==data
  else:q.write_bytes(data)
manifest=json.loads((D/'manifest.json').read_text())
for r in manifest:assert hashlib.sha256((D/r['archive']).read_bytes()).hexdigest()==r['sha256']
print(json.dumps(manifest,indent=2))
