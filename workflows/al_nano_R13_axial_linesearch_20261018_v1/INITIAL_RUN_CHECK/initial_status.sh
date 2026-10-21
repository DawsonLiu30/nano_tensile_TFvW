set -eu
python3 - <<'PY'
import datetime,json,subprocess
from pathlib import Path
run=Path('/work/dawson666/dftpy_project/relax/AL_NANO_FIVE_RADII_CORRECTED_20261015/R13_0V')
a=run/'attempt_004_axial_linesearch'
def command(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=30)
 return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':'2254539','queue':command(['squeue','-h','-j','2254539','-o','%i|%T|%M|%l|%R']),'attempt_exists':a.exists()}
for key,p in [('result',a/'result.json'),('checkpoint',a/'checkpoint.json')]:
 if p.exists():r[key]=json.loads(p.read_text())
for name in ['axial-linesearch-2254539.err']:
 p=run/name
 if p.exists():r[name]=p.read_text(errors='replace')[-4000:]
p=a/'dftpy_full_stdout.log'
if p.exists():
 with p.open('rb') as f:
  f.seek(0,2);size=f.tell();f.seek(max(0,size-3500));r['stdout_tail']=f.read().decode(errors='replace')
 r['stdout_bytes']=size
print(json.dumps(r,indent=2))
PY