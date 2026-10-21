set -eu
python3 - <<'PY'
import datetime,json,subprocess
from pathlib import Path
case=Path('/work/dawson666/dftpy_project/relax/AL_NANO_FIVE_RADII_CORRECTED_20261015/R13_0V')
a=case/'attempt_005_axial_linesearch_continue'
def cmd(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=40)
 return {'command':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'queue':cmd(['squeue','-u','dawson666','-o','%.18i %.12P %.22j %.10T %.12M %.12l %R']), 'accounting':cmd(['sacct','-j','2255218','--format=JobID,JobName%24,State,ExitCode,Elapsed,Timelimit,MaxRSS,ReqMem,TotalCPU,Start,End','--parsable2'])}
for name in ('result.json','checkpoint.json'):
 if (a/name).exists():r[name]=json.loads((a/name).read_text())
for name in ['relax.log','optimizer_events.jsonl','ionic_line_search.jsonl','electronic_evaluations.jsonl']:
 p=a/name
 if p.exists():r[name+'_tail']=p.read_text(errors='replace').splitlines()[-6:]
for suffix in ('out','err'):
 p=case/('axial-continue-2255218.'+suffix)
 if p.exists():r['slurm_'+suffix]=p.read_text(errors='replace')[-3500:]
print(json.dumps(r,indent=2))
PY