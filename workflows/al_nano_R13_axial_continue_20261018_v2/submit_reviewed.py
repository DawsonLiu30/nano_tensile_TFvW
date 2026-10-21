"""Submit once through the existing SSH master; preserve all transport evidence."""
import json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
w = Path(__file__).resolve().parent
p = subprocess.run(['wsl','python3','/mnt/c/Users/m1331/Documents/Codex/2026-09-24/g/work/remote_exec.py','/mnt/c/Users/m1331/Documents/Codex/2026-09-24/g/work/r13_axial_continue_20261018_v2/submit_reviewed.sh'],capture_output=True)
out=p.stdout.decode(errors='replace');err=p.stderr.decode(errors='replace')
record={'utc':datetime.now(timezone.utc).isoformat(),'returncode':p.returncode,'stdout':out,'stderr':err}
(w/'last_submission_transport.json').write_text(json.dumps(record,indent=2),encoding='utf8')
print(out);print(err,file=sys.stderr)
if p.returncode:raise SystemExit(p.returncode)
r=json.loads(out)
if r.get('status')!='continuation_submitted':raise SystemExit('Submission incomplete; inspect remote receipt before retrying')
(w/'submission_receipt.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf8')
