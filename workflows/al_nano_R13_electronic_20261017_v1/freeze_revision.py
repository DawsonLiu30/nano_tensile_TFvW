"""Build deterministic small-source manifest; no DFT/remote changes."""
from pathlib import Path
import hashlib,json,difflib
from recover_electronic import EXPECTED,SETTINGS
from staged_optimizer import POLICY
root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prior=root.parent/'r13_fire_recovery_20261017/implementation'
if (prior/'recovery_base.py').exists():
    if sha(prior/'recovery_base.py')!=sha(root/'recovery_base.py'):raise ValueError('Scientific base runner must stay byte-identical to previous smallstep revision')
    (root/'staged_optimizer.diff').write_text(''.join(difflib.unified_diff((prior/'staged_optimizer.py').read_text().splitlines(True),(root/'staged_optimizer.py').read_text().splitlines(True),fromfile='prior/staged_optimizer.py',tofile='current/staged_optimizer.py')))
names=['recover_electronic.py','electronic_trace.py','recovery_base.py','staged_optimizer.py','test_recovery.py','freeze_revision.py','staged_optimizer.diff','README.txt','diagnose.sbatch','continue.sbatch','submission_metadata.json']
manifest={'revision':'R13_bounded_TN_force_gate_20261017_v1','expected':EXPECTED,'settings':SETTINGS,'ionic_policy':POLICY,
          'files':{n:sha(root/n) for n in names},'scope':'Bounded checkpoint electronic/force diagnosis, optional guarded smallstep continuation; no submission in these scripts'}
(root/'REVISION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'manifest_sha256':sha(root/'REVISION_MANIFEST.json'),'files':len(names)},indent=2))
