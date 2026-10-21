from pathlib import Path
import hashlib,json,difflib
from recover_axial import EXPECTED
from staged_optimizer import POLICY
root=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if sha(root/'recovery_base.py')!=EXPECTED['base_runner']:raise ValueError('Base runner changed')
prior=root.parent/'r13_linesearch_recovery_20261018/electronic_trace.py'
if prior.exists():(root/'electronic_trace.diff').write_text(''.join(difflib.unified_diff(prior.read_text().splitlines(True),(root/'electronic_trace.py').read_text().splitlines(True),fromfile='previous/electronic_trace.py',tofile='current/electronic_trace.py')))
script=(root/'continue.sbatch').read_bytes()
if b'\r' in script or not script.startswith(b'#!/bin/bash\n'):raise ValueError('SBATCH requires LF line endings and valid shebang')
if b'al_nano_R13_axial_continue_20261018_v2' not in script:raise ValueError('Wrong v2 snapshot path')
names=['packaging_fix.json','recover_axial.py','recovery_base.py','electronic_trace.py','electronic_trace.diff','staged_optimizer.py','test_recovery.py','freeze_revision.py','continue.sbatch','README.txt']
if (root/'api_smoke.py').exists():names.append('api_smoke.py')
names+=sorted(str(p.relative_to(root)).replace('\\','/') for p in (root/'runtime_sources').rglob('*') if p.is_file())
manifest={'revision':'R13_axial_BFGSLineSearch_continue_20261018_v2','expected':EXPECTED,'optimizer_policy':POLICY,'files':{n:sha(root/n) for n in names},'scope':'Single bounded ct56 continuation; no automatic followup; not yet physically accepted'}
(root/'REVISION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'manifest_sha256':sha(root/'REVISION_MANIFEST.json'),'payload_files_including_manifest':len(names)+1}))
