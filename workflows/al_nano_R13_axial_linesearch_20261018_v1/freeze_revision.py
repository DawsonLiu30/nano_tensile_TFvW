from pathlib import Path
import hashlib,json,difflib
from recover_axial import EXPECTED
from staged_optimizer import POLICY
root=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if sha(root/'recovery_base.py')!=EXPECTED['base_runner']:raise ValueError('Base runner changed')
prior=root.parent/'r13_electronic_recovery_20261017/electronic_trace.py'
if prior.exists():(root/'electronic_trace.diff').write_text(''.join(difflib.unified_diff(prior.read_text().splitlines(True),(root/'electronic_trace.py').read_text().splitlines(True),fromfile='previous/electronic_trace.py',tofile='current/electronic_trace.py')))
names=['recover_axial.py','recovery_base.py','electronic_trace.py','electronic_trace.diff','staged_optimizer.py','test_recovery.py','freeze_revision.py','trial.sbatch','README.txt']
if (root/'api_smoke.py').exists():names.append('api_smoke.py')
names+=sorted(str(p.relative_to(root)).replace('\\','/') for p in (root/'runtime_sources').rglob('*') if p.is_file())
manifest={'revision':'R13_axial_BFGSLineSearch_20261018_v1','expected':EXPECTED,'optimizer_policy':POLICY,'files':{n:sha(root/n) for n in names},'scope':'Single bounded ctest candidate, not yet physically accepted'}
(root/'REVISION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'manifest_sha256':sha(root/'REVISION_MANIFEST.json'),'payload_files_including_manifest':len(names)+1}))
