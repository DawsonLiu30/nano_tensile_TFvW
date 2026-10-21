"""Deterministic minimal copy/patch and manifest, without changing the parent."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
from staged_optimizer import POLICY
from recover_nano import EXPECTED

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-snapshot',type=Path,required=True);a=p.parse_args()
    out=Path(__file__).resolve().parent;source=a.parent_snapshot/'scripts/base_runner.py'
    if sha(source)!=EXPECTED['parent_runner']:raise ValueError('Wrong frozen base runner')
    text=source.read_text(encoding='utf-8')
    replacements=[
      ('    from ase.optimize import LBFGS\n','    from staged_optimizer import make_optimizer\n'),
      ('                dyn = LBFGS(cell_filter, logfile=str(attempt / "relax.log"),\n                            maxstep=manifest["ionic"]["maxstep_A"])',
       '                dyn = make_optimizer(cell_filter, logfile=str(attempt / "relax.log"),\n                                     ionic=manifest["ionic"])'),
      ('"relaxation_mode": "atoms_and_axial_zz_FrechetCellFilter_zero_pressure",',
       '"relaxation_mode": "fixed_cell_atoms_after_axial_relaxation; original axial filter/stress acceptance retained",'),
    ]
    original=text
    for old,new in replacements:
        if text.count(old)!=1:raise ValueError('Minimal patch anchor changed')
        text=text.replace(old,new)
    (out/'recovery_base.py').write_text(text,encoding='utf-8',newline='\n')
    (out/'base_runner_minimal.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='parent/base_runner.py',tofile='recovery_base.py')),encoding='utf-8')
    names=['recover_nano.py','recovery_base.py','staged_optimizer.py','test_recovery.py','build_revision.py','base_runner_minimal.diff']
    manifest={'revision':'r13_smallstep_checkpoint_recovery_20261017_v1','parent':EXPECTED,
              'optimizer_policy':POLICY,'files':{n:sha(out/n) for n in names},
              'scope':'Only R13_0V actual execution; fixed-cell smallstep ionic LBFGS finish; atomic/filter/stress/electronic acceptance unchanged'}
    (out/'REVISION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'revision_manifest_sha256':sha(out/'REVISION_MANIFEST.json'),'files':manifest['files']},indent=2))

if __name__=='__main__':main()
