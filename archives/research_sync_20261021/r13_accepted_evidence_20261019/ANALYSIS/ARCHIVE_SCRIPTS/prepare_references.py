"""Read-only source verification; write staging evidence, without DFT or remote I/O."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = Path('C:/OFDFT/NAS_SHARE')
FLOW = Path('C:/OFDFT/nano_tensile_TFvW/workflows/al_nano_R13_axial_continue_20261018_v2')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

manifest = FLOW / 'REVISION_MANIFEST.json'
expected = '0873dc84e7e796fea4f743ab37e1f2dcc1a34dd86bee2cf9fef70cc7e3970bbf'
assert sha(manifest) == expected
checks = []
for rel, digest in json.loads(manifest.read_text(encoding='utf-8'))['files'].items():
    p = FLOW / rel
    checks.append({'path': str(p), 'sha256': sha(p), 'expected_sha256': digest,
                   'matches': sha(p) == digest})
assert all(c['matches'] for c in checks)
files = [
 ('accepted_R13_1V_result', 'LUDER_RESULTS_20261017/RAW/DFTPY_NANO/R13_1V/attempt_001/result.json'),
 ('accepted_R13_1V_structure', 'LUDER_RESULTS_20261017/RAW/DFTPY_NANO/R13_1V/attempt_001/final.extxyz'),
 ('accepted_R13_1V_provenance', 'LUDER_RESULTS_20261017/RAW/DFTPY_NANO/R13_1V/attempt_001/provenance.json'),
 ('same_method_bulk_chemical_potential', 'Al_QE_DFTpy_DATA/DFTpy/01_Bulk_relaxed/0V/result.json'),
 ('same_method_bulk_provenance', 'Al_QE_DFTpy_DATA/DFTpy/01_Bulk_relaxed/0V/provenance.json'),
 ('same_local_pseudopotential', 'Al_QE_DFTpy_DATA/DFTpy/01_Bulk_relaxed/0V/inputs/al.lda.recpot'),
 ('parent_coherent_step12_bundle', 'LUDER_RESULTS_20261018/R13_AXIAL_TRIAL_RESULTS/R13_0V/checkpoint_bundle.npz'),
 ('submitted_workflow_manifest', 'LUDER_RESULTS_20261018/RECOVERY_R13_AXIAL_CONTINUATION/SUBMITTED_V2/REVISION_MANIFEST.json'),
 ('actual_submission_receipt', 'LUDER_RESULTS_20261018/RECOVERY_R13_AXIAL_CONTINUATION/SUBMITTED_V2/submission_receipt.json'),
 ('previous_accepted_four_pair_analysis', 'LUDER_RESULTS_20261017/ANALYSIS/ACCEPTED_RESULTS/formation_analysis.json'),
 ('latest_QE_k9_status_unchanged', 'LUDER_RESULTS_20261018/CURRENT_STATUS.json'),
]
refs = []
for purpose, rel in files:
    p = ROOT / rel
    assert p.is_file(), p
    refs.append({'purpose': purpose, 'NAS_SHARE_relative_path': rel,
                 'absolute_local_path': str(p), 'bytes': p.stat().st_size,
                 'sha256': sha(p), 'copied_to_new_package': False})
doc = {'scope': 'References to previously preserved source files; no old package edited and no large density duplicate created.',
       'workflow_manifest_sha256': expected, 'immutable_workflow_file_checks': checks,
       'references': refs}
(OUT/'SOURCE_REFERENCES.prepared.json').write_text(json.dumps(doc, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
print(json.dumps({'workflow_checks_passed': len(checks), 'references_verified':len(refs)}))
