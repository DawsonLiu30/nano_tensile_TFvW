"""First-SCF bulk k4 checkpoint adapter; frozen runner files remain unchanged.

The three 2026-10-15 parents stopped at SCF iteration 15 before any ionic
step. Their distributed restart_k/restart_scf files replace the BFGS/update
requirements of the previously successful nano checkpoint-copy wrapper.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import re
import types
import xml.etree.ElementTree as ET

CASES = {'2V_1NN_D110', '2V_2NN_D100', '2V_5NN_D310'}
BINARY_SHA256 = 'd8e498c583ef8bc69e71ddc89c83bc33b06d583d6bc22ab39cc84e2c02bd022b'
PSEUDO_SHA256 = '450c947cbbf30c33a5a77685bbe9a270947816f8a93837c02c749d0ee4e3ab4a'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def local_file(root, rel):
    name = PurePosixPath(rel)
    require(not name.is_absolute() and '..' not in name.parts and '\\' not in rel,
            'Unsafe parent relative path: ' + rel)
    path = root.joinpath(*name.parts)
    require(path.resolve().is_relative_to(root.resolve()), 'Parent path escaped root')
    require(not any(p.is_symlink() for p in [path, *path.parents] if p != root.parent),
            'Symlink in parent path: ' + rel)
    require(path.is_file(), 'Missing parent file: ' + rel)
    return path


def validate_parent(c):
    parent = Path(c['parent_attempt'])
    require(parent.is_dir() and not parent.is_symlink(), 'Parent attempt missing or symlink')
    evidence = c['parent_evidence_sha256']
    require({'pw.in', 'pw.out', 'result.json', 'provenance.json'} <= set(evidence),
            'Incomplete parent evidence hashes')
    for rel, expected in evidence.items():
        require(sha(local_file(parent, rel)) == expected, 'Parent evidence changed: ' + rel)
    provenance = json.loads((parent / 'provenance.json').read_text())
    result = json.loads((parent / 'result.json').read_text())
    case = provenance['case']['id']
    require(case in CASES and c['prefix'] == 'bulk_' + case + '_k4_vcr_20261015',
            'Wrong parent case/prefix')
    require(c.get('id', case) == case and result['case'] == case, 'Parent case identity mismatch')
    require(provenance['case']['prefix'] == c['prefix'], 'Parent provenance prefix mismatch')
    require(provenance['case']['nat'] == 106, 'Parent is not a 106-atom divacancy')
    require(sha(parent / 'pw.in') == provenance['case']['input_sha256'], 'Parent input provenance mismatch')
    require(provenance['binary_sha256'] == BINARY_SHA256, 'Parent QE binary changed')
    require(provenance['pseudo_sha256'] == PSEUDO_SHA256, 'Parent pseudopotential changed')
    require(provenance['case']['resources']['ntasks'] == 29 and
            provenance['case']['resources']['nodes'] == 1, 'Parent MPI allocation changed')
    command = provenance['command']
    require('-n' in command and command[command.index('-n') + 1] == '29',
            'Parent command did not use 29 ranks')
    require(not any(flag in command for flag in ['-nk', '-npool', '-ndiag', '-nb', '-nt']),
            'Parent command contains an unexpected parallel decomposition')
    require(result['returncode'] == 0 and result['accepted'] is False,
            'Parent is not a cleanly stopped unaccepted calculation')
    text = (parent / 'pw.out').read_text(errors='replace')
    require('Program stopped by user request' in text and 'JOB DONE.' in text,
            'Parent did not stop cleanly by user request')
    require(re.search(r'Calculation stopped in scf loop at iteration\s*#\s*15\b', text),
            'Unexpected parent restart stage')
    require('Forces acting on atoms' not in text and 'bfgs converged' not in text.lower(),
            'This adapter is scoped to the first unfinished SCF')

    scratch = parent / 'scratch'
    require(scratch.is_dir() and not scratch.is_symlink(), 'Missing or linked parent scratch')
    all_paths = list(scratch.rglob('*'))
    require(not any(p.is_symlink() for p in all_paths), 'Unexpected scratch symlink')
    files = {p.relative_to(scratch).as_posix(): p for p in all_paths if p.is_file()}
    inventory = c['parent_scratch_files']
    require(set(files) == set(inventory), 'Parent scratch inventory changed')
    prefix = c['prefix']
    require(len(files) == 119, 'Expected 119 first-SCF checkpoint files')
    for family in ['wfc', 'mix', 'restart_k', 'restart_scf']:
        members = [n for n in files if re.fullmatch(re.escape(prefix + '.' + family) + r'\d*', n)]
        require(len(members) == 29, 'Missing distributed ' + family + ' files')
    for rel in ['charge-density.dat', 'data-file-schema.xml', 'paw.txt']:
        require(prefix + '.save/' + rel in files, 'Missing saved-state file: ' + rel)
    require(not any(n.endswith('.EXIT') for n in files), 'EXIT marker in copied scratch')
    for rel, path in files.items():
        record = inventory[rel]
        require(path.stat().st_size == record['bytes'] and record['bytes'] > 0,
                'Parent scratch size changed: ' + rel)
        require(sha(path) == record['sha256'], 'Parent scratch hash changed: ' + rel)
    xml = ET.parse(files[prefix + '.save/data-file-schema.xml']).getroot()
    require(xml.findtext('parallel_info/nprocs') == '29' and
            xml.findtext('parallel_info/nthreads') == '1' and
            xml.findtext('parallel_info/npool') == '1', 'Saved-state MPI layout changed')
    require(xml.find('output/atomic_structure').get('nat') == '106', 'Saved-state nat changed')
    require(float(xml.findtext('output/band_structure/nelec')) == 318, 'Saved-state electron count changed')
    require(xml.findtext('output/convergence_info/scf_conv/convergence_achieved') == 'false',
            'Unexpected completed SCF in first-SCF restart')
    return parent, files


def copy_restart(c, attempt):
    """Reuse the exact old copy function with this parent's validator binding."""
    import run_nano_qe_restart_20261004 as frozen
    attempt = Path(attempt)
    require(not any(attempt.glob('*.EXIT')), 'New attempt contains a stop marker')
    require(not attempt.resolve().is_relative_to(Path(c['parent_attempt']).resolve()),
            'Cannot restart in immutable parent')
    copied_function = types.FunctionType(
        frozen.copy_restart.__code__,
        dict(frozen.__dict__, validate_parent=validate_parent),
        frozen.copy_restart.__name__, frozen.copy_restart.__defaults__, frozen.copy_restart.__closure__)
    parent = copied_function(c, attempt)
    require(not any((attempt / 'scratch').rglob('*.EXIT')), 'Copied stop marker')
    copied_names = {p.relative_to(attempt / 'scratch').as_posix()
                    for p in (attempt / 'scratch').rglob('*') if p.is_file()}
    require(copied_names == set(c['parent_scratch_files']), 'Copied scratch inventory changed')
    return parent
