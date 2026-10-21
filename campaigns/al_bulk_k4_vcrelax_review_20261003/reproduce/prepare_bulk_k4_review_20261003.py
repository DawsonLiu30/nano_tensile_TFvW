"""Prepare, never execute, five traceable k4 vc-relax inputs for review."""
from pathlib import Path
import hashlib
import json
import shutil
import re
import io
import csv
import argparse
import copy
from collections.abc import Mapping
import numpy as np
from ase.io import read, write
from ase.io.espresso import read_fortran_namelist
from ase.geometry import find_mic

W = Path(__file__).resolve().parents[1]
CASES = ['0V', '1V', '2V_1NN_D110', '2V_2NN_D100', '2V_5NN_D310']
PP_HASH = '450c947cbbf30c33a5a77685bbe9a270947816f8a93837c02c749d0ee4e3ab4a'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    def plain(v):
        if isinstance(v, Mapping):
            return {k: plain(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [plain(x) for x in v]
        return v
    Path(p).write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def verify(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    checks = []

    def check(name, ok, detail=None):
        checks.append({'check': name, 'passed': bool(ok), 'detail': detail})

    canonical = json.loads((root / 'sources/independently_decoded_endpoints.json').read_text())
    canonical = {r['case']: r for r in canonical if r['method'] == 'QE_PBE_PAW'}
    protocol = json.loads((root / 'protocol.json').read_text())
    check('protocol_hash', sha(root / 'protocol.json') == manifest['protocol_sha256'])
    check('five_selected_cases_only', [r['case'] for r in manifest['cases']] == CASES)
    shared = None
    for entry in manifest['cases']:
        case = entry['case']
        target = root / entry['input']
        source = root / entry['parent_input']
        check(case + ': input_hash', sha(target) == entry['input_sha256'])
        check(case + ': parent_hash', sha(source) == entry['parent_input_sha256'])
        check(case + ': pseudo_hash', sha(target.parent / 'pseudo/Al_PAW_PBE.UPF') == PP_HASH)
        old = source.read_text()
        new = target.read_text()
        check(case + ': all_cards_unchanged', old.split('ATOMIC_SPECIES', 1)[1] == new.split('ATOMIC_SPECIES', 1)[1])
        n0, c0 = read_fortran_namelist(io.StringIO(old))
        n1, c1 = read_fortran_namelist(io.StringIO(new))
        # Validate the whole namelist against a strict semantic change allowlist.
        expected = copy.deepcopy(dict(n0))
        expected['control'] = dict(expected['control'])
        expected['control'].update(calculation='vc-relax', prefix=entry['prefix'], max_seconds=255600)
        expected['ions'] = {'ion_dynamics': 'bfgs', 'trust_radius_min': 1e-6}
        expected['cell'] = {'cell_dynamics': 'bfgs', 'press': 0.0, 'press_conv_thr': 0.5, 'cell_dofree': 'all'}
        check(case + ': only_allowlisted_namelist_changes', dict(n1) == expected)
        check(case + ': cards_match_independent_parser', c0 == c1)
        a = read(target, format='espresso-in')
        b = read(source, format='espresso-in')
        q = 0 if case == '0V' else 1 if case == '1V' else 2
        check(case + ': atom_count', len(a) == 108 - q)
        check(case + ': full_pbc', a.pbc.tolist() == [True, True, True])
        check(case + ': species', set(a.get_chemical_symbols()) == {'Al'})
        check(case + ': geometry_matches_parent_exactly', np.array_equal(a.cell, b.cell) and np.array_equal(a.positions, b.positions))
        r = canonical[case]
        dc = float(np.max(np.abs(a.cell.array - np.asarray(r['cell_A']))))
        _, dr = find_mic(a.positions - np.asarray(r['positions_A']), a.cell, pbc=True)
        check(case + ': cell_matches_decoded_k3_endpoint', dc < 1e-8, {'max_abs_A': dc})
        check(case + ': positions_match_decoded_k3_endpoint', np.max(dr) < 1e-7, {'max_MIC_A': float(np.max(dr))})
        check(case + ': kmesh_4x4x4_unshifted', bool(re.search(r'K_POINTS automatic\s+4 4 4 0 0 0\s*$', new)))
        common = copy.deepcopy(dict(n1))
        common['control'] = dict(common['control'])
        common['system'] = dict(common['system'])
        del common['control']['prefix']
        del common['system']['nat']
        check(case + ': shared_protocol', shared is None or shared == common)
        shared = common
        check(case + ': manifest_protocol_matches_input', common == protocol['common_namelists'])
        check(case + ': nonzero_cell', a.get_volume() > 1000)
        rd = json.loads((source.parent / 'result.json').read_text())
        check(case + ': parent_is_accepted_SCF_diagnostic', rd['accepted'] and rd['calculation'] == 'scf')
        x = read(target.parent / 'initial.extxyz')
        check(case + ': exported_structure_roundtrip', len(x) == len(a) and np.allclose(x.cell, a.cell, atol=1e-8, rtol=0) and np.allclose(x.positions, a.positions, atol=1e-8, rtol=0))
    for f, expected_hash in manifest['source_sha256'].items():
        check('source:' + f, sha(root / f) == expected_hash)
    check('no_submission_script', not list(root.rglob('*.sbatch')))
    check('no_production_result', not list((root / 'inputs').rglob('pw.out')))
    report = {'scope': 'Local provenance/input/geometry checks only; no QE run, no convergence certification',
              'passed': all(r['passed'] for r in checks), 'check_count': len(checks), 'checks': checks}
    save(root / 'preparation_QA.json', report)
    if not report['passed']:
        raise AssertionError([r for r in checks if not r['passed']])
    return report


def prepare(root):
    evidence = W / 'outputs/luder_review_20261003/05_prior_evidence'
    pp = Path('/mnt/c/OFDFT/nano_tensile_TFvW/campaigns/al_nano_qe_memory_recovery_20261001/pseudo/Al_PAW_PBE.UPF')
    assert sha(pp) == PP_HASH
    root.mkdir(parents=True, exist_ok=False)
    (root / 'sources').mkdir()
    (root / 'reproduce').mkdir()
    shutil.copy2(__file__, root / 'reproduce' / Path(__file__).name)
    shutil.copy2(evidence / 'independently_decoded_endpoints.json', root / 'sources')
    repo = Path('/mnt/c/OFDFT/nano_tensile_TFvW')
    for name in ['docs/QE_PARAMETER_BASELINES_20260925.md', 'scripts/run_bulk_vacancy_qe_1v_repair_20260928.py']:
        shutil.copy2(repo / name, root / 'sources' / Path(name).name)
    entries = []
    for case in CASES:
        src = evidence / 'bulk_qe_diagnostics' / ('bulk_' + case + '_k4') / 'attempt_001'
        dest = root / 'inputs' / case
        dest.mkdir(parents=True)
        (dest / 'pseudo').mkdir()
        shutil.copy2(pp, dest / 'pseudo' / pp.name)
        saved = root / 'sources' / case
        saved.mkdir()
        for name in ['pw.in', 'pw.out', 'result.json', 'provenance.json']:
            shutil.copy2(src / name, saved / name)
        old = (src / 'pw.in').read_text()
        prefix = 'bulk_' + case + '_k4_vcr_20261003'
        new, count = re.subn(r"calculation = 'scf'", "calculation = 'vc-relax'", old)
        assert count == 1
        new, count = re.subn(r"prefix = '[^']+'", "prefix = '" + prefix + "'", new)
        assert count == 1
        new, count = re.subn(r'max_seconds = 169200', 'max_seconds = 255600', new)
        assert count == 1
        extra = "&IONS\n  ion_dynamics = 'bfgs',\n  trust_radius_min = 1.0d-6,\n/\n&CELL\n  cell_dynamics = 'bfgs',\n  press = 0.0,\n  press_conv_thr = 0.5,\n  cell_dofree = 'all',\n/\n"
        new = new.replace('ATOMIC_SPECIES\n', extra + 'ATOMIC_SPECIES\n', 1)
        (dest / 'pw.in').write_text(new, encoding='utf-8')
        a = read(dest / 'pw.in', format='espresso-in')
        write(dest / 'initial.extxyz', a, format='extxyz')
        write(dest / 'POSCAR', a, format='vasp', direct=True, vasp5=True)
        n, _ = read_fortran_namelist(io.StringIO(new))
        common = copy.deepcopy(dict(n))
        common['control'] = dict(common['control'])
        common['system'] = dict(common['system'])
        del common['control']['prefix']
        del common['system']['nat']
        entries.append({'case': case, 'nat': len(a), 'prefix': prefix,
                        'input': str((dest / 'pw.in').relative_to(root)), 'input_sha256': sha(dest / 'pw.in'),
                        'parent_input': str((saved / 'pw.in').relative_to(root)), 'parent_input_sha256': sha(saved / 'pw.in'),
                        'geometry_origin': 'Accepted k3 vc-relax endpoint, carried unchanged through accepted k4 fixed-geometry SCF',
                        'initial_volume_A3': a.get_volume(),
                        'initial_cell_length_spread_percent': float((max(a.cell.lengths()) / min(a.cell.lengths()) - 1) * 100)})
    protocol = {
        'status': 'review_prepared_not_submitted', 'common_namelists': common,
        'kmesh_and_offset': [4, 4, 4, 0, 0, 0], 'periodic_boundary_conditions': [True, True, True],
        'method': 'QE PBE PAW, unchanged pseudopotential', 'pseudo_sha256': PP_HASH,
        'resource_proposal': {'partition': 'ct56', 'nodes_per_case': 1, 'MPI_ranks_per_case': 29,
                             'threads_per_rank': 1, 'memory_GiB_per_case': 64, 'Slurm_hours': 72,
                             'QE_max_seconds': 255600, 'array_case_count': 5, 'array_concurrency_cap': 5,
                             'note': 'No sbatch or runner deployed. Check live queue, quotas and binary before submission. The k4 SCFs printed >36.58–37.04 GB estimates; 64 GiB is a proposal, not a RAM guarantee.'},
        'acceptance': {'normal_QE_exit_and_JOB_DONE': True, 'BFGS_converged': True,
                       'final_relaxed_cell_SCF_required': True, 'final_accuracy_Ry_max': 1e-10,
                       'final_max_force_component_eV_A': 0.0005, 'final_max_force_norm_eV_A': 0.001,
                       'final_full_stress_max_abs_GPa': 0.05, 'pressure_target_GPa': 0.0,
                       'all_six_independent_stress_components_required': True,
                       'final_XML_stdout_geometry_force_energy_alignment_required': True,
                       'normal_MPI_and_no_OOM_required': True, 'source_and_output_hashes_required': True},
        'allowed_changes_from_parent_k4_SCF': ['calculation scf -> vc-relax', 'new prefix',
                                              'max_seconds 169200 -> 255600', 'add BFGS IONS with trust_radius_min=1e-6',
                                              'add CELL BFGS, zero pressure, all cell freedoms, 0.5 kbar tolerance'],
        'optimizer_note': 'Uniform trust_radius_min=1e-6 borrows the successful 1V repair setting. This is an explicit optimizer change, not a force-tolerance relaxation.',
        'conditional_next_stage': 'After all five accepted k4 endpoints, prepare matched k5 SCFs from those new geometries. No k5 geometries or DFT results exist in this package.',
        'inference_limit': 'Selected Ef/Eb/D12/5NN-vs-1NN only; not all-six ordering, isolated-defect limit, dynamics or mechanism proof.'}
    save(root / 'protocol.json', protocol)
    source_hashes = {str(p.relative_to(root)): sha(p) for p in sorted((root / 'sources').rglob('*')) if p.is_file()}
    save(root / 'manifest.json', {'campaign': 'al_bulk_k4_vcrelax_review_20261003',
                                'status': 'prepared_for_review_only', 'submitted': False,
                                'protocol_sha256': sha(root / 'protocol.json'),
                                'cases': entries, 'source_sha256': source_hashes})
    with (root / 'five_case_review.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case', 'nat', 'initial_volume_A3', 'initial_cell_length_spread_percent', 'input', 'input_sha256'])
        writer.writeheader()
        writer.writerows({k: e[k] for k in writer.fieldnames} for e in entries)
    return verify(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--root', type=Path, default=W / 'outputs/bulk_k4_preparation_20261003')
    args = parser.parse_args()
    result = verify(args.root) if args.verify_only else prepare(args.root)
    print(json.dumps({'root': str(args.root), 'passed': result['passed'], 'checks': result['check_count'], 'DFT_launched': False}))
