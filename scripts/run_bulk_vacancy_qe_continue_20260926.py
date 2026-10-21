#!/usr/bin/env python3
"""Verified four-case continuation: fresh electronic state and new BFGS history."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve()
SPEC = importlib.util.spec_from_file_location('qe_continuation_base', HERE.with_name('run_bulk_vacancy_qe_vcrelax_20260925.py'))
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
base.CAMPAIGN_ID = 'al_bulk_vacancy_vcrelax_continue_20260926'
base.DEFAULT_CAMPAIGN = base.REPO_ROOT / 'campaigns' / base.CAMPAIGN_ID
base.CASE_IDS = {'1V', '2V_1NN_D110', '2V_3NN_D211', '2V_5NN_D310'}
SOURCE_NAMES = {'parent_pw_in', 'parent_pw_out', 'parent_attempt', 'parent_manifest', 'parent_source_input', 'parent_geometry', 'aligned_step'}
_verify = base.verify_campaign
_create = base.create_attempt
CampaignError = base.CampaignError


def require(condition, message):
    if not condition:
        raise CampaignError(message)


def settings_without_geometry(parsed):
    result = copy.deepcopy(parsed['scientific'])
    for key in ('CELL_PARAMETERS', 'ATOMIC_POSITIONS'):
        del result['cards'][key]
    return result


def verify_campaign(campaign):
    verified = _verify(campaign)
    geometries = json.loads(verified['geometry_bytes'])['cases']
    for cid, case in verified['cases'].items():
        source_refs = case['metadata'].get('continuation_sources', {})
        require(set(source_refs) == SOURCE_NAMES, f'{cid}: all parent evidence files are required')
        blobs = {name: base.checked_bytes(base.campaign_file(verified['root'], item['file']), item['sha256'], cid + ' ' + name)
                 for name, item in source_refs.items()}
        parent = json.loads(blobs['parent_attempt'])
        parent_manifest = json.loads(blobs['parent_manifest'])
        aligned = json.loads(blobs['aligned_step'])
        parsed_parent = base.parse_input(blobs['parent_pw_in'].decode('utf-8-sig'))
        parsed_source = base.parse_input(blobs['parent_source_input'].decode('utf-8-sig'))
        require(parent.get('case_id') == cid and aligned.get('case_id') == cid, f'{cid}: parent identity mismatch')
        require(parent.get('campaign_id') == 'al_bulk_vacancy_vcrelax_20260925', f'{cid}: unexpected parent campaign')
        for blob, key in (('parent_pw_in', 'runtime_input_sha256'), ('parent_pw_out', 'output_sha256'),
                          ('parent_manifest', 'manifest_sha256'), ('parent_source_input', 'source_input_sha256')):
            require(base.sha_bytes(blobs[blob]) == parent.get(key), f'{cid}: parent record hash mismatch for {blob}')
        require(parent_manifest['cases'][cid]['sha256'] == parent['source_input_sha256'], f'{cid}: parent manifest input mismatch')
        require(parent.get('pseudo_sha256') == parent_manifest['pseudo']['sha256'] == verified['pseudo_sha256'], f'{cid}: pseudopotential changed')
        require(base.sha_bytes(blobs['parent_geometry']) == parent_manifest['geometry_manifest']['sha256'], f'{cid}: parent geometry hash mismatch')
        require(aligned.get('eligible_as_new_BFGS_start') is True, f'{cid}: geometry not eligible')
        require(aligned['source_pw_out_sha256'] == base.sha_bytes(blobs['parent_pw_out']), f'{cid}: alignment/raw output mismatch')
        common = settings_without_geometry(case['parsed'])
        require(common == settings_without_geometry(parsed_parent) == settings_without_geometry(parsed_source), f'{cid}: scientific settings changed from parent')
        lines = blobs['parent_pw_out'].decode('utf-8', errors='replace').splitlines()
        loc = aligned['source_lines_1based']
        require(0 < loc['cell_header'] < loc['positions_header'] < loc['positions_first'] <= loc['positions_last'] < loc['total_energy'] < loc['scf_accuracy'] < loc['scf_convergence'] < loc['forces_header'] < loc['stress_header'] <= len(lines), f'{cid}: invalid completed-step ordering')
        require('CELL_PARAMETERS (angstrom)' in lines[loc['cell_header'] - 1] and 'ATOMIC_POSITIONS (crystal)' in lines[loc['positions_header'] - 1], f'{cid}: geometry units mismatch')
        cell = [[base.number(x) for x in lines[loc['cell_header'] + i].split()] for i in range(3)]
        rows = [lines[i - 1].split() for i in range(loc['positions_first'], loc['positions_last'] + 1)]
        fractions = [[base.number(x) for x in row[1:4]] for row in rows]
        symbols = [row[0] for row in rows]
        require(cell == aligned['cell_A'] and fractions == aligned['fractional_positions'] and symbols == aligned['symbols'], f'{cid}: aligned geometry differs from raw output')
        require(len(rows) == case['parsed']['nat'] and all(len(row) == 4 for row in rows), f'{cid}: atom count/order mismatch')
        between = lines[loc['positions_last']:loc['scf_convergence']]
        require(not any(re.match(r'\s*(CELL_PARAMETERS|ATOMIC_POSITIONS)\b', line) for line in between), f'{cid}: SCF belongs to a later geometry')
        require('convergence has been achieved' in lines[loc['scf_convergence'] - 1], f'{cid}: missing completed SCF evidence')
        energy = re.search(r'!\s+total energy\s*=\s*(' + base.NUMBER + ')', lines[loc['total_energy'] - 1])
        accuracy = re.search(r'estimated scf accuracy\s*<\s*(' + base.NUMBER + ')', lines[loc['scf_accuracy'] - 1])
        require(energy is not None and accuracy is not None, f'{cid}: energy/accuracy evidence missing')
        require(base.number(energy[1]) == aligned['energy_Ry'] and base.number(accuracy[1]) == aligned['completed_scf_accuracy_Ry'] < case['parsed']['conv_thr_Ry'], f'{cid}: completed SCF values mismatch')
        require(case['parsed']['cell_A'] == cell and case['parsed']['cards']['ATOMIC_POSITIONS']['units'] == 'crystal', f'{cid}: continuation cell/units mismatch')
        actual_frac = [[base.number(x) for x in row[1:]] for row in case['parsed']['cards']['ATOMIC_POSITIONS']['rows']]
        require(actual_frac == fractions, f'{cid}: continuation coordinates mismatch')
        require(case['metadata']['cell_A'] == cell, f'{cid}: per-case cell metadata mismatch')
        geometry_hash = base.sha_json({'cell_A': cell, 'fractional_positions': fractions, 'symbols': symbols, 'pbc': [True, True, True]})
        require(case['metadata']['completed_geometry_sha256'] == geometry_hash, f'{cid}: per-case geometry hash mismatch')
        geometry = geometries[cid]
        parent_geometry = json.loads(blobs['parent_geometry'])['cases'][cid]
        require(geometry['cell_A'] == cell and geometry['fractional'] == fractions, f'{cid}: geometry manifest mismatch')
        for key in ('remaining_parent_indices_0based', 'remaining_parent_ids', 'removed_parent_ids'):
            require(geometry[key] == parent_geometry[key], f'{cid}: parent site mapping changed')
        case['continuation_evidence'] = blobs
    return verified


def create_attempt(verified, root, case_id, command, threads=1, max_seconds=base.MAX_SECONDS):
    directory, record = _create(verified, root, case_id, command, threads, max_seconds)
    blobs = verified['cases'][case_id]['continuation_evidence']
    extras = {'source/continuation_runner.py': HERE.read_bytes()}
    extras.update({'source/parent_' + name + ('.out' if name == 'parent_pw_out' else '.json' if name not in {'parent_pw_in', 'parent_source_input'} else '.in'): value for name, value in blobs.items()})
    for name, data in extras.items():
        base.snapshot_file(directory / name, data)
        record['immutable_file_sha256'][name] = base.sha_bytes(data)
    record['execution_mode'] = 'fresh_electronic_new_BFGS_at_verified_parent_completed_SCF_geometry'
    record['continuation_sources'] = verified['cases'][case_id]['metadata']['continuation_sources']
    record['parent_attempt'] = json.loads(blobs['parent_attempt'])
    record['parent_completed_step'] = json.loads(blobs['aligned_step'])
    base.save_json(directory / 'attempt.json', record)
    return directory, record


base.verify_campaign = verify_campaign
base.create_attempt = create_attempt
main = base.main
if __name__ == '__main__':
    raise SystemExit(main())
