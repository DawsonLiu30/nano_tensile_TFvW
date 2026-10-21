#!/usr/bin/env python3
"""One-case BFGS repair with exactly two reviewed numerical changes.

Only IONS.trust_radius_min=1e-6 and ELECTRONS.conv_thr=1e-10 may differ
from the continuation parent (conv_thr=1e-8, no explicit trust_radius_min).
Force/stress targets, physical settings, original IDs and raw provenance remain
mandatory. This wrapper never relabels a failed parent as an accepted result.
"""
from __future__ import annotations
import copy
import importlib.util
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve()
SPEC = importlib.util.spec_from_file_location('qe_1v_repair_base', HERE.with_name('run_bulk_vacancy_qe_vcrelax_20260925.py'))
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
base.CAMPAIGN_ID = 'al_bulk_vacancy_1v_repair_20260928'
base.DEFAULT_CAMPAIGN = base.REPO_ROOT / 'campaigns' / base.CAMPAIGN_ID
base.CASE_IDS = {'1V'}
PARENT_CAMPAIGN_ID = 'al_bulk_vacancy_vcrelax_continue_20260926'
SOURCE_NAMES = {'parent_pw_in', 'parent_pw_out', 'parent_attempt', 'parent_manifest', 'parent_source_input', 'parent_geometry', 'aligned_step'}
_verify = base.verify_campaign
_create = base.create_attempt
_parse_output = base.parse_output
CampaignError = base.CampaignError


def require(condition, message):
    if not condition:
        raise CampaignError(message)


def settings_without_geometry(parsed):
    result = copy.deepcopy(parsed['scientific'])
    for key in ('CELL_PARAMETERS', 'ATOMIC_POSITIONS'):
        del result['cards'][key]
    return result


def validate_repair_settings(protocol, parsed, parsed_parent, parsed_source):
    repair = protocol.get('optimizer_repair')
    require(isinstance(repair, dict), 'protocol.optimizer_repair is required')
    required = {'kind', 'trust_radius_min_bohr', 'conv_thr_Ry', 'parent_conv_thr_Ry', 'rationale'}
    require(required <= set(repair), 'Incomplete optimizer_repair metadata')
    require(repair['kind'] == 'bfgs_small_step_with_tighter_scf', 'Unexpected optimizer_repair kind')
    require(isinstance(repair['rationale'], str) and repair['rationale'].strip(), 'Repair rationale must be nonempty')
    require(base.number(repair['trust_radius_min_bohr']) == 1e-6, 'Repair trust_radius_min must equal 1e-6 bohr')
    require(base.number(repair['conv_thr_Ry']) == base.number(protocol.get('conv_thr_Ry')) == 1e-10,
            'Repair conv_thr must equal 1e-10 Ry')
    require(base.number(repair['parent_conv_thr_Ry']) == 1e-8, 'Parent conv_thr metadata must equal 1e-8 Ry')
    parent_settings = settings_without_geometry(parsed_parent)
    source_settings = settings_without_geometry(parsed_source)
    require(parent_settings == source_settings, 'Parent runtime/source scientific settings disagree')
    require('trust_radius_min' not in parsed_parent['blocks']['ions'] and
            'trust_radius_min' not in parsed_source['blocks']['ions'], 'Parent must not explicitly set trust_radius_min')
    require(base.number(parsed_parent['blocks']['electrons'].get('conv_thr')) == 1e-8,
            'Parent conv_thr must equal 1e-8 Ry')
    require(base.number(parsed['blocks']['ions'].get('trust_radius_min')) == 1e-6,
            'Input trust_radius_min must equal 1e-6 bohr')
    require(base.number(parsed['blocks']['electrons'].get('conv_thr')) == 1e-10,
            'Input conv_thr must equal 1e-10 Ry')
    normalized = settings_without_geometry(parsed)
    del normalized['namelists']['ions']['trust_radius_min']
    normalized['namelists']['electrons']['conv_thr'] = parent_settings['namelists']['electrons']['conv_thr']
    require(normalized == parent_settings, 'Scientific settings changed outside the two-item repair allowlist')
    return copy.deepcopy(repair)


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
        require(parent.get('campaign_id') == PARENT_CAMPAIGN_ID and parent_manifest.get('campaign_id') == PARENT_CAMPAIGN_ID,
                f'{cid}: unexpected parent campaign')
        upstream = parent.get('parent_attempt')
        require(isinstance(upstream, dict) and upstream.get('case_id') == cid and
                upstream.get('campaign_id') == 'al_bulk_vacancy_vcrelax_20260925', f'{cid}: missing original parent identity')
        require(parent.get('nat') == upstream.get('nat') == 107 and parent.get('q') == upstream.get('q') == 1,
                f'{cid}: upstream atom/vacancy inventory changed')
        for blob, key in (('parent_pw_in', 'runtime_input_sha256'), ('parent_pw_out', 'output_sha256'),
                          ('parent_manifest', 'manifest_sha256'), ('parent_source_input', 'source_input_sha256')):
            require(base.sha_bytes(blobs[blob]) == parent.get(key), f'{cid}: parent record hash mismatch for {blob}')
        require(parent_manifest['cases'][cid]['sha256'] == parent['source_input_sha256'], f'{cid}: parent manifest input mismatch')
        require(parent.get('pseudo_sha256') == parent_manifest['pseudo']['sha256'] == verified['pseudo_sha256'], f'{cid}: pseudopotential changed')
        require(base.sha_bytes(blobs['parent_geometry']) == parent_manifest['geometry_manifest']['sha256'], f'{cid}: parent geometry hash mismatch')
        require(aligned.get('eligible_as_new_BFGS_start') is True, f'{cid}: geometry not eligible')
        require(aligned['source_pw_out_sha256'] == base.sha_bytes(blobs['parent_pw_out']), f'{cid}: alignment/raw output mismatch')
        case['optimizer_repair'] = validate_repair_settings(verified['manifest']['protocol'], case['parsed'], parsed_parent, parsed_source)
        # The continuation record also anchors its original input/output evidence;
        # retaining that chain prevents accidental replacement by a fake parent.
        immutable = parent.get('immutable_file_sha256', {})
        for name, key in (('parent_parent_pw_in.in','runtime_input_sha256'), ('parent_parent_pw_out.out','output_sha256'),
                          ('parent_parent_manifest.json','manifest_sha256'), ('parent_parent_source_input.in','source_input_sha256')):
            require(immutable.get('source/' + name) == upstream.get(key), f'{cid}: upstream immutable evidence mismatch: {name}')
        require(parent.get('pseudo_sha256') == upstream.get('pseudo_sha256'), f'{cid}: upstream pseudopotential changed')
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
        require(base.number(energy[1]) == aligned['energy_Ry'] and
                0 <= base.number(accuracy[1]) == aligned['completed_scf_accuracy_Ry'] <= parsed_parent['conv_thr_Ry'],
                f'{cid}: completed parent SCF values mismatch')
        require('Forces acting on atoms' in lines[loc['forces_header']-1] and
                re.search(r'total\s+stress', lines[loc['stress_header']-1]), f'{cid}: completed force/stress headers missing')
        require(not any(re.match(r'\s*(CELL_PARAMETERS|ATOMIC_POSITIONS)\b', line)
                        for line in lines[loc['positions_last']:loc['stress_header']]), f'{cid}: force/stress belongs to a later geometry')
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
        removed = geometry['removed_parent_ids']
        require(len(removed) == 1 and type(removed[0]) is int and 0 <= removed[0] < 108 and
                geometry['remaining_parent_ids'] == [i for i in range(108) if i not in removed], f'{cid}: invalid original site inventory')
        case['continuation_evidence'] = blobs
    return verified


def create_attempt(verified, root, case_id, command, threads=1, max_seconds=base.MAX_SECONDS):
    directory, record = _create(verified, root, case_id, command, threads, max_seconds)
    blobs = verified['cases'][case_id]['continuation_evidence']
    extras = {'source/repair_runner.py': HERE.read_bytes()}
    extras.update({'source/parent_' + name + ('.out' if name == 'parent_pw_out' else '.json' if name not in {'parent_pw_in', 'parent_source_input'} else '.in'): value for name, value in blobs.items()})
    for name, data in extras.items():
        base.snapshot_file(directory / name, data)
        record['immutable_file_sha256'][name] = base.sha_bytes(data)
    record['execution_mode'] = '1V_new_BFGS_verified_parent_geometry_smaller_trust_radius_tighter_SCF'
    record['continuation_sources'] = verified['cases'][case_id]['metadata']['continuation_sources']
    record['parent_attempt'] = json.loads(blobs['parent_attempt'])
    record['parent_completed_step'] = json.loads(blobs['aligned_step'])
    record['optimizer_repair'] = copy.deepcopy(verified['cases'][case_id]['optimizer_repair'])
    record['repair_rationale'] = record['optimizer_repair']['rationale']
    record['repair_path'] = [record['parent_attempt']['parent_attempt']['campaign_id'],
                             record['parent_attempt']['campaign_id'], base.CAMPAIGN_ID]
    record['repair_default_note'] = ('No numeric parent trust-radius default is used as a validation premise; '
                                    'QE 7.5 documentation and implementation report different defaults. '
                                    'This input explicitly requests 1e-6 bohr.')
    base.save_json(directory / 'attempt.json', record)
    return directory, record


def parse_output(output, parsed, returncode=0, stderr='', force_target_eV_A=0.001):
    """Retain all base gates and check the rebuilt final SCF component forces."""
    result = _parse_output(output, parsed, returncode, stderr, force_target_eV_A)
    forces = result.get('forces_eV_A')
    valid = (result.get('final_scf_verified') and isinstance(forces, list) and
             len(forces) == parsed['nat'] and all(isinstance(row, list) and len(row) == 3 and
             all(isinstance(x, (int, float)) and math.isfinite(x) for x in row) for row in forces))
    maximum = max(abs(x) for row in forces for x in row) if valid else None
    target = parsed['force_threshold_Ry_bohr'] * base.FORCE_TO_EV_A
    component_pass = maximum is not None and maximum <= target + 1e-8
    result.update(max_force_component_eV_A=maximum, force_component_target_eV_A=target,
                  force_component_target_pass=component_pass, force_component_rounding_tolerance_eV_A=1e-8)
    if result['usable'] and not component_pass:
        result['usable'] = False
        result['status'] = 'converged_needs_review'
        result['data_quality_reasons'].append('Final relaxed-cell SCF force component exceeds input forc_conv_thr')
    return result


base.verify_campaign = verify_campaign
base.create_attempt = create_attempt
base.parse_output = parse_output
main = base.main
if __name__ == '__main__':
    raise SystemExit(main())
