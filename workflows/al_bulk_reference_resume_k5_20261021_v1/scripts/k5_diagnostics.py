"""Matched bulk k5 SCF diagnostics, reusing the frozen QE inspectors.

This module never submits jobs. A valid SCF is distinct from an accepted
relaxation endpoint. k4-to-k5 agreement is a sensitivity check only.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import run_luder_qe_20260930 as BASE
import run_bulk_vacancy_qe_vcrelax_20260925 as PARSER

IDS = ['0V', '1V', '2V_1NN_D110', '2V_2NN_D100', '2V_5NN_D310']
PP = '450c947cbbf30c33a5a77685bbe9a270947816f8a93837c02c749d0ee4e3ab4a'
require = BASE.require


def make_input(parent_text, accepted_result, case_id):
    require(case_id in IDS, 'Unexpected bulk case')
    require(accepted_result.get('case', case_id) == case_id, 'Parent case mismatch')
    require(accepted_result.get('accepted') is True, 'Parent is not accepted')
    require(accepted_result.get('final_scf_verified') is True, 'Parent final SCF missing')
    require(accepted_result.get('full_stress_target_pass') is True, 'Parent full stress failed')
    require(accepted_result.get('fmax_eV_A', math.inf) <= .001, 'Parent force failed')
    require(accepted_result.get('max_force_component_eV_A', math.inf) <= .0005, 'Parent force component failed')
    require(accepted_result.get('stress_max_abs_GPa', math.inf) <= .05, 'Parent full stress failed')
    parsed = PARSER.parse_input(parent_text)
    nat = 108 if case_id == '0V' else (107 if case_id == '1V' else 106)
    require(parsed['nat'] == accepted_result['n_atoms'] == nat, 'Parent atom count mismatch')
    require(parsed['cards']['K_POINTS']['rows'] == [['4', '4', '4', '0', '0', '0']], 'Parent is not unshifted k4')
    blocks = deepcopy(parsed['blocks'])
    prefix = 'bulk_' + case_id + '_k5_scf_20261021'
    ctl = blocks['control']
    ctl.update(calculation="'scf'", prefix=repr(prefix), restart_mode="'from_scratch'",
               pseudo_dir="'./pseudo'", outdir="'./scratch'", disk_io="'medium'",
               max_seconds='255600')
    for name in ['nstep', 'etot_conv_thr', 'forc_conv_thr']:
        ctl.pop(name, None)
    blocks['electrons'].update(startingpot="'atomic'", startingwfc="'atomic+random'")
    rows = []
    for name in ['control', 'system', 'electrons']:
        rows += ['&' + name.upper()]
        rows += [f'  {k} = {v},' for k, v in blocks[name].items()]
        rows += ['/']
    cell = accepted_result['cell_A']
    positions = accepted_result['positions_A']
    require(len(cell) == 3 and all(len(x) == 3 for x in cell) and len(positions) == nat,
            'Incomplete accepted geometry')
    require(all(math.isfinite(v) for row in cell + positions for v in row), 'Nonfinite geometry')
    rows += ['ATOMIC_SPECIES', 'Al 26.9815385 Al_PAW_PBE.UPF', 'CELL_PARAMETERS angstrom']
    rows += [' '.join(format(v, '.16g') for v in row) for row in cell]
    rows += ['ATOMIC_POSITIONS angstrom']
    rows += ['Al ' + ' '.join(format(v, '.16g') for v in row) for row in positions]
    rows += ['K_POINTS automatic', '5 5 5 0 0 0']
    text = '\n'.join(rows) + '\n'
    c = {'id': case_id, 'nat': nat, 'calculation': 'scf',
         'kind': 'bulk_fixed_k4_geometry_k5_sensitivity', 'prefix': prefix,
         'cell_A': deepcopy(cell), 'positions_A': deepcopy(positions),
         'kmesh': [5, 5, 5], 'nbnd': 200,
         'input_sha256': hashlib.sha256(text.encode()).hexdigest(),
         'parent_input_sha256': hashlib.sha256(parent_text.encode()).hexdigest(),
         'disk_io_change': 'low to medium; execution memory/disk policy only'}
    return text, c


def inspect(attempt, case, returncode=0):
    """Return accepted SCF diagnostic status; never promote to relaxed endpoint."""
    attempt = Path(attempt)
    r = {'case': case['id'], 'accepted': False, 'accepted_relaxed_endpoint': False,
         'status': 'not_accepted_scf_diagnostic', 'geometry_compatible': False}
    try:
        require(returncode == 0, 'MPI process returned nonzero')
        err = (attempt / 'pw.stderr').read_text(errors='replace') if (attempt / 'pw.stderr').exists() else ''
        require(not re.search(r'oom-kill|out of memory|killed by signal|MPI_ABORT|BAD TERMINATION', err, re.I),
                'MPI/memory failure')
        require(BASE.sha(attempt / 'pseudo/Al_PAW_PBE.UPF') == PP, 'Wrong PP hash')
        r.update(BASE.inspect(attempt, case))
        require(r.get('accepted') is True, 'Frozen SCF inspector rejected: ' + str(r.get('error')))
        require(r['final_scf_accuracy_Ry'] <= 1e-10, 'SCF accuracy above strict threshold')
        xi = ET.parse(attempt / 'scratch' / (case['prefix'] + '.save') / 'data-file-schema.xml').getroot().find('input')
        require(xi is not None, 'Missing XML input')
        for path, target in [('control_variables/calculation', 'scf'),
                             ('control_variables/prefix', case['prefix']),
                             ('dft/functional', 'PBE'),
                             ('atomic_species/species/pseudo_file', 'Al_PAW_PBE.UPF')]:
            require(xi.findtext(path) == target, 'XML setting mismatch: ' + path)
        require(int(xi.findtext('bands/nbnd')) == 200, 'XML band count mismatch')
        require(xi.findtext('bands/smearing') == 'mp', 'XML smearing mismatch')
        require(abs(2 * float(xi.find('bands/smearing').attrib['degauss']) - .02) < 1e-12,
                'XML smearing width mismatch')
        for path, target in [('basis/ecutwfc', 58.798915481),
                             ('basis/ecutrho', 470.3913238484),
                             ('electron_control/conv_thr', 1e-10)]:
            require(abs(2 * float(xi.findtext(path)) - target) <= max(1e-20, target * 1e-12),
                    'XML numerical setting mismatch: ' + path)
        kp = xi.find('k_points_IBZ/monkhorst_pack')
        require([int(kp.attrib[k]) for k in ['nk1', 'nk2', 'nk3', 'k1', 'k2', 'k3']] == [5,5,5,0,0,0],
                'XML mesh/offset mismatch')
        ai = xi.find('atomic_structure')
        require(int(ai.attrib['nat']) == case['nat'], 'XML input nat mismatch')
        cell = [[float(v) * BASE.BOHR for v in ai.find('cell/' + k).text.split()] for k in ['a1','a2','a3']]
        pos = [[float(v) * BASE.BOHR for v in x.text.split()] for x in ai.findall('atomic_positions/atom')]
        require(len(pos) == case['nat'], 'XML input position count')
        require(max(abs(v-w) for x,y in zip(cell,case['cell_A']) for v,w in zip(x,y)) < 2e-7,
                'XML input cell mismatch')
        require(max(abs(v-w) for x,y in zip(pos,case['positions_A']) for v,w in zip(x,y)) < 2e-7,
                'XML input position mismatch')
        text = (attempt / 'pw.out').read_text(errors='replace')
        num = lambda x: float(x.replace('D','E').replace('d','e'))
        vals = re.findall(r'!\s+total energy\s*=\s*([\d.EeDd+\-]+)\s+Ry', text)
        require(vals and abs(num(vals[-1]) * BASE.HA / 2 - r['energy_eV']) < 2e-6,
                'XML/stdout energy mismatch')
        # Reuse the frozen bulk parser's total-force selection, not the last
        # printed component: verbosity=high also prints component force rows.
        parts = re.split(r'Forces acting on atoms', text, flags=re.I)
        require(len(parts)>1,'Missing total-force heading')
        ff = re.findall(r'atom\s+\d+\s+type\s+\d+\s+force\s*=\s*([\d.EeDd+\-]+)\s+([\d.EeDd+\-]+)\s+([\d.EeDd+\-]+)', parts[-1].split('total   stress',1)[0])
        require(len(ff) >= case['nat'], 'Missing stdout forces')
        ff = [[num(v) * BASE.HA / (2*BASE.BOHR) for v in row] for row in ff[:case['nat']]]
        require(max(abs(v-w) for x,y in zip(ff,r['forces_eV_A']) for v,w in zip(x,y)) < 3e-7,
                'XML/stdout force mismatch')
        stress_blocks = re.findall(r'total\s+stress\s*\([^\n]*\)\s*[^\n]*\n((?:[^\n]*\n){3})',text,re.I)
        require(stress_blocks,'Missing stdout total stress')
        sr = [re.findall(r'[\d.EeDd+\-]+',line) for line in stress_blocks[-1].splitlines()]
        require(len(sr)==3 and all(len(row)>=6 for row in sr),'Incomplete stdout stress')
        stdout_stress = [[num(v)*BASE.GPA/2 for v in row[:3]] for row in sr]
        require(max(abs(v-w) for x,y in zip(stdout_stress,r['stress_QE_GPa']) for v,w in zip(x,y))<.00008,
                'XML/stdout stress mismatch')
        stress_max = max(abs(v) for row in r['stress_QE_GPa'] for v in row)
        compatible = r['fmax_eV_A'] <= .001 and r['max_force_component_eV_A'] <= .0005 and stress_max <= .05
        r.update(accepted=True, status='accepted_scf_diagnostic', accepted_relaxed_endpoint=False,
                 geometry_compatible=compatible, stress_max_abs_GPa=stress_max,
                 scientific_scope='k5 fixed at accepted k4 relaxed geometry; not a k5 relaxation',
                 limitations='A k4-to-k5 sensitivity check only; further sampling, smearing and finite-size validation may be needed.')
    except Exception as exc:
        r.update(accepted=False, status='not_accepted_scf_diagnostic',
                 accepted_relaxed_endpoint=False, geometry_compatible=False, error=str(exc))
    return r


def quantities(records):
    require(set(records) == set(IDS), 'Need exactly the five matched states')
    for cid in IDS:
        require(records[cid].get('accepted') is True, 'Unaccepted state: ' + cid)
        require(math.isfinite(records[cid]['energy_eV']), 'Nonfinite state energy')
        require(records[cid]['n_atoms'] == (108 if cid=='0V' else 107 if cid=='1V' else 106),
                'Wrong atom count for energy reference: ' + cid)
    e = {k: records[k]['energy_eV'] for k in IDS}
    q = {'E_ref_Al_eV': e['0V']/108, 'Ef_1V_eV': e['1V']-107/108*e['0V']}
    for cid in IDS[2:]:
        q['Ef_' + cid + '_eV'] = e[cid]-106/108*e['0V']
        q['Eb_' + cid + '_eV'] = 2*e['1V']-e[cid]-e['0V']
    q['D12_eV'] = e[IDS[2]]-e[IDS[3]]
    q['D51_eV'] = e[IDS[4]]-e[IDS[2]]
    return q


def evaluate(k4, k5):
    """Five-state report. 5 meV is an operational resolution, not certification."""
    q4, q5 = quantities(k4), quantities(k5)
    checks = []
    changes = {key: 1000*(q5[key]-q4[key]) for key in q4}
    for key, delta in changes.items():
        if key != 'E_ref_Al_eV':
            checks.append({'criterion': key + ' change', 'actual_meV': delta,
                           'limit_meV': 5.0, 'passed': abs(delta) <= 5.0})
    responses = {}
    for cid in IDS:
        a,b = k4[cid],k5[cid]
        require(len(a['positions_A']) == len(b['positions_A']) == a['n_atoms'], 'Matched geometry atom count')
        require(max(abs(v-w) for x,y in zip(a['cell_A'],b['cell_A']) for v,w in zip(x,y)) < 2e-7,
                'Not a matched fixed-geometry cell: ' + cid)
        # XML may wrap positions. Compare Cartesian difference in fractional coordinates.
        ca, cb, cc = a['cell_A']
        cross = lambda x,y: [x[1]*y[2]-x[2]*y[1],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0]]
        dot = lambda x,y: sum(v*w for v,w in zip(x,y))
        det = dot(ca,cross(cb,cc)); require(det>0,'Invalid matched cell')
        rec = [cross(cb,cc),cross(cc,ca),cross(ca,cb)]
        for x,y in zip(a['positions_A'],b['positions_A']):
            frac = [dot([v-w for v,w in zip(x,y)],z)/det for z in rec]
            require(max(abs(v-round(v)) for v in frac)<2e-7,'Not matched atom coordinates: '+cid)
        require(len(a['forces_eV_A']) == len(b['forces_eV_A']), 'Force count mismatch')
        df = max(math.sqrt(sum((v-w)**2 for v,w in zip(x,y))) for x,y in zip(a['forces_eV_A'],b['forces_eV_A']))
        ds = max(abs(v-w) for x,y in zip(a['stress_QE_GPa'],b['stress_QE_GPa']) for v,w in zip(x,y))
        stress = max(abs(v) for row in b['stress_QE_GPa'] for v in row)
        for label,actual,limit in [('force norm',b['fmax_eV_A'],.001),
                                   ('force component',b['max_force_component_eV_A'],.0005),
                                   ('full stress',stress,.05)]:
            checks.append({'criterion':cid+' '+label,'actual':actual,'limit':limit,'passed':actual<=limit})
        responses[cid] = {'max_atomwise_force_vector_change_eV_A':df,
                          'max_stress_component_change_GPa':ds}
    passed = all(c['passed'] for c in checks)
    return {'passed':passed,'status':'operational_sensitivity_pass' if passed else 'further_review_required',
            'k4_quantities':q4,'k5_quantities':q5,'changes_meV':changes,
            'force_stress_response':responses,'checks':checks,
            'definition':'Ef(nV)=E(nV)-(108-n)/108 E(0V); Eb(2V)=2E(1V)-E(2V)-E(0V), evaluated separately at each mesh.',
            'scope':'Matched fixed-geometry k4-to-k5 sensitivity, not proof of k convergence or a k5 relaxation.',
            'resolution_note':'Differences comparable to 5 meV remain unresolved; no automatic k6 or relaxation is launched.'}
