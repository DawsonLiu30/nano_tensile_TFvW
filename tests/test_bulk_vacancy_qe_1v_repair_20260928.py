"""Repair allowlist, chained provenance and synthetic-process tests; no QE jobs."""
import copy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('repair_1v', Path(__file__).parents[1] / 'scripts/run_bulk_vacancy_qe_1v_repair_20260928.py')
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)
qe = repair.base


def fake_output(parsed, component=0.0, accuracy=1e-12):
    coordinates = '\n'.join(' '.join(row) for row in parsed['cards']['ATOMIC_POSITIONS']['rows'])
    cell = '\n'.join(' '.join(row) for row in parsed['cards']['CELL_PARAMETERS']['rows'])
    body = ('Program PWSCF v.7.5 starts\n'
            'bfgs converged in 2 scf cycles and 1 bfgs steps\n'
            'Final enthalpy = -4226.7 Ry\nBegin final coordinates\n'
            'CELL_PARAMETERS (angstrom)\n' + cell + '\nATOMIC_POSITIONS (crystal)\n' + coordinates + '\nEnd final coordinates\n'
            'Final scf calculation at the relaxed structure.\n'
            '! total energy = -4226.7 Ry\nestimated scf accuracy < ' + str(accuracy) + ' Ry\n'
            'convergence has been achieved in 1 iterations\nForces acting on atoms\n')
    body += '\n'.join(f'atom {i+1} type 1 force = {component / qe.FORCE_TO_EV_A if i == 0 else 0.0:.16g} 0.0 0.0' for i in range(parsed['nat']))
    body += '\ntotal stress (Ry/bohr**3) (kbar) P= 0.0\n0 0 0 0 0 0\n0 0 0 0 0 0\n0 0 0 0 0 0\nJOB DONE.\n'
    return body


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.campaign = self.root / 'campaign'
        shutil.copytree(qe.DEFAULT_CAMPAIGN, self.campaign, copy_function=shutil.copyfile)

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self):
        return json.loads((self.campaign / 'qe_manifest.json').read_text(encoding='utf-8'))

    def save_manifest(self, value):
        qe.save_json(self.campaign / 'qe_manifest.json', value)

    def change_input(self, parameter, value, protocol_key=None):
        manifest = self.manifest()
        case = manifest['cases']['1V']
        path = self.campaign / case['input']
        text = path.read_text(encoding='utf-8')
        text, n = re.subn(r'(?m)^(\s*' + parameter + r'\s*=\s*)[^,\n]+', lambda m: m[1] + str(value), text)
        self.assertEqual(n, 1)
        path.write_text(text, encoding='utf-8')
        case['sha256'] = qe.sha_file(path)
        if protocol_key:
            manifest['protocol'][protocol_key] = qe.number(value)
        self.save_manifest(manifest)

    def change_evidence(self, name, transform):
        manifest = self.manifest()
        item = manifest['cases']['1V']['continuation_sources'][name]
        path = self.campaign / item['file']
        value = json.loads(path.read_text(encoding='utf-8'))
        transform(value)
        qe.save_json(path, value)
        item['sha256'] = qe.sha_file(path)
        self.save_manifest(manifest)

    def test_real_campaign_and_exact_two_differences(self):
        verified = repair.verify_campaign(self.campaign)
        self.assertEqual(set(verified['cases']), {'1V'})
        case = verified['cases']['1V']
        self.assertEqual(case['parsed']['nat'], 107)
        self.assertEqual(case['parsed']['conv_thr_Ry'], 1e-10)
        self.assertEqual(qe.number(case['parsed']['blocks']['ions']['trust_radius_min']), 1e-6)
        self.assertEqual(verified['manifest']['protocol']['force_norm_target_eV_A'], .001)
        self.assertEqual(len(case['continuation_evidence']), 7)
        self.assertNotIn('upscale', case['parsed']['blocks']['ions'])
        self.assertNotIn('parent_default', case['optimizer_repair'])

    def test_parent_output_tamper_rejected_after_manifest_rehash(self):
        manifest = self.manifest()
        item = manifest['cases']['1V']['continuation_sources']['parent_pw_out']
        path = self.campaign / item['file']
        path.write_bytes(path.read_bytes() + b' altered\n')
        item['sha256'] = qe.sha_file(path)
        self.save_manifest(manifest)
        with self.assertRaisesRegex(qe.CampaignError, 'parent record hash mismatch'):
            repair.verify_campaign(self.campaign)

    def test_fake_parent_identity_rejected(self):
        self.change_evidence('parent_attempt', lambda v: v.update(campaign_id='fake_campaign'))
        with self.assertRaisesRegex(qe.CampaignError, 'unexpected parent campaign'):
            repair.verify_campaign(self.campaign)

    def test_fake_upstream_hash_rejected(self):
        self.change_evidence('parent_attempt', lambda v: v['parent_attempt'].update(runtime_input_sha256='0' * 64))
        with self.assertRaisesRegex(qe.CampaignError, 'upstream immutable evidence mismatch'):
            repair.verify_campaign(self.campaign)

    def test_parent_input_tamper_rejected_after_manifest_rehash(self):
        manifest = self.manifest()
        item = manifest['cases']['1V']['continuation_sources']['parent_pw_in']
        path = self.campaign / item['file']
        path.write_bytes(path.read_bytes() + b'\n! changed parent input\n')
        item['sha256'] = qe.sha_file(path)
        self.save_manifest(manifest)
        with self.assertRaisesRegex(qe.CampaignError, 'parent record hash mismatch'):
            repair.verify_campaign(self.campaign)

    def test_parent_conv_and_absence_of_explicit_trust_min_enforced(self):
        verified = repair.verify_campaign(self.campaign)
        case = verified['cases']['1V']
        parent = qe.parse_input(case['continuation_evidence']['parent_pw_in'].decode('utf-8-sig'))
        source = qe.parse_input(case['continuation_evidence']['parent_source_input'].decode('utf-8-sig'))
        protocol = verified['manifest']['protocol']
        for field, value in [('conv_thr', '1e-7'), ('trust_radius_min', '1e-4')]:
            with self.subTest(field=field):
                p, s = copy.deepcopy(parent), copy.deepcopy(source)
                block = 'electrons' if field == 'conv_thr' else 'ions'
                for item in (p, s):
                    item['blocks'][block][field] = value
                    item['scientific']['namelists'][block][field] = value
                with self.assertRaises(qe.CampaignError):
                    repair.validate_repair_settings(protocol, case['parsed'], p, s)

    def test_wrong_minimum_radius_rejected(self):
        self.change_input('trust_radius_min', '1.0d-5')
        with self.assertRaisesRegex(qe.CampaignError, 'Input trust_radius_min must equal'):
            repair.verify_campaign(self.campaign)

    def test_wrong_new_conv_rejected_even_with_updated_metadata(self):
        self.change_input('conv_thr', '1.0d-9', 'conv_thr_Ry')
        manifest = self.manifest()
        manifest['protocol']['optimizer_repair']['conv_thr_Ry'] = 1e-9
        self.save_manifest(manifest)
        with self.assertRaisesRegex(qe.CampaignError, 'Repair conv_thr must equal'):
            repair.verify_campaign(self.campaign)

    def test_other_optimizer_change_rejected_even_with_updated_protocol(self):
        self.change_input('nstep', 201, 'nstep')
        with self.assertRaisesRegex(qe.CampaignError, 'two-item repair allowlist'):
            repair.verify_campaign(self.campaign)

    def test_upscale_not_silently_added(self):
        manifest = self.manifest()
        case = manifest['cases']['1V']
        path = self.campaign / case['input']
        path.write_text(path.read_text(encoding='utf-8').replace('&IONS', '&IONS\n  upscale = 100'), encoding='utf-8')
        case['sha256'] = qe.sha_file(path)
        self.save_manifest(manifest)
        with self.assertRaisesRegex(qe.CampaignError, 'two-item repair allowlist'):
            repair.verify_campaign(self.campaign)

    def test_missing_rationale_rejected(self):
        manifest = self.manifest()
        manifest['protocol']['optimizer_repair']['rationale'] = ' '
        self.save_manifest(manifest)
        with self.assertRaisesRegex(qe.CampaignError, 'rationale must be nonempty'):
            repair.verify_campaign(self.campaign)

    def test_wrong_completed_scf_line_rejected(self):
        def change(v):
            v['source_lines_1based']['scf_convergence'] -= 1
        self.change_evidence('aligned_step', change)
        with self.assertRaisesRegex(qe.CampaignError, 'missing completed SCF evidence'):
            repair.verify_campaign(self.campaign)

    def test_original_parent_mapping_preserved(self):
        manifest = self.manifest()
        item = manifest['geometry_manifest']
        path = self.campaign / item['file']
        geometry = json.loads(path.read_text(encoding='utf-8'))
        geometry['cases']['1V']['remaining_parent_ids'][0:2] = [1, 0]
        qe.save_json(path, geometry)
        item['sha256'] = qe.sha_file(path)
        self.save_manifest(manifest)
        with self.assertRaisesRegex(qe.CampaignError, 'parent site mapping changed'):
            repair.verify_campaign(self.campaign)

    def test_final_scf_component_gate_stricter_than_norm_gate(self):
        parsed = repair.verify_campaign(self.campaign)['cases']['1V']['parsed']
        result = qe.parse_output(fake_output(parsed, component=.0008), parsed)
        self.assertTrue(result['force_norm_target_pass'])
        self.assertAlmostEqual(result['max_force_component_eV_A'], .0008)
        self.assertAlmostEqual(result['force_component_target_eV_A'], .0005)
        self.assertFalse(result['force_component_target_pass'])
        self.assertFalse(result['usable'])
        self.assertEqual(result['status'], 'converged_needs_review')
        self.assertIn('force component', ' '.join(result['data_quality_reasons']))
        self.assertTrue(qe.parse_output(fake_output(parsed, component=.00049), parsed)['usable'])

    def test_new_scf_threshold_and_bfgs_confirmation_remain_required(self):
        parsed = repair.verify_campaign(self.campaign)['cases']['1V']['parsed']
        result = qe.parse_output(fake_output(parsed, accuracy=1e-9), parsed)
        self.assertFalse(result['usable'])
        self.assertFalse(result['electronic_converged'])
        body = fake_output(parsed).replace('bfgs converged in 2 scf cycles and 1 bfgs steps', 'bfgs failed after 2 scf cycles and 1 bfgs steps')
        result = qe.parse_output(body, parsed)
        self.assertFalse(result['usable'])
        self.assertTrue(result['ionic_optimizer_failed'])

    def test_attempt_snapshots_repair_and_chained_evidence(self):
        parsed = repair.verify_campaign(self.campaign)['cases']['1V']['parsed']
        fake = self.root / 'fake.py'
        fake.write_text('print(' + repr(fake_output(parsed)) + ')', encoding='utf-8')
        command = '"' + sys.executable.replace('\\', '/') + '" "' + str(fake).replace('\\', '/') + '"'
        result = qe.run(self.campaign, self.root / 'runs', '1V', execute=True, pw=command)
        self.assertTrue(result['usable'], result['reasons'])
        directory = Path(result['attempt'])
        record = json.loads((directory / 'attempt.json').read_text(encoding='utf-8'))
        self.assertEqual(record['max_seconds'], 169200)
        self.assertIn('source/runner.py', record['immutable_file_sha256'])
        self.assertIn('source/repair_runner.py', record['immutable_file_sha256'])
        self.assertEqual(len(record['repair_path']), 3)
        self.assertEqual(record['repair_path'][-1], qe.CAMPAIGN_ID)
        self.assertEqual(record['optimizer_repair']['conv_thr_Ry'], 1e-10)
        self.assertTrue(record['repair_rationale'])
        self.assertTrue(qe.inspect_attempt(directory, self.campaign)['usable'])
        path = directory / 'source/repair_runner.py'
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b'\n# changed\n')
        self.assertEqual(qe.inspect_attempt(directory, self.campaign)['status'], 'provenance_failed')


if __name__ == '__main__':
    unittest.main()
