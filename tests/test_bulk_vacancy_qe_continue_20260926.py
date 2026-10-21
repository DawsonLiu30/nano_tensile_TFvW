"""Continuation provenance and fake-process tests; no production QE is launched."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('continuation', Path(__file__).parents[1] / 'scripts/run_bulk_vacancy_qe_continue_20260926.py')
cont = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cont)
qe = cont.base


class ContinuationTests(unittest.TestCase):
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

    def test_four_per_case_cells_and_unchanged_scientific_settings(self):
        verified = cont.verify_campaign(self.campaign)
        self.assertEqual(set(verified['cases']), qe.CASE_IDS)
        self.assertEqual(len({json.dumps(c['parsed']['cell_A']) for c in verified['cases'].values()}), 4)
        for cid, case in verified['cases'].items():
            self.assertEqual(qe.number(case['parsed']['blocks']['control']['max_seconds']), 169200)
            self.assertEqual(qe.unquote(case['parsed']['blocks']['ions']['ion_dynamics']), 'bfgs')
            self.assertEqual(len(case['continuation_evidence']), 7)

    def test_parent_output_tampering_rejected_even_after_manifest_rehash(self):
        manifest = self.manifest()
        item = manifest['cases']['1V']['continuation_sources']['parent_pw_out']
        path = self.campaign / item['file']
        with path.open('ab') as handle:
            handle.write(b' altered\n')
        item['sha256'] = qe.sha_file(path)
        self.save_manifest(manifest)
        with self.assertRaisesRegex(qe.CampaignError, 'parent record hash mismatch'):
            cont.verify_campaign(self.campaign)

    def test_rehashed_global_scientific_change_cannot_bypass_parent_check(self):
        manifest = self.manifest()
        for case in manifest['cases'].values():
            path = self.campaign / case['input']
            text = path.read_text(encoding='utf-8')
            self.assertIn('nstep = 200', text)
            path.write_text(text.replace('nstep = 200', 'nstep = 201'), encoding='utf-8')
            case['sha256'] = qe.sha_file(path)
        manifest['protocol']['nstep'] = 201
        self.save_manifest(manifest)
        with self.assertRaisesRegex(qe.CampaignError, 'scientific settings changed from parent'):
            cont.verify_campaign(self.campaign)

    def test_wrong_completed_scf_line_rejected_even_when_alignment_rehashed(self):
        manifest = self.manifest()
        item = manifest['cases']['2V_1NN_D110']['continuation_sources']['aligned_step']
        path = self.campaign / item['file']
        aligned = json.loads(path.read_text(encoding='utf-8'))
        aligned['source_lines_1based']['scf_convergence'] -= 1
        qe.save_json(path, aligned)
        item['sha256'] = qe.sha_file(path)
        self.save_manifest(manifest)
        with self.assertRaisesRegex(qe.CampaignError, 'missing completed SCF evidence'):
            cont.verify_campaign(self.campaign)

    def test_new_attempt_snapshots_dependencies_and_parent_evidence(self):
        verified = cont.verify_campaign(self.campaign)
        parsed = verified['cases']['1V']['parsed']
        coordinates = '\n'.join(' '.join(row) for row in parsed['cards']['ATOMIC_POSITIONS']['rows'])
        cell = '\n'.join(' '.join(row) for row in parsed['cards']['CELL_PARAMETERS']['rows'])
        body = ('Program PWSCF v.7.5 starts\n'
                'bfgs converged in 2 scf cycles and 1 bfgs steps\n'
                'Final enthalpy = -4226.7 Ry\nBegin final coordinates\n'
                'CELL_PARAMETERS (angstrom)\n' + cell + '\nATOMIC_POSITIONS (crystal)\n' + coordinates + '\nEnd final coordinates\n'
                'Final scf calculation at the relaxed structure.\n'
                '! total energy = -4226.7 Ry\nestimated scf accuracy < 1.0E-12 Ry\n'
                'convergence has been achieved in 1 iterations\nForces acting on atoms\n')
        body += '\n'.join(f'atom {i+1} type 1 force = 0.0 0.0 0.0' for i in range(parsed['nat']))
        body += '\ntotal stress (Ry/bohr**3) (kbar) P= 0.0\n0 0 0 0 0 0\n0 0 0 0 0 0\n0 0 0 0 0 0\nJOB DONE.\n'
        fake = self.root / 'fake.py'
        fake.write_text('print(' + repr(body) + ')', encoding='utf-8')
        command = '"' + sys.executable.replace('\\', '/') + '" "' + str(fake).replace('\\', '/') + '"'
        result = qe.run(self.campaign, self.root / 'runs', '1V', execute=True, pw=command)
        self.assertTrue(result['usable'], result['reasons'])
        directory = Path(result['attempt'])
        record = json.loads((directory / 'attempt.json').read_text(encoding='utf-8'))
        self.assertEqual(record['max_seconds'], 169200)
        self.assertIn('source/runner.py', record['immutable_file_sha256'])
        self.assertIn('source/continuation_runner.py', record['immutable_file_sha256'])
        self.assertEqual(record['parent_attempt']['case_id'], '1V')
        self.assertTrue(qe.inspect_attempt(directory, self.campaign)['usable'])
        path = directory / 'source/continuation_runner.py'
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b'\n# changed\n')
        self.assertEqual(qe.inspect_attempt(directory, self.campaign)['status'], 'provenance_failed')

    def test_sbatch_declares_four_way_parallelism_and_extended_budget(self):
        text = (qe.REPO_ROOT / 'hpc/bulk_vacancy_qe_continue_20260926.sbatch').read_text(encoding='utf-8')
        self.assertIn('#SBATCH --array=0-3%4', text)
        self.assertIn('#SBATCH --time=48:00:00', text)
        self.assertIn('--max-seconds 169200', text)


if __name__ == '__main__':
    unittest.main()
