"""Migration checks use real campaign inputs and synthetic process output, not production QE."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('qe_portable', Path(__file__).parents[1] / 'scripts/qe_portable.py')
qe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qe)


def output(energy=-444.044, version='7.5', accuracy='3.0E-10'):
    return (f'Program PWSCF v.{version} starts\n'
            f'!    total energy = {energy:.11f} Ry\n'
            f'     estimated scf accuracy < {accuracy} Ry\n'
            '     convergence has been achieved in 20 iterations\n     JOB DONE.\n')


class PortableQETests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.campaign = self.root / 'campaign'
        shutil.copytree(qe.DEFAULT_CAMPAIGN, self.campaign)
        self.work = self.root / 'work'

    def tearDown(self):
        self.temp.cleanup()

    def fake_run(self, version='7.5', code=0):
        fake = self.root / 'fake_pw.py'
        fake.write_text('import sys\nprint(' + repr(output(version=version)) + f')\nsys.exit({code})\n', encoding='utf-8')
        command = '"' + sys.executable.replace('\\', '/') + '" "' + str(fake).replace('\\', '/') + '"'
        return qe.run(self.campaign, self.work, '2V_2NN_D100', execute=True, pw=command)

    def test_real_campaign_and_archived_completion(self):
        verified = qe.verify_campaign(self.campaign)
        self.assertEqual({r['parsed']['nat'] for r in verified['cases'].values()}, {106})
        report = qe.collect(self.campaign, self.work)
        self.assertEqual(report['complete_cases'], ['2V_1NN_D110'])
        self.assertTrue(all(not row['complete'] and row['delta_Ry'] is None for row in report['pairs']))

    def test_corrupt_input_rejected(self):
        p = self.campaign / 'inputs/2V_2NN_D100/pw.in'
        p.write_text(p.read_text(encoding='utf-8') + '\n! changed\n')
        with self.assertRaises(qe.CampaignError):
            qe.verify_campaign(self.campaign)

    def test_prepare_preserves_science_and_never_runs(self):
        result = qe.prepare(self.campaign, self.work, '2V_D310_r1', max_seconds=10000, threads=2)
        self.assertFalse(result['launched'])
        p = Path(result['attempt'])
        self.assertFalse((p / 'pw.out').exists())
        record = json.loads((p / 'attempt.json').read_text(encoding='utf-8'))
        source = qe.verify_campaign(self.campaign)['cases']['2V_D310_r1']
        self.assertEqual(record['scientific_sha256'], source['scientific_sha256'])
        self.assertEqual(record['threads']['OMP_NUM_THREADS'], '2')
        second = qe.prepare(self.campaign, self.work, '2V_D310_r1')
        self.assertNotEqual(result['attempt'], second['attempt'])
        self.assertTrue(p.is_dir())

    def test_refuse_run_without_explicit_execute(self):
        with self.assertRaises(qe.CampaignError):
            qe.run(self.campaign, self.work, '2V_2NN_D100')

    def test_refuse_scratch_inside_repo(self):
        with self.assertRaises(qe.CampaignError):
            qe.prepare(self.campaign, qe.REPO_ROOT / 'scratch-test', '2V_2NN_D100')

    def test_lock_prevents_duplicate_case(self):
        self.work.mkdir()
        with qe.case_lock(self.work, '2V_2NN_D100'):
            with self.assertRaises(qe.CampaignError):
                qe.prepare(self.campaign, self.work, '2V_2NN_D100')

    def test_job_done_alone_is_not_converged(self):
        self.assertFalse(qe.parse_output('JOB DONE.', 1e-9, 0)['complete'])
        self.assertFalse(qe.parse_output(output(accuracy='1e-4'), 1e-9, 0)['complete'])
        self.assertFalse(qe.parse_output(output() + '\nconvergence NOT achieved\n', 1e-9, 0)['complete'])

    def test_failed_process_is_not_accepted(self):
        self.assertFalse(self.fake_run(code=9)['complete'])
        report = qe.collect(self.campaign, self.work)
        self.assertFalse(report['pairs'][0]['complete'])

    def test_complete_pair_uses_verified_archived_reference(self):
        self.assertTrue(self.fake_run()['complete'])
        pair = qe.collect(self.campaign, self.work)['pairs'][0]
        self.assertTrue(pair['complete'])
        self.assertAlmostEqual(pair['delta_Ry'], -444.04662686 - (-444.044), places=9)

    def test_qe_version_mismatch_blocks_automatic_pair(self):
        self.fake_run(version='7.3.1')
        pair = qe.collect(self.campaign, self.work)['pairs'][0]
        self.assertFalse(pair['complete'])
        self.assertIsNone(pair['delta_Ry'])

    def test_modified_output_and_archive_rejected(self):
        result = self.fake_run()
        (Path(result['attempt']) / 'pw.out').write_text(output(energy=-500))
        self.assertFalse(qe.collect(self.campaign, self.work)['pairs'][0]['complete'])
        archive = self.campaign / 'evidence/2V_1NN_D110/pw.out'
        archive.write_text(archive.read_text(encoding='utf-8') + 'modified')
        with self.assertRaises(qe.CampaignError):
            qe.collect(self.campaign, self.work)


if __name__ == '__main__':
    unittest.main()
