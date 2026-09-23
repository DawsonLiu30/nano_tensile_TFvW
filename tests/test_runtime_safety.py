"""Regression tests for preserving data and rejecting unsafe execution plans."""
import importlib.util
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from sync_wsl_source import synchronize
from divacancy_run_control import validate


class RuntimeSafety(unittest.TestCase):
    def test_sync_preserves_multiple_previous_copies_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source, runtime = base / 'source', base / 'runtime'
            (source / 'app').mkdir(parents=True)
            (source / 'app/dft_engine.py').write_text('# fixture')
            (source / 'entry.sh').write_bytes(b'echo safe\r\n')
            (source / '.venv').mkdir()
            (source / '.venv/private').write_text('omit')
            synchronize(source, runtime)
            self.assertEqual((runtime / 'repo/entry.sh').read_bytes(), b'echo safe\n')
            self.assertFalse((runtime / 'repo/.venv').exists())
            (runtime / 'repo/runtime_only_result').write_text('preserve me')
            second = synchronize(source, runtime)
            self.assertEqual((Path(second['previous_copy']) / 'runtime_only_result').read_text(), 'preserve me')
            synchronize(source, runtime)
            self.assertEqual(len(list((runtime / 'repo-history').iterdir())), 2)
            self.assertTrue((runtime / 'repo/SOURCE_SYNC_MANIFEST.json').is_file())

    def test_sync_rejects_containment(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / 'app').mkdir()
            (source / 'app/dft_engine.py').write_text('# fixture')
            with self.assertRaises(ValueError):
                synchronize(source, source / 'runtime')

    def test_manifest_override_duplicate_and_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / 'pair_scan/pair_01'
            case.mkdir(parents=True)
            manifest = dict(setting='pair_01', scan_type='pair', spacing_A=.20,
                            fmax_eV_per_A=.005, pristine_n_atoms=108, vacancy_n_atoms=106,
                            relax_steps=1000, kedf_x=.9, kedf_y=.1,
                            pair_selection='fixed_direction', pair_direction_indices=[1, 1, 0])
            (case / 'point_manifest.json').write_text(json.dumps(manifest))
            settings = root / 'settings_pair_scan.txt'
            settings.write_text('pair_01\n')
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(validate(root), ['pair_01'])
            for direction in ('-1,-1,0', '2,2,0', '[1 1 0]'):
                with patch.dict(os.environ, {'DIRECTION': direction}, clear=True):
                    self.assertEqual(validate(root), ['pair_01'])
            with patch.dict(os.environ, {'FMAX': '.01'}, clear=True):
                with self.assertRaises(ValueError):
                    validate(root)
            settings.write_text('pair_01\npair_01\n')
            with self.assertRaises(ValueError):
                validate(root)
            settings.write_text('../escape\n')
            with self.assertRaises(ValueError):
                validate(root)

    def test_single_case_runner_rejects_path_traversal(self):
        # This module requires the actual rebuilt scientific environment.
        import run_dftpy_vcrelax_vacancy_one as runner
        with tempfile.TemporaryDirectory() as temporary:
            for setting in ('../escape', '/tmp/escape', 'bad/name', '..'):
                with self.assertRaises(ValueError):
                    runner.resolve_case(Path(temporary), setting, 'pair')

    def test_no_qualified_points_removes_stale_collector_figure(self):
        from collect_dftpy_conventional_vacancy import plot_scan
        with tempfile.TemporaryDirectory() as temporary:
            figure = Path(temporary) / 'curve.png'
            figure.write_bytes(b'old valid figure')
            plot_scan(figure, [{'qualified': False}], xkey='pair_distance_A', xlabel='r', title='fixture')
            self.assertFalse(figure.exists())

    @unittest.skipUnless(sys.platform == 'linux', 'WSL/Linux shell regression')
    def test_qe_job_done_without_convergence_fails_and_retry_preserves_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            runtime, data = base / 'runtime', base / 'data'
            binary = runtime / 'env/bin'
            binary.mkdir(parents=True)
            (binary / 'pw.x').write_text('#!/bin/bash\necho "JOB DONE."\n')
            (binary / 'mpirun').write_text('#!/bin/bash\nshift 2\nexec "$@"\n')
            for path in binary.iterdir():
                path.chmod(0o755)
            reference = data / '03_ACTIVE_QE_VCRELAX_REFERENCE'
            for name in ('pristine_vcrelax', 'vacancy_vcrelax'):
                (reference / name).mkdir(parents=True)
                (reference / name / 'pw.in').write_text('fixture input only\n')
            (reference / 'pseudo').mkdir()
            (reference / 'pseudo/Al_PAW_PBE.UPF').write_text('fixture pseudo only\n')
            env = dict(os.environ, AL_DEFECTS_RUNTIME=str(runtime),
                       AL_DEFECTS_ENV_PREFIX=str(runtime / 'env'), AL_DEFECTS_DATA_ROOT=str(data),
                       QE_RUNROOT=str(runtime / 'runs/qe'))
            command = ['bash', str(ROOT / 'scripts/run_local_qe_vaclm_wsl.sh'), '--run', '--case', 'pristine', '--np', '1']
            first = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(first.returncode, 1, first.stdout + first.stderr)
            second = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(second.returncode, 1, second.stdout + second.stderr)
            archives = list((runtime / 'runs/qe/audit').glob('*/pw.out'))
            self.assertEqual(len(archives), 1)
            self.assertEqual(archives[0].read_text(), 'JOB DONE.\n')


if __name__ == '__main__':
    unittest.main()
