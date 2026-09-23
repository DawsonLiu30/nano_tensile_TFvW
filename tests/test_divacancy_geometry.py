"""Independent geometric regression checks; no electronic calculations run."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from ase.io import read

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import divacancy_geometry as geometry
import prepare_dftpy_divacancy_rscan_20260616 as dftpy
import prepare_qe_divacancy_vcrelax_rscan_20260616 as qe


class DivacancyGeometryTests(unittest.TestCase):
    def setUp(self):
        self.a0 = 3.9545804060131293
        self.atoms, self.center, _ = geometry.build_centered_pristine(self.a0, (3, 3, 3))

    def test_corrected_three_points_are_110_and_shells_1_4_9(self):
        pairs = geometry.enumerate_pairs(self.atoms, self.center)
        self.assertEqual(len(pairs), 3)
        np.testing.assert_allclose([v[0] for v in pairs], np.arange(1, 4) * self.a0 / math.sqrt(2), atol=1e-12)
        shells = []
        for distance, index, delta in pairs:
            # ASE's MIC distance is independent of our helper implementation.
            self.assertAlmostEqual(self.atoms.get_distance(self.center, index, mic=True), distance, places=11)
            self.assertAlmostEqual(delta[0], delta[1], places=12)
            self.assertAlmostEqual(delta[2], 0, places=12)
            metadata = geometry.pair_geometry_metadata(self.atoms, self.center, index, delta, self.a0)
            self.assertEqual(metadata['pair_direction_indices'], [1, 1, 0])
            shells.append(metadata['fcc_shell_index'])
            defect = geometry.remove_two_atoms(self.atoms, self.center, index)
            self.assertEqual(len(defect), 106)
            expected = np.delete(self.atoms.positions, sorted((self.center, index)), axis=0)
            np.testing.assert_array_equal(defect.positions, expected)
        self.assertEqual(shells, [1, 4, 9])

    def test_shells_include_true_second_neighbour_100_and_third_211(self):
        pairs = geometry.enumerate_pairs(self.atoms, self.center, selection='shells')
        metadata = [geometry.pair_geometry_metadata(self.atoms, self.center, i, d, self.a0) for _, i, d in pairs]
        self.assertEqual([m['fcc_shell_index'] for m in metadata[:3]], [1, 2, 3])
        self.assertEqual([m['pair_direction_indices'] for m in metadata[:3]], [[1, 1, 0], [1, 0, 0], [2, 1, 1]])
        np.testing.assert_allclose([m['pair_distance_A'] for m in metadata[:3]],
                                   [self.a0 / math.sqrt(2), self.a0, self.a0 * math.sqrt(6) / 2])
        fixed100 = geometry.enumerate_pairs(self.atoms, self.center, direction=(1, 0, 0))
        self.assertEqual(len(fixed100), 1)
        self.assertAlmostEqual(fixed100[0][0], self.a0, places=12)

    def test_other_signed_and_out_of_plane_directions_are_available(self):
        for direction, expected in [((1, -1, 0), 3), ((0, 1, 1), 3), ((1, 1, 1), 1), ((3, 1, 0), 1)]:
            pairs = geometry.enumerate_pairs(self.atoms, self.center, direction=direction)
            self.assertEqual(len(pairs), expected)
            for distance, index, delta in pairs:
                self.assertLess(np.linalg.norm(np.cross(delta, direction)), 1e-10)
                self.assertAlmostEqual(distance, self.atoms.get_distance(self.center, index, mic=True), places=11)

    def test_wrapping_and_half_cell_ties_do_not_change_scan(self):
        baseline = geometry.enumerate_pairs(self.atoms, self.center)
        self.assertAlmostEqual(baseline[-1][0], math.sqrt(2) * self.atoms.cell.lengths()[0] / 2)
        self.assertEqual(len(geometry.minimum_image_vectors(baseline[-1][2], self.atoms.cell, self.atoms.pbc)), 4)
        # Move the vacancy centre across the origin: raw Cartesian separations
        # now cross the cell boundaries, but physical MIC pairs must be unchanged.
        shifted = self.atoms.copy()
        shifted.translate([0.61 * self.atoms.cell[0, 0], -0.47 * self.atoms.cell[1, 1], 0.3])
        shifted.wrap()
        for direction in ((1, 1, 0), (1, -1, 0)):
            pairs = geometry.enumerate_pairs(shifted, self.center, direction=direction)
            self.assertEqual(len(pairs), 3)
            np.testing.assert_allclose([p[0] for p in pairs], [p[0] for p in baseline], atol=1e-11)
            for r, i, delta in pairs:
                self.assertAlmostEqual(r, shifted.get_distance(self.center, i, mic=True), places=11)
                geometry.pair_geometry_metadata(shifted, self.center, i, delta, self.a0)

    def test_rectangular_repeat_keeps_mic_bounds(self):
        atoms, center, _ = geometry.build_centered_pristine(self.a0, (2, 3, 4))
        pairs = geometry.enumerate_pairs(atoms, center)
        self.assertEqual(len(pairs), 2)
        for distance, index, delta in pairs:
            self.assertTrue(np.all(np.abs(delta) <= atoms.cell.lengths() / 2 + 1e-12))
            self.assertAlmostEqual(distance, atoms.get_distance(center, index, mic=True), places=11)

    def test_generators_share_geometry_but_keep_distinct_method_defaults(self):
        d = dftpy.parse_args(['--outdir', 'unused'])
        q = qe.parse_args(['--outdir', 'unused', '--pseudo', 'unused'])
        self.assertEqual((d.kedf_x, d.kedf_y, d.fmax, d.a0), (0.9, 0.1, 0.005, self.a0))
        self.assertEqual((q.a0, q.ecut, q.force_conv, q.kmesh), (4.039848, 800.0, 0.002, '3x3x3'))
        for module in (dftpy, qe):
            atoms, center, _ = module.build_centered_pristine(self.a0, (3, 3, 3))
            generated = module.enumerate_pairs(atoms, center)
            np.testing.assert_allclose([p[0] for p in generated], [2.796310621839334, 5.592621243678668, 8.388931865518002])

    def test_invalid_arguments_and_geometry_fail_explicitly(self):
        for value in ('0,0,0', '1,1', 'a,1,0', '1.5,1,0'):
            with self.assertRaises(argparse.ArgumentTypeError):
                geometry.parse_direction(value)
        self.assertEqual(geometry.parse_direction('[-2 -2 0]'), (1, 1, 0))
        for value in ('0x3x3', '-1x3x3', '3x3', 'axbxc'):
            with self.assertRaises(argparse.ArgumentTypeError):
                geometry.parse_repeat(value)
        for module, base in ((dftpy, ['--outdir', 'unused']), (qe, ['--outdir', 'unused', '--pseudo', 'unused'])):
            for bad in (['--a0', 'nan'], ['--a0', '-1'], ['--repeat', '0x3x3'],
                        ['--direction=0,0,0'], ['--direction-tol', '-1'], ['--max-pairs', '-1'], ['--max-parallel', '0']):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    module.parse_args(base + bad)
        for bad in (['--spacing', 'nan'], ['--fmax', '0'], ['--kedf-x', '-1'], ['--kedf-y', 'nan']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                dftpy.parse_args(['--outdir', 'unused'] + bad)
        for bad in (['--ecut', '-1'], ['--kmesh', '0x3x3'], ['--kmesh', '3x3x3,4x4x4']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                qe.parse_args(['--outdir', 'unused', '--pseudo', 'unused'] + bad)
        with self.assertRaises(ValueError):
            geometry.remove_two_atoms(self.atoms, 1, 1)
        with self.assertRaises(ValueError):
            geometry.enumerate_pairs(self.atoms, -1)
        with self.assertRaises(ValueError):
            geometry.enumerate_pairs(self.atoms, self.center, selection='mixed')
        with self.assertRaises(ValueError):
            geometry.pair_geometry_metadata(self.atoms, self.center, 0, [99, 99, 0], self.a0)

    def test_cli_packages_are_portable_and_protect_existing_data(self):
        with tempfile.TemporaryDirectory(prefix='divacancy geometry ') as tmp:
            root = Path(tmp)
            pseudo = root / 'test-potential.dat'
            pseudo.write_text('Preparation-only fixture; no electronic calculation.\n')
            for module, ppflag, manifest_name in ((dftpy, '--pp', 'point_manifest.json'), (qe, '--pseudo', 'pair_manifest.json')):
                out = root / module.__name__
                command = [sys.executable, '-B', module.__file__, '--outdir', str(out), ppflag, str(pseudo)]
                completed = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                top = json.loads((out / 'manifest.json').read_text())
                self.assertEqual(top['pair_count'], 3)
                self.assertEqual(top['pair_selection'], 'fixed_direction')
                cases = sorted((out / 'pair_scan').glob('pair_*'))
                self.assertEqual(len(cases), 3)
                for i, case in enumerate(cases):
                    meta = json.loads((case / manifest_name).read_text())
                    self.assertEqual(meta['pair_direction_indices'], [1, 1, 0])
                    self.assertEqual(meta['fcc_shell_index'], (1, 4, 9)[i])
                    self.assertEqual(len(read(case / 'divacancy_start.vasp')), 106)
                    if module is dftpy:
                        self.assertTrue((case / meta['pp_file']).is_file())
                    else:
                        input_text = (case / 'divacancy_vcrelax/vc-relax.in').read_text()
                        self.assertIn("pseudo_dir = '../../../psp'", input_text)
                        self.assertTrue((case / 'divacancy_vcrelax/../../../psp' / meta['pseudo']).is_file())
                self.assertTrue((out / 'preparation_sources/divacancy_geometry.py').is_file())
                submit = next(out.glob('submit_*.sh')).read_text()
                self.assertIn('--array=0-2%', submit)
                marker = out / 'do-not-overwrite.txt'
                marker.write_text('research evidence')
                old_manifest = (out / 'manifest.json').read_bytes()
                refused = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(refused.returncode, 0)
                self.assertEqual(marker.read_text(), 'research evidence')
                self.assertEqual((out / 'manifest.json').read_bytes(), old_manifest)

                shells_out = root / (module.__name__ + '_shells')
                shells_cmd = command.copy()
                shells_cmd[shells_cmd.index('--outdir') + 1] = str(shells_out)
                shells_cmd += ['--pair-selection', 'shells', '--max-pairs', '2']
                completed = subprocess.run(shells_cmd, capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(json.loads((shells_out / 'manifest.json').read_text())['pair_count'], 2)
                meta = [json.loads(p.read_text()) for p in sorted(shells_out.glob('pair_scan/*/' + manifest_name))]
                self.assertEqual([m['fcc_shell_index'] for m in meta], [1, 2])
                self.assertEqual([m['pair_direction_indices'] for m in meta], [[1, 1, 0], [1, 0, 0]])
                self.assertIn('--array=0-1%', next(shells_out.glob('submit_*.sh')).read_text())


if __name__ == '__main__':
    unittest.main()
