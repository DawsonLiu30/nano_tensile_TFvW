"""Synthetic geometry regression tests; no private inputs, DFT, or scheduler."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from ase.io import read
from scipy.spatial import Voronoi, cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.ase_nanocrystal import _cross_section_mask, build_circular_nanowire, build_periodic_prism
from app.hex_111 import (SELECTION_MODES, _NORMALS, _UNIT_VERTICES,
                         _clip_hex, _polygon_area, _site_id, build_hex_111)


def keys(atoms):
    return list(zip(*(atoms.arrays[name].tolist() for name in ("lattice_i", "lattice_j", "layer_k"))))


def independent_selection(radius, mode, extent):
    """Independent finite Qhull tessellation, with all relevant cells bounded."""
    a0 = 4.0
    d = a0 / np.sqrt(2)
    u, v = np.array([d, 0.0]), np.array([d / 2, d * np.sqrt(3) / 2])
    result = set()
    for phase in range(3):
        lattice_keys = [(i, j) for i in range(-extent, extent + 1) for j in range(-extent, extent + 1)]
        xy = np.array([[-d / 2, phase * a0 / np.sqrt(6)] + i * u + j * v for i, j in lattice_keys])
        voronoi = Voronoi(xy)
        for index, center in enumerate(xy):
            if np.linalg.norm(center) > radius + d / np.sqrt(3) + 1e-8:
                continue
            region = voronoi.regions[voronoi.point_region[index]]
            assert region and -1 not in region
            polygon = voronoi.vertices[region]
            polygon = polygon[np.argsort(np.arctan2(polygon[:, 1] - center[1], polygon[:, 0] - center[0]))]
            apothem = radius * np.sqrt(3) / 2
            if mode == "center_inside":
                keep = np.all(center @ _NORMALS.T <= apothem + 1e-9)
            elif mode == "full_containment":
                keep = np.all(polygon @ _NORMALS.T <= apothem + 1e-9)
            else:
                keep = _polygon_area(_clip_hex(polygon, apothem)) > 1e-10
            if keep:
                result.add((*lattice_keys[index], phase))
    return result


class Hex111Tests(unittest.TestCase):
    def make(self, **overrides):
        config = dict(a0=4.0, radius_A=6.0, vacuum=5.0)
        config.update(overrides)
        return build_hex_111(**config)

    def test_all_rules_match_qhull_and_enlarged_mother(self):
        for radius in (6.0, 6.8, 8.0):
            for mode in SELECTION_MODES:
                with self.subTest(radius=radius, mode=mode):
                    atoms = self.make(radius_A=radius, selection_mode=mode)
                    actual = {(i, j, k) for i, j, k in keys(atoms) if k < 3}
                    self.assertEqual(actual, independent_selection(radius, mode, 8))
                    self.assertEqual(actual, independent_selection(radius, mode, 14))

    def test_analytic_ws_vertices_match_qhull(self):
        d = 4 / np.sqrt(2)
        xy = np.array([[i*d + j*d/2, j*d*np.sqrt(3)/2] for i in range(-4, 5) for j in range(-4, 5)])
        vor = Voronoi(xy)
        index = np.argmin(np.linalg.norm(xy, axis=1))
        polygon = vor.vertices[vor.regions[vor.point_region[index]]]
        expected = _UNIT_VERTICES * d / np.sqrt(3)
        self.assertEqual(len(polygon), 6)
        self.assertLess(cKDTree(expected).query(polygon)[0].max(), 1e-12)

    def test_clipping_closed_boundary_and_positive_area(self):
        apothem = np.sqrt(3) / 2
        self.assertAlmostEqual(_polygon_area(_clip_hex(_UNIT_VERTICES, apothem)), 3*np.sqrt(3)/2)
        self.assertLess(_polygon_area(_clip_hex(_UNIT_VERTICES + [2*apothem, 0], apothem)), 1e-10)
        self.assertEqual(_polygon_area(_clip_hex(_UNIT_VERTICES + [4, 0], apothem)), 0)
        self.assertGreater(_polygon_area(_clip_hex(_UNIT_VERTICES + [2*apothem - .1, 0], apothem)), 0)
        self.assertAlmostEqual(_polygon_area(_clip_hex(_UNIT_VERTICES*.2, apothem)), .04*3*np.sqrt(3)/2)

    def test_selection_set_inclusions_and_known_generic_counts(self):
        generated = {m: self.make(selection_mode=m) for m in SELECTION_MODES}
        self.assertEqual([len(generated[m]) for m in SELECTION_MODES], [88, 140, 48])
        contained, centers, intersected = (set(keys(generated[m])) for m in
                                           ("full_containment", "center_inside", "positive_area_intersection"))
        self.assertTrue(contained < centers < intersected)

    def test_deterministic_ids_coordinates_and_shared_site_identity(self):
        small, large = self.make(), self.make(radius_A=8.0)
        repeat = self.make()
        np.testing.assert_array_equal(small.positions, repeat.positions)
        np.testing.assert_array_equal(small.arrays["generated_site_id"], repeat.arrays["generated_site_id"])
        self.assertEqual(small.info, repeat.info)
        self.assertEqual(len(set(small.arrays["generated_site_id"])), len(small))
        large_map = dict(zip(keys(large), large.arrays["generated_site_id"].tolist()))
        for key, sid in zip(keys(small), small.arrays["generated_site_id"]):
            self.assertEqual(sid, large_map[key])
        ids = [_site_id(i, j, k) for i in range(-8, 9) for j in range(-8, 9) for k in range(6)]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn("parent_site_id", small.arrays)

    def test_fixed_target_not_selected_centroid_and_scale_covariance(self):
        atoms = self.make()
        centered = atoms.positions[:, :2] - atoms.info["target_center_xy_A"]
        d = 4 / np.sqrt(2)
        expected = np.array([[-d/2 + i*d + j*d/2, j*d*np.sqrt(3)/2 + (k%3)*4/np.sqrt(6)]
                             for i, j, k in keys(atoms)])
        np.testing.assert_allclose(centered, expected, atol=2e-15, rtol=0)
        scaled = self.make(a0=8.0, radius_A=12.0, vacuum=10.0,
                           length_tolerance_A=2e-9, area_tolerance_A2=4e-10)
        self.assertEqual(keys(atoms), keys(scaled))
        np.testing.assert_array_equal(atoms.positions*2, scaled.positions)
        np.testing.assert_array_equal(atoms.cell.array*2, scaled.cell.array)
        self.assertEqual(atoms.info["hex_vertex_angle_deg"], 30.0)
        self.assertEqual(atoms.info["ws_method"], "analytic_infinite_triangular_lattice_2d")

    def test_axial_abc_repeat_neighbours_connectivity_and_vacuum(self):
        for mode in SELECTION_MODES:
            atoms = self.make(selection_mode=mode)
            p, lz = atoms.positions, atoms.cell[2, 2]
            delta = p[:, None, :] - p[None, :, :]
            delta[:, :, 2] -= np.rint(delta[:, :, 2]/lz)*lz
            distance = np.linalg.norm(delta, axis=2)
            np.fill_diagonal(distance, np.inf)
            self.assertAlmostEqual(distance.min(), 4/np.sqrt(2))
            np.testing.assert_allclose(distance.min(axis=1), 4/np.sqrt(2), atol=1e-12)
            adjacency = abs(distance - 4/np.sqrt(2)) < 1e-10
            visited, frontier = {0}, [0]
            while frontier:
                for node in np.flatnonzero(adjacency[frontier.pop()]):
                    if node not in visited:
                        visited.add(node)
                        frontier.append(node)
            self.assertEqual(len(visited), len(atoms))
            self.assertTrue(np.any(adjacency & (abs(p[:, None, 2] - p[None, :, 2]) > lz/2)))
            self.assertAlmostEqual(lz, 6*4/np.sqrt(3))
            self.assertTrue(all((i, j, (k+3)%6) in set(keys(atoms)) for i, j, k in keys(atoms)))
            full = atoms.get_all_distances(mic=True)
            np.fill_diagonal(full, np.inf)
            np.testing.assert_array_equal(abs(full - 4/np.sqrt(2)) < 1e-10, adjacency)
            self.assertTrue(atoms.pbc.all())

    def test_repeats_are_explicit_complete_abc_periods(self):
        atoms = self.make(abc_repeats=3)
        self.assertEqual(len(atoms), len(self.make())*3//2)
        self.assertAlmostEqual(atoms.cell[2, 2], 9*4/np.sqrt(3))

    def test_invalid_inputs_fail_without_generating(self):
        for field in ("a0", "radius_A", "vacuum"):
            for value in (0, -1, float("nan"), float("inf")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.make(**{field: value})
        for field in ("length_tolerance_A", "area_tolerance_A2"):
            for value in (-1, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    self.make(**{field: value})
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                self.make(abc_repeats=value)
        with self.assertRaises(ValueError):
            self.make(selection_mode="intersection")
        with self.assertRaises(ValueError):
            self.make(radius_A=.001, selection_mode="full_containment")
        with self.assertRaises(ValueError):
            self.make(selection_mode="positive_area_intersection", vacuum=.001)

    def test_legacy_circle_mask_and_wrapper_unchanged(self):
        points = np.array([[0, 0, 2], [3, 4, -2], [5+1e-9, 0, 1], [4, 4, 0]], float)
        np.testing.assert_array_equal(_cross_section_mask(points, center_xy=(0, 0), radius_A=5, shape="circle"),
                                      [True, True, False, False])
        for orientation in ("111", "100", "110"):
            kwargs = dict(a0=4., diameter_nm=None, radius_A=6., length_z=8., vacuum=5., orientation=orientation)
            a, b = build_circular_nanowire(**kwargs), build_periodic_prism(**kwargs, cross_section_shape="circle")
            np.testing.assert_array_equal(a.positions, b.positions)
            np.testing.assert_array_equal(a.cell.array, b.cell.array)
            self.assertEqual(a.info, b.info)
            self.assertNotIn("generated_site_id", a.arrays)

    def test_cli_rejects_ignored_legacy_geometry_options(self):
        base = [sys.executable, str(ROOT / "app/ase_nanocrystal.py"), "--builder", "hex_111"]
        for option in (["--length-z", "100"], ["--length=100"],
                       ["--cross-section-shape", "circle"], ["--shape-rotation-deg", "0"],
                       ["--size", "4"], ["--gamma100", "1"], ["--gamma110", "1"],
                       ["--orientation", "100"]):
            with self.subTest(option=option):
                run = subprocess.run(base + option, capture_output=True, text=True)
                self.assertNotEqual(run.returncode, 0)

    def test_cli_extxyz_and_exact_json_are_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "column.extxyz"
            command = [sys.executable, str(ROOT / "app/ase_nanocrystal.py"), "--builder", "hex_111",
                       "--a0", "4", "--radius-A", "6", "--vacuum", "5", "--output", str(output)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            record_path = Path(str(output) + ".json")
            first = (output.read_bytes(), record_path.read_bytes())
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(first, (output.read_bytes(), record_path.read_bytes()))
            record = json.loads(record_path.read_text())
            atoms = self.make()
            np.testing.assert_array_equal(record["positions_A"], atoms.positions)
            np.testing.assert_array_equal(record["cell_A"], atoms.cell.array)
            reread = read(output)
            np.testing.assert_array_equal(reread.arrays["generated_site_id"], atoms.arrays["generated_site_id"])
            np.testing.assert_allclose(reread.positions, atoms.positions, atol=5.1e-9, rtol=0)
            self.assertEqual(reread.info["selection_mode"], "center_inside")
            self.assertEqual(reread.info["site_id_namespace"], "hex111_v1")


if __name__ == "__main__":
    unittest.main()
