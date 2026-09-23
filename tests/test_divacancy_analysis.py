"""Synthetic failure cases plus read-only validation of the corrected source."""
import copy
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from divacancy_analysis_checks import qualify_case, compatible_comparison, optimizer_last_record
from collect_dftpy_conventional_vacancy import collect_scan, add_deltas, series_key
from analyze_divacancy_geometry_strain import green_lagrange_strain, minimum_image_vectors
from divacancy_geometry import build_centered_pristine, enumerate_pairs


def save_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def synthetic_case(base, vacancies=2):
    case = base / ("pair_scan" if vacancies == 2 else "size_scan") / "synthetic"
    case.mkdir(parents=True)
    pp = base / "al.lda.recpot"
    pp.write_bytes(b"Synthetic pseudopotential fixture; not for calculations")
    pristine, first, _ = build_centered_pristine(3.95, (2, 2, 2))
    distance, second, _ = enumerate_pairs(pristine, first)[0]
    defect = pristine.copy()
    del defect[sorted([first, second] if vacancies == 2 else [first], reverse=True)]
    n, nd = len(pristine), len(defect)
    manifest = dict(setting="synthetic", scan_type="pair" if vacancies == 2 else "size",
        pair_selection="fixed_direction", pair_direction_indices=[1, 1, 0], pair_direction_family="[110]",
        pair_distance_A=distance, pristine_n_atoms=n, vacancy_n_atoms=nd, vacancy_count=vacancies,
        cell_basis="conventional cubic fcc", conventional_repeat=[2, 2, 2],
        first_vacancy_index=first, second_vacancy_index=second, spacing_A=.2, ecut_analogue_eV=940.,
        kedf="TFVW", kedf_x=.9, kedf_y=.1, xc="LDA", fmax_eV_per_A=.005)
    result = dict(manifest, relaxation_mode="full_atom_and_cell_relaxation_vc_relax_equivalent",
        target_pressure_GPa=0., pristine_energy_eV=-100., vacancy_energy_eV=-90.,
        vacancy_formation_energy_eV=-90. - nd / n * -100.,
        pristine_final_fmax_eV_A=1e-9, vacancy_final_fmax_eV_A=1e-9)
    save_json(case / "point_manifest.json", manifest)
    save_json(case / "result.json", result)
    write(case / "pristine_raw.vasp", pristine, direct=True, vasp5=True)
    label = "divacancy" if vacancies == 2 else "vacancy"
    write(case / f"{label}_start.vasp", defect, direct=True, vasp5=True)
    calculator = {"PATH": {"pppath": str(base)}, "PP": {"Al": pp.name}, "GRID": {"spacing": .2},
                  "EXC": {"xc": "LDA"}, "KEDF": {"kedf": "TFVW", "x": .9, "y": .1}}
    for name, atoms, energy in (("pristine", pristine, -100.), (label, defect, -90.)):
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=np.zeros((len(atoms), 3)), stress=np.zeros(6))
        write(case / f"{name}_relax.traj", atoms)
        write(case / f"{name}_vc_relaxed.vasp", atoms, direct=True, vasp5=True)
        (case / f"{name}_relax.log").write_text(f"LBFGS:   12 11:22:33 {energy:.6f} 0.004000\n")
        (case / f"{name}_dftpy.out").write_text(f"total energy (eV) : {energy}\n")
        save_json(case / f"dftpy_{name}_calculator_config.json", {
            "dftpy_calculator": calculator, "ase_full_relaxation": {"cell_filter": "FrechetCellFilter",
            "fmax_eV_A": .005, "scalar_pressure_GPa": 0}})
    return case


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.case = synthetic_case(self.base)

    def tearDown(self):
        self.temporary.cleanup()

    def edit_result(self, **changes):
        path = self.case / "result.json"
        result = json.loads(path.read_text())
        result.update(changes)
        save_json(path, result)

    def test_full_evidence_qualifies_but_not_thesis_acceptance(self):
        checked = qualify_case(self.case)
        self.assertTrue(checked["qualified"], checked)
        self.assertEqual(checked["thesis_acceptance"], "not_assessed")
        self.assertEqual(checked["electronic_convergence_status"], "not_independently_verified")

    def test_atomic_force_cannot_hide_unconverged_combined_cell_force(self):
        (self.case / "divacancy_relax.log").write_text("LBFGS: 99 11:22:33 -90.000000 0.020000\n")
        result = qualify_case(self.case)
        self.assertEqual(result["status"], "unconverged")
        self.assertIn("combined fmax", result["qualification_reasons"])

    def test_pristine_force_is_also_required(self):
        (self.case / "pristine_relax.log").write_text("BFGS: 99 11:22:33 -100.000000 0.020000\n")
        self.assertEqual(qualify_case(self.case)["status"], "unconverged")

    def test_exact_optimizer_metadata_cannot_be_overridden_by_rounded_log(self):
        for converged, force in ((False, .004), (True, .005000001)):
            self.edit_result(defect_relaxation_evidence={
                'optimizer_converged': converged, 'combined_filter_fmax_eV_A': force,
                'target_fmax_eV_A': .005})
            self.assertEqual(qualify_case(self.case)['status'], 'unconverged')

    def test_missing_result_and_missing_log_do_not_qualify(self):
        (self.case / "pristine_relax.log").unlink()
        self.assertEqual(qualify_case(self.case)["status"], "missing")
        (self.case / "result.json").unlink()
        self.assertEqual(qualify_case(self.case)["status"], "missing")

    def test_wrong_reference_formula_and_nan_fail(self):
        self.edit_result(vacancy_formation_energy_eV=42.)
        self.assertEqual(qualify_case(self.case)["status"], "failed")
        self.edit_result(vacancy_energy_eV=float("nan"))
        self.assertEqual(qualify_case(self.case)["status"], "failed")

    def test_stale_optimizer_energy_fails(self):
        (self.case / "divacancy_relax.log").write_text("MDMin: 9 11:22:33 -91.000000 0.001000\n")
        self.assertEqual(qualify_case(self.case)["status"], "failed")

    def test_wrong_claimed_direction_fails_even_with_converged_energy(self):
        path = self.case / "point_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["pair_direction_indices"] = [3, 1, 0]
        save_json(path, manifest)
        self.assertEqual(qualify_case(self.case)["status"], "failed")

    def test_reference_calculator_mismatch_fails(self):
        path = self.case / "dftpy_divacancy_calculator_config.json"
        config = json.loads(path.read_text())
        config["dftpy_calculator"]["GRID"]["spacing"] = .250343
        save_json(path, config)
        self.assertEqual(qualify_case(self.case)["status"], "failed")

    def test_single_vacancy_collector_remains_supported(self):
        single = synthetic_case(self.base / "single", vacancies=1)
        self.assertTrue(qualify_case(single)["qualified"])
        rows = collect_scan(self.base / "single", "size_scan")
        self.assertEqual(rows[0]["vacancy_count"], 1)
        self.assertTrue(rows[0]["done"])

    def test_incomplete_manifest_is_reported_not_crashed(self):
        save_json(self.case / "point_manifest.json", {"setting": "synthetic"})
        rows = collect_scan(self.base, "pair_scan")
        self.assertEqual(rows[0]["status"], "failed")

    def test_mixed_direction_and_parameter_series_have_no_cross_deltas(self):
        row = collect_scan(self.base, "pair_scan")[0]
        second = dict(row, pair_direction_verified="[310]", pair_distance_A=5., Ef_vac_eV=1.)
        third = dict(row, pair_distance_A=6., Ef_vac_eV=2.)
        fourth = dict(row, pair_distance_A=7., Ef_vac_eV=3., spacing_A=.25)
        rows = [row, second, third, fourth]
        add_deltas(rows, "pair_distance_A")
        self.assertTrue(math.isnan(second["delta_from_previous_eV"]))
        self.assertTrue(math.isnan(fourth["delta_from_previous_eV"]))
        self.assertAlmostEqual(third["delta_from_previous_eV"], abs(2. - row["Ef_vac_eV"]))
        self.assertNotEqual(series_key(row, "pair_distance_A"), series_key(second, "pair_distance_A"))

    def test_qe_distance_only_join_is_rejected_and_missing_qe_is_explicit(self):
        rows = collect_scan(self.base, "pair_scan")
        comparison, reasons = compatible_comparison(rows, [{"setting": "oldQE", "pair_distance_A": rows[0]["pair_distance_A"], "Ef_vac_eV": .1}])
        self.assertFalse(comparison)
        self.assertIn("missing comparison evidence", reasons[0])
        comparison, reasons = compatible_comparison(rows, [])
        self.assertFalse(comparison)
        self.assertIn("unavailable", reasons[0])

    def test_green_strain_uses_material_row_vector_convention(self):
        initial = np.eye(3) * 3.
        f = np.array([[1., .3, 0.], [0., 1., 0.], [0., 0., 1.]])
        np.testing.assert_allclose(green_lagrange_strain(initial, initial @ f.T), .5 * (f.T @ f - np.eye(3)), atol=1e-14)

    def test_minimum_image_handles_skewed_cell(self):
        cell = np.array([[2., 0., 0.], [1.8, 1., 0.], [0., 0., 2.]])
        frac = np.array([.49, .49, 0.])
        computed = np.linalg.norm(minimum_image_vectors(frac, cell))
        brute = min(np.linalg.norm((frac - np.array([i, j, k])) @ cell)
                    for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1))
        self.assertAlmostEqual(computed, brute)


class CorrectedSourceReadOnlyTests(unittest.TestCase):
    def test_august31_three_points(self):
        root = Path(os.environ.get("AL_DEFECTS_DIVACANCY_ROOT", "/mnt/c/OFDFT/DFTPY_DIVACANCY_D110_L0p9_M0p1_RERUN_20260831"))
        if not root.is_dir():
            self.skipTest("Corrected 20260831 evidence package not installed")
        rows = collect_scan(root, "pair_scan")
        self.assertEqual(len(rows), 3)
        np.testing.assert_allclose([r["Ef_recomputed_eV"] for r in rows],
                                  [1.278389506042, 1.332160884010, 1.334938480819], atol=1e-9, rtol=0)
        for row in rows:
            self.assertTrue(row["qualified"], row["qualification_reasons"])
            self.assertEqual(row["pair_direction_verified"], "[110]")


if __name__ == "__main__":
    unittest.main()
