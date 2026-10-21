"""Scientific input/convergence gates; no production DFT calculation is run."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bulk_vacancy_runner", REPO / "scripts/run_bulk_vacancy_dftpy_20260924.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerGateTests(unittest.TestCase):
    def test_all_eight_ideal_structures_and_config_without_calculating(self):
        from dftpy.api.api4ase import DFTpyCalculator
        with patch.object(DFTpyCalculator, "calculate", side_effect=AssertionError("No DFT in preflight")):
            for case in runner.CASE_IDS:
                report, atoms, _, _ = runner.preflight(runner.CAMPAIGN, case, REPO / "al.lda.recpot")
                config = runner.make_config(atoms, REPO / "al.lda.recpot")
                self.assertEqual(config["GRID"]["nr"], [78] * 3)
                self.assertEqual(config["KEDF"]["x"], 0.9)
                self.assertEqual(config["KEDF"]["y"], 0.1)
                self.assertEqual(report["expected_electrons"], 3 * len(atoms))

    def test_dry_check_cannot_create_requested_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "should-not-exist"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(["--case", "1V", "--dry-check",
                                               "--output-root", str(output)]), 0)
            self.assertFalse(output.exists())

    def test_reject_modified_structure_before_calculation(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory) / "campaign"
            shutil.copytree(runner.CAMPAIGN, campaign)
            geometry = json.loads((campaign / "geometry.json").read_text())
            structure = campaign / geometry["cases"]["1V"]["structure_files"]["extxyz"]
            structure.write_text(structure.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                runner.preflight(campaign, "1V", REPO / "al.lda.recpot")

    def test_reject_nonideal_geometry_even_if_hash_is_updated(self):
        from ase.io import read, write
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory) / "campaign"
            shutil.copytree(runner.CAMPAIGN, campaign)
            geometry = json.loads((campaign / "geometry.json").read_text())
            case = geometry["cases"]["1V"]
            structure = campaign / case["structure_files"]["extxyz"]
            atoms = read(structure)
            atoms.positions[0, 0] += 0.1
            write(structure, atoms, format="extxyz")
            case["structure_sha256"]["extxyz"] = runner.sha256(structure)
            case["positions_A"] = atoms.positions.tolist()
            (campaign / "geometry.json").write_text(json.dumps(geometry))
            with self.assertRaisesRegex(ValueError, "not the ideal parent"):
                runner.preflight(campaign, "1V", REPO / "al.lda.recpot")

    def test_reject_changed_pseudopotential(self):
        with tempfile.TemporaryDirectory() as directory:
            pseudo = Path(directory) / "al.lda.recpot"
            pseudo.write_text("wrong pseudo")
            with self.assertRaisesRegex(ValueError, "pseudopotential"):
                runner.preflight(runner.CAMPAIGN, "1V", pseudo)

    def test_reject_inside_git_and_path_traversal(self):
        with self.assertRaises(ValueError):
            runner.validate_output_root(REPO / "results")
        with self.assertRaises(ValueError):
            runner.contained_path(runner.CAMPAIGN, "../outside")
        with self.assertRaises(ValueError):
            runner.safe_token("../attempt")

    def test_electronic_gate_requires_true_flag_grid_and_electron_count(self):
        event = {"converged_flag": 0, "grid_shape": [78] * 3,
                 "density_integral_electrons": 318.0, "energy_Ha": -10.0}
        runner.validate_electronic_event(event, 106)
        for change in ({"converged_flag": 1}, {"converged_flag": 2},
                       {"grid_shape": [72] * 3}, {"energy_Ha": math.nan},
                       {"density_integral_electrons": 317.0}):
            with self.assertRaises(RuntimeError):
                runner.validate_electronic_event({**event, **change}, 106)

    def test_kinetic_components_do_not_include_total_twice(self):
        calc = SimpleNamespace(dftpy_results={"energypotential": {
            "TOTAL": SimpleNamespace(energy=-10.0), "KEDF-TF": SimpleNamespace(energy=2.0),
            "KEDF-VW": SimpleNamespace(energy=0.5), "HARTREE": SimpleNamespace(energy=1.0)}})
        terms = runner.energy_components(calc)
        self.assertAlmostEqual(terms["KEDF"], 2.5 * runner.HA_TO_EV)

    def mock_execute(self, directory, converged_flag):
        """Exercise all output/acceptance logic using a zero-force fake calculator."""
        import numpy as np
        from ase.calculators.calculator import Calculator, all_changes
        from dftpy.optimization.optimization import Optimization

        checked, atoms, files, _ = runner.preflight(runner.CAMPAIGN, "1V", REPO / "al.lda.recpot")
        config = runner.make_config(atoms, REPO / "al.lda.recpot")
        fake_grid = SimpleNamespace(nr=np.array([78] * 3), lattice=atoms.cell.array,
                                    origin=np.zeros(3))

        class FakeDensity:
            grid = fake_grid

            def integral(self):
                return 321.0

            def __array__(self, dtype=None, copy=None):
                return np.zeros((78, 78, 78), dtype=dtype)

        density = FakeDensity()

        def fake_optimize(opt, *args, **kwargs):
            opt.converged = converged_flag
            opt.optimization_options = {"econv": 107e-9}
            opt.functional = SimpleNamespace(energy=-10.0)
            return density

        class FakeCalculator(Calculator):
            implemented_properties = ["energy", "forces", "stress"]

            def __init__(self, config):
                super().__init__()
                self.dftpy_results = {}

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                Optimization.optimize_rho(SimpleNamespace())
                self.results = {"energy": -10 * runner.HA_TO_EV,
                                "forces": np.zeros((len(atoms), 3)), "stress": np.zeros(6)}
                self.dftpy_results = {"density": density, "energypotential": {
                    "TOTAL": SimpleNamespace(energy=-10.0),
                    "KEDF-TF": SimpleNamespace(energy=2.0), "KEDF-VW": SimpleNamespace(energy=0.5)}}

        args = SimpleNamespace(output_root=Path(directory) / "output", attempt="attempt_001",
                               case="1V", campaign=runner.CAMPAIGN, pp=REPO / "al.lda.recpot")
        for key in runner.THREAD_KEYS:
            runner.os.environ[key] = "1"
        with patch("dftpy.api.api4ase.DFTpyCalculator", FakeCalculator), \
                patch.object(Optimization, "optimize_rho", fake_optimize), \
                contextlib.redirect_stdout(io.StringIO()):
            code = runner.run(args, checked, atoms, files, config)
        return code, args, checked, atoms, files, config

    def test_mock_success_writes_outputs_and_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            code, args, checked, atoms, files, config = self.mock_execute(directory, 0)
            attempt = args.output_root / "1V/attempt_001"
            result = json.loads((attempt / "result.json").read_text())
            self.assertEqual(code, 0)
            self.assertTrue(result["accepted"])
            self.assertEqual(result["status"], "converged")
            self.assertTrue(result["all_electronic_calls_converged"])
            for name in ("initial.traj", "relax.traj", "final.traj", "final.extxyz",
                         "final_density.npz", "config.json", "provenance.json"):
                self.assertTrue((attempt / name).is_file(), name)
            args.attempt = "attempt_002"
            with self.assertRaisesRegex(FileExistsError, "Already converged"):
                runner.run(args, checked, atoms, files, config)

    def test_mock_electronic_failure_is_never_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            code, args, *_ = self.mock_execute(directory, 2)
            attempt = args.output_root / "1V/attempt_001"
            result = json.loads((attempt / "result.json").read_text())
            self.assertEqual(code, 1)
            self.assertFalse(result["accepted"])
            self.assertEqual(result["status"], "failed")
            self.assertNotIn("energy_eV", result)
            events = [json.loads(line) for line in (attempt / "electronic_evaluations.jsonl").read_text().splitlines()]
            self.assertFalse(events[-1]["accepted"])


if __name__ == "__main__":
    unittest.main()
