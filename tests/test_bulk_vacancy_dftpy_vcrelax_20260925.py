"""Manifest, cell/SCF gates and failure persistence, without real DFT jobs."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bulk_vc_runner", REPO / "scripts/run_bulk_vacancy_dftpy_vcrelax_20260925.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class VariableCellTests(unittest.TestCase):
    def test_all_real_inputs_dry_check_no_dft_and_manifest_cell(self):
        from dftpy.api.api4ase import DFTpyCalculator
        with patch.object(DFTpyCalculator, "calculate", side_effect=AssertionError("No real DFT")):
            for case in runner.CASE_IDS:
                checked, atoms, sources = runner.preflight(runner.CAMPAIGN, case)
                protocol = checked["protocol"]
                pp = runner.contained_path(runner.CAMPAIGN, protocol["pseudo"]["file"])
                config = runner.make_config(atoms, pp, protocol)
                np.testing.assert_allclose(atoms.cell.lengths(), [12.119544] * 3)
                self.assertEqual(config["GRID"]["nr"], protocol["grid"])
                self.assertEqual(config["EXC"]["libxc"], ["lda_x", "lda_c_pz"])
                self.assertEqual(config["KEDF"]["x"], protocol["lambda_tf"])
                self.assertIn("dftpy_manifest.json", sources)

    def test_dry_check_does_not_write_attempt(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / "untouched"
            self.assertEqual(runner.main(["--case", "1V", "--dry-check", "--threads", "1",
                                          "--max-runtime-s", "120", "--output-root", str(output)]), 0)
            self.assertFalse(output.exists())

    def test_hash_tampering_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory) / "campaign"
            # Copy bytes, not the immutable deployed source's 0444 permissions.
            shutil.copytree(runner.CAMPAIGN, campaign, copy_function=shutil.copyfile)
            with (campaign / "geometry.json").open("a") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "Geometry hash"):
                runner.preflight(campaign, "1V")
        with self.assertRaises(ValueError):
            runner.contained_path(runner.CAMPAIGN, "../escape")
        with self.assertRaises(ValueError):
            runner.validate_output_root(REPO / "production")

    def test_scf_gate_rejects_flag_density_grid(self):
        checked, _, _ = runner.preflight(runner.CAMPAIGN, "1V")
        event = {"grid_shape": [78] * 3, "density_integral_electrons": 321.,
                 "converged_flag": 0, "energy_Ha": -10.}
        runner.validate_electronic_event(event, 107, checked["protocol"])
        for change in ({"converged_flag": 1}, {"converged_flag": 2},
                       {"grid_shape": [80] * 3}, {"density_integral_electrons": 320.},
                       {"energy_Ha": math.nan}):
            with self.assertRaises(RuntimeError):
                runner.validate_electronic_event({**event, **change}, 107, checked["protocol"])

    def test_three_independent_structural_gates(self):
        manifest = runner.preflight(runner.CAMPAIGN, "1V")[0]["protocol"]
        good = {"fmax_eV_A": .0009, "filter_fmax_eV_A": .0009, "stress_max_abs_GPa": .04}
        events = [{"accepted": True}]
        self.assertTrue(runner.acceptance(True, good, events, manifest))
        self.assertFalse(runner.acceptance(False, good, events, manifest))
        for key, bad in (("fmax_eV_A", .0011), ("filter_fmax_eV_A", .0011),
                         ("stress_max_abs_GPa", .051)):
            self.assertFalse(runner.acceptance(True, {**good, key: bad}, events, manifest))
        self.assertFalse(runner.acceptance(True, good, [{"accepted": False}], manifest))

    def test_budget_guard_really_interrupts_and_restores_handler(self):
        import signal
        original = signal.getsignal(signal.SIGALRM)
        with self.assertRaises(runner.BudgetExpired), runner.budget_guard(.01):
            time.sleep(.04)
        self.assertEqual(signal.getsignal(signal.SIGALRM), original)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)

    def mock_execute(self, directory, *, flag=0, stop_second=False, bad_lattice=False, case="1V"):
        from ase.calculators.calculator import Calculator, all_changes
        from ase.units import Bohr, Hartree
        from dftpy.optimization.optimization import Optimization

        checked, atoms, sources = runner.preflight(runner.CAMPAIGN, case)
        manifest = checked["protocol"]
        config = runner.make_config(atoms, runner.contained_path(runner.CAMPAIGN, manifest["pseudo"]["file"]), manifest)
        n_atoms = len(atoms)
        target = atoms.cell.lengths().copy()
        target[0] *= .99  # Full-cell test: x relaxes while y/z stay unchanged.
        initial_volume = atoms.get_volume()
        calls = {"n": 0}

        class FakeDensity:
            def __init__(self, cell):
                lattice = np.asarray(cell) / Bohr
                if bad_lattice:
                    lattice = lattice * 2
                self.grid = SimpleNamespace(nr=np.array(manifest["grid"]), lattice=lattice, origin=np.zeros(3))

            def integral(self):
                return 3. * n_atoms

            def __array__(self, dtype=None, copy=None):
                return np.full(tuple(manifest["grid"]), 3 * n_atoms / np.linalg.det(self.grid.lattice), dtype=dtype)

        def fake_optimize(opt, *args, **kwargs):
            calls["n"] += 1
            if stop_second and calls["n"] == 2:
                raise runner.BudgetExpired("simulated budget exhaustion after one checkpoint")
            opt.converged = flag
            opt.optimization_options = {"econv": n_atoms * 1e-9}
            opt.functional = SimpleNamespace(energy=opt.fake_energy_Ha)
            return opt.fake_density

        class FakeCalculator(Calculator):
            implemented_properties = ["energy", "forces", "stress"]

            def __init__(self, config):
                super().__init__()
                self.dftpy_results = {}

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                strain = atoms.cell.lengths() / target - 1
                energy = -10 * Hartree + .5 * initial_volume * np.sum(strain**2)
                sigma = initial_volume / atoms.get_volume() * strain * (1 + strain)
                density = FakeDensity(atoms.cell.array)
                Optimization.optimize_rho(SimpleNamespace(fake_energy_Ha=energy / Hartree, fake_density=density))
                self.results = {"energy": energy, "forces": np.zeros((len(atoms), 3)),
                                "stress": np.r_[sigma, [0., 0., 0.]]}
                self.dftpy_results = {"density": density, "energypotential": {
                    "TOTAL": SimpleNamespace(energy=energy / Hartree),
                    "KEDF-TF": SimpleNamespace(energy=2.), "KEDF-VW": SimpleNamespace(energy=.5)}}

        args = SimpleNamespace(output_root=Path(directory) / "output", attempt="attempt_001", case=case,
                               campaign=runner.CAMPAIGN, max_runtime_s=60)
        for key in runner.THREAD_KEYS:
            runner.os.environ[key] = "1"
        with patch("dftpy.api.api4ase.DFTpyCalculator", FakeCalculator), \
                patch.object(Optimization, "optimize_rho", fake_optimize), contextlib.redirect_stdout(io.StringIO()):
            code = runner.run(args, checked, atoms, sources, config)
        attempt = args.output_root / case / args.attempt
        return code, args, json.loads((attempt / "result.json").read_text()), attempt

    def test_mock_full_cell_success_preserves_cell_history_and_units(self):
        from ase.io.trajectory import Trajectory
        with tempfile.TemporaryDirectory() as directory:
            code, args, result, attempt = self.mock_execute(directory)
            self.assertEqual(code, 0)
            self.assertTrue(result["accepted"])
            self.assertEqual(result["status"], "converged")
            self.assertGreater(result["ionic_steps"], 0)
            self.assertLess(result["cell_A"][0][0], result["initial_cell_A"][0][0])
            self.assertAlmostEqual(result["cell_A"][1][1], result["initial_cell_A"][1][1])
            self.assertAlmostEqual(result["energy_eV"], result["energy_components_eV"]["TOTAL"])
            events = [json.loads(line) for line in (attempt / "ionic_evaluations.jsonl").read_text().splitlines()]
            self.assertEqual(len(events), result["ionic_steps"] + 1)
            with Trajectory(attempt / "relax.traj") as trajectory:
                self.assertEqual(len(trajectory), len(events))
                np.testing.assert_allclose(trajectory[-1].cell.array, result["cell_A"])
            for relative, expected in result["artifact_sha256"].items():
                self.assertEqual(runner.sha256(attempt / relative), expected)
            with np.load(attempt / "checkpoint_bundle.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["parent_site_id"].size, 107)
                self.assertEqual(saved["grid"].tolist(), [78] * 3)
            checked, atoms, sources = runner.preflight(runner.CAMPAIGN, "1V")
            with self.assertRaisesRegex(FileExistsError, "Already converged"):
                runner.run(args, checked, atoms, sources, {})

    def test_mock_electronic_failure_never_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            code, _, result, _ = self.mock_execute(directory, flag=2)
            self.assertEqual(code, 1)
            self.assertFalse(result["accepted"])
            self.assertEqual(result["status"], "failed")
            self.assertNotIn("energy_eV", result)

    def test_mock_wrong_density_lattice_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, result, _ = self.mock_execute(directory, bad_lattice=True)
            self.assertFalse(result["accepted"])
            self.assertIn("Density lattice", result["error"])

    def test_mock_budget_stop_retains_coherent_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            code, _, result, attempt = self.mock_execute(directory, stop_second=True)
            self.assertEqual(code, 1)
            self.assertEqual(result["status"], "time_budget_exhausted")
            self.assertFalse(result["accepted"])
            self.assertEqual(result["last_completed_checkpoint"], "checkpoint_bundle.npz")
            with np.load(attempt / result["last_completed_checkpoint"], allow_pickle=False) as saved:
                self.assertEqual(json.loads(str(saved["metadata_json"]))["ionic_step"], 0)
                self.assertEqual(saved["positions_A"].shape, (107, 3))

    def test_locks_are_case_local(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            held_dir = Path(directory) / "output/1V"
            held_dir.mkdir(parents=True)
            with (held_dir / ".execution.lock").open("a") as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    self.mock_execute(directory, case="1V")
                code, _, result, _ = self.mock_execute(directory, case="0V")
                self.assertEqual(code, 0)
                self.assertTrue(result["accepted"])


if __name__ == "__main__":
    unittest.main()
