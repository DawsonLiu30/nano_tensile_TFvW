"""Variable-cell parser/provenance tests; no production QE is ever launched."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("vc_qe", Path(__file__).parents[1] / "scripts/run_bulk_vacancy_qe_vcrelax_20260925.py")
qe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qe)


def positions(parsed, cell, *, final=False, units="crystal"):
    lines = ["Begin final coordinates"] if final else []
    lines += ["CELL_PARAMETERS (angstrom)"]
    lines += [" ".join(str(x) for x in row) for row in cell]
    lines += ["ATOMIC_POSITIONS (" + units + ")"]
    lines += [f"Al 0.5 0.25 {0.125 + i * 0.001}" for i in range(parsed["nat"])]
    if final:
        lines += ["End final coordinates"]
    return "\n".join(lines)


def scf(parsed, *, force=0.00001, stress=None, accuracy="3.8E-12", failed=False):
    stress = stress or [[0.1, 0.02, 0.0], [0.02, 0.1, 0.0], [0.0, 0.0, 0.1]]
    pressure = sum(stress[i][i] for i in range(3)) / 3
    lines = ["! total energy = -4226.750120 Ry", "estimated scf accuracy < " + accuracy + " Ry",
             "convergence NOT achieved after 300 iterations: stopping" if failed else "convergence has been achieved in 9 iterations",
             "Forces acting on atoms (cartesian axes, Ry/au):", ""]
    lines += [f"atom {i + 1} type 1 force = {force} 0.0 0.0" for i in range(parsed["nat"])]
    lines += [f"total   stress  (Ry/bohr**3)                   (kbar)     P= {pressure:.2f}"]
    lines += [" ".join(f"{v / qe.STRESS_TO_KBAR:.12e}" for v in row) + " " + " ".join(f"{v:.4f}" for v in row) for row in stress]
    return "\n".join(lines) + "\n"


def output(parsed, *, force=0.00001, stress=None, final_scf=True, final_cell=True, failed_scf=False, bfgs_failed=False):
    cell = [[12.0, 0.15, 0.0], [0.0, 11.8, 0.0], [0.0, 0.0, 12.1]]
    final = positions(parsed, cell, final=True)
    if not final_cell:
        final = final.replace("CELL_PARAMETERS (angstrom)\n12.0 0.15 0.0\n0.0 11.8 0.0\n0.0 0.0 12.1\n", "")
    stop = ("history already reset at previous step: exiting\n"
            "bfgs failed after 18 scf cycles and 17 bfgs steps, convergence not achieved") if bfgs_failed else "bfgs converged in 8 scf cycles and 7 bfgs steps"
    body = "Program PWSCF v.7.5 starts\n" + scf(parsed, force=0.01)
    body += positions(parsed, [[11.9, 0, 0], [0, 11.9, 0], [0, 0, 11.9]]) + "\n"
    body += stop + "\nFinal enthalpy = -4226.750000 Ry\n" + final + "\n"
    if final_scf:
        body += "Final scf calculation at the relaxed structure.\n" + scf(parsed, force=force, stress=stress, failed=failed_scf)
    return body + "JOB DONE.\n"


class BulkVacancyVCQETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = qe.verify_campaign(qe.DEFAULT_CAMPAIGN)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.campaign = self.root / "campaign"
        # The deployed source is immutable (0444); tamper tests need a writable copy.
        shutil.copytree(qe.DEFAULT_CAMPAIGN, self.campaign, copy_function=shutil.copyfile)
        self.run_root = self.root / "runs"
        self.parsed = self.original["cases"]["2V_1NN_D110"]["parsed"]

    def tearDown(self):
        self.temp.cleanup()

    def fake_run(self, body=None, code=0):
        fake = self.root / "fake_pw.py"
        fake.write_text("import sys\nprint(" + repr(body or output(self.parsed)) + f")\nsys.exit({code})\n", encoding="utf-8")
        command = '"' + sys.executable.replace("\\", "/") + '" "' + str(fake).replace("\\", "/") + '"'
        return qe.run(self.campaign, self.run_root, "2V_1NN_D110", execute=True, pw=command)

    def test_actual_eight_case_manifest(self):
        self.assertEqual(set(self.original["cases"]), qe.CASE_IDS)
        self.assertEqual(self.parsed["pseudo_name"], "Al_PAW_PBE.UPF")
        self.assertAlmostEqual(qe.number(self.parsed["blocks"]["system"]["ecutwfc"]), 58.798915481)
        self.assertEqual(self.parsed["cards"]["ATOMIC_POSITIONS"]["units"], "crystal")

    def test_dynamic_cells_and_final_crystal_positions_use_final_cell(self):
        result = qe.parse_output(output(self.parsed), self.parsed)
        self.assertTrue(result["usable"])
        self.assertTrue(result["final_scf_verified"])
        self.assertEqual(len(result["trajectory"]), 3)
        self.assertEqual(result["trajectory"][1]["cell_A"][0][0], 11.9)
        xyz = result["final_geometry"]["positions_A"][0]
        for actual, expected in zip(xyz, [6.0, 3.025, 1.5125]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(result["final_volume_A3"], 12.0 * 11.8 * 12.1)
        self.assertEqual(result["energy_Ry"], -4226.75012)
        self.assertEqual(result["optimization_final_enthalpy_Ry"], -4226.75)
        self.assertAlmostEqual(result["pressure_kbar"], 0.1)

    def test_hydrostatic_pressure_is_not_enough_when_shear_fails(self):
        stress = [[0.1, 0.6, 0], [0.6, 0.1, 0], [0, 0, 0.1]]
        result = qe.parse_output(output(self.parsed, stress=stress), self.parsed)
        self.assertTrue(result["pressure_target_pass"])
        self.assertFalse(result["full_stress_target_pass"])
        self.assertFalse(result["usable"])
        self.assertEqual(result["status"], "converged_needs_review")

    def test_pressure_and_force_gates(self):
        stress = [[0.6, 0, 0], [0, 0.6, 0], [0, 0, 0.6]]
        result = qe.parse_output(output(self.parsed, stress=stress), self.parsed)
        self.assertFalse(result["pressure_target_pass"])
        self.assertFalse(result["usable"])
        result = qe.parse_output(output(self.parsed, force=0.00005), self.parsed)
        self.assertFalse(result["force_norm_target_pass"])
        self.assertFalse(result["usable"])

    def test_missing_final_cell_cannot_reuse_last_trial_cell(self):
        result = qe.parse_output(output(self.parsed, final_cell=False), self.parsed)
        self.assertIsNone(result["final_geometry"])
        self.assertFalse(result["usable"])

    def test_malformed_cell_and_unclosed_final_region_rejected(self):
        for body in (output(self.parsed).replace("12.0 0.15 0.0", "bad 0.15 0.0"),
                     output(self.parsed).replace("End final coordinates", "unfinished")):
            self.assertFalse(qe.parse_output(body, self.parsed)["usable"])

    def test_alat_and_bohr_cell_conversions(self):
        for header, factor in (("CELL_PARAMETERS (bohr)", qe.BOHR_TO_A),
                               ("CELL_PARAMETERS (alat= 2.0)", 2 * qe.BOHR_TO_A)):
            body = "Begin final coordinates\n" + header + "\n2 0 0\n0 2 0\n0 0 2\nATOMIC_POSITIONS (crystal)\n"
            body += "\n".join("Al 0.5 0.25 0.125" for _ in range(self.parsed["nat"]))
            body += "\nEnd final coordinates\n"
            frames, errors = qe.geometry_frames(body, self.parsed)
            self.assertFalse(errors)
            self.assertAlmostEqual(frames[-1]["positions_A"][0][0], factor)
            self.assertTrue(frames[-1]["final_section_complete"])

    def test_final_relaxed_cell_scf_is_required_and_must_converge(self):
        for options in ({"final_scf": False}, {"failed_scf": True}):
            result = qe.parse_output(output(self.parsed, **options), self.parsed)
            self.assertFalse(result["usable"])
            self.assertFalse(result["complete"])

    def test_bfgs_failure_is_not_electronic_failure(self):
        result = qe.parse_output(output(self.parsed, bfgs_failed=True), self.parsed)
        self.assertTrue(result["electronic_converged"])
        self.assertFalse(result["ionic_converged"])
        self.assertEqual(result["status"], "ionic_optimizer_failed")
        self.assertFalse(result["usable"])

    def test_missing_final_scf_data_never_falls_back_to_optimizer_data(self):
        body = output(self.parsed)
        before, after = body.split("Final scf calculation at the relaxed structure.", 1)
        for remove in ("estimated scf accuracy < 3.8E-12 Ry", "! total energy = -4226.750120 Ry",
                       "Forces acting on atoms", "total   stress"):
            altered = before + "Final scf calculation at the relaxed structure." + after.replace(remove, "unavailable")
            self.assertFalse(qe.parse_output(altered, self.parsed)["usable"], remove)

    def test_time_step_and_process_failure_rejected(self):
        for suffix, status in (("Maximum CPU time exceeded", "time_limit"),
                               ("The maximum number of steps has been reached.", "step_limit")):
            result = qe.parse_output(output(self.parsed) + suffix, self.parsed)
            self.assertFalse(result["usable"])
            self.assertEqual(result["status"], status)
        self.assertFalse(qe.parse_output(output(self.parsed), self.parsed, returncode=7)["usable"])

    def test_source_hashes_and_protocol_disagreement_rejected(self):
        for filename in ("qe/inputs/0V/pw.in", "pseudo/Al_PAW_PBE.UPF", "structures/0V/POSCAR", "geometry.json"):
            path = self.campaign / filename
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.assertRaises(qe.CampaignError):
                qe.verify_campaign(self.campaign)
            path.write_bytes(original)
        path = self.campaign / "qe_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["protocol"]["ecutrho_Ry"] = 240
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(qe.CampaignError):
            qe.verify_campaign(self.campaign)

    def test_differing_case_settings_fail_even_when_rehashed(self):
        path = self.campaign / "qe/inputs/1V/pw.in"
        path.write_text(path.read_text(encoding="utf-8").replace("diago_david_ndim = 2", "diago_david_ndim = 4"), encoding="utf-8")
        mpath = self.campaign / "qe_manifest.json"
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        manifest["cases"]["1V"]["sha256"] = qe.sha_file(path)
        mpath.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(qe.CampaignError):
            qe.verify_campaign(self.campaign)

    def test_runtime_is_fresh_preserves_science_and_all_cell_settings(self):
        text = self.original["cases"]["2V_1NN_D110"]["input_bytes"].decode()
        text = text.replace("'from_scratch'", "'restart'").replace("startingpot = 'atomic'", "startingpot = 'file'")
        text = text.replace("&CONTROL", "&CONTROL\n wfcdir='/old/unsafe'\n wf_collect=.true.")
        parsed = qe.parse_input(text)
        runtime = qe.runtime_input(parsed, "test", 123)
        self.assertEqual(qe.parse_input(runtime)["scientific"], parsed["scientific"])
        self.assertIn("restart_mode = 'from_scratch'", runtime)
        self.assertIn("startingpot = 'atomic'", runtime)
        self.assertIn("&CELL", runtime)
        self.assertNotIn("/old/unsafe", runtime)

    def test_execution_safety_lock_and_new_attempt_provenance(self):
        with self.assertRaises(qe.CampaignError):
            qe.run(self.campaign, self.run_root, "0V", pw="pw.x")
        with self.assertRaises(qe.CampaignError):
            qe.command_argv("pw.x > output")
        with self.assertRaises(qe.CampaignError):
            qe.run_root_checked(qe.REPO_ROOT / "test-output", self.campaign)
        with qe.case_lock(self.run_root, "0V"):
            with self.assertRaises(qe.CampaignError):
                with qe.case_lock(self.run_root, "0V"):
                    pass
        first = self.fake_run()
        second = self.fake_run()
        self.assertTrue(first["usable"])
        self.assertNotEqual(first["attempt"], second["attempt"])
        directory = Path(first["attempt"])
        record = json.loads((directory / "attempt.json").read_text(encoding="utf-8"))
        self.assertEqual(record["max_seconds"], 169200)
        self.assertEqual(qe.number(qe.parse_input((directory / "pw.in").read_text(encoding="utf-8"))["blocks"]["control"]["max_seconds"]), 169200)
        self.assertTrue((directory / "pseudo/Al_PAW_PBE.UPF").is_file())
        self.assertTrue((directory / "scratch").is_dir())
        self.assertTrue(qe.inspect_attempt(directory, self.campaign)["usable"])
        self.assertTrue((directory / "source/runner.py").is_file())
        result = json.loads((directory / "final_geometry.json").read_text())
        self.assertAlmostEqual(result["cell_A"][0][1], 0.15)
        path = directory / "pw.out"
        path.write_text(path.read_text() + "tampered")
        self.assertFalse(qe.inspect_attempt(directory, self.campaign)["usable"])

    def test_runtime_budget_accepts_47_hours_without_changing_science(self):
        verified = qe.verify_campaign(self.campaign)
        for budget in (39600, 169200):
            with self.subTest(max_seconds=budget):
                directory, record = qe.create_attempt(verified, self.run_root, "0V", ["pw.x"], max_seconds=budget)
                runtime = qe.parse_input((directory / "pw.in").read_text(encoding="utf-8"))
                self.assertEqual(qe.number(runtime["blocks"]["control"]["max_seconds"]), budget)
                self.assertEqual(record["max_seconds"], budget)
                self.assertEqual(record["scientific_sha256"], verified["cases"]["0V"]["scientific_sha256"])
                self.assertEqual(record["manifest_sha256"], verified["manifest_sha256"])
                self.assertFalse(qe.immutable_errors(directory, record))

    def test_invalid_runtime_budgets_fail_before_creating_attempt(self):
        for budget in (0, -1, 169201, float("inf"), float("nan")):
            with self.subTest(max_seconds=budget), self.assertRaises(qe.CampaignError):
                qe.create_attempt(self.original, self.run_root, "0V", ["pw.x"], max_seconds=budget)
        self.assertFalse(self.run_root.exists())


if __name__ == "__main__":
    unittest.main()
