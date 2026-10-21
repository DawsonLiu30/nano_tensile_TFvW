"""Production is never launched: real inputs plus synthetic QE subprocess/output."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("bulk_qe", Path(__file__).parents[1] / "scripts/run_bulk_vacancy_qe_20260924.py")
qe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qe)


def output(parsed, *, ionic=True, accuracy="3.0E-11", force="0.00001000", suffix=""):
    lines = ["Program PWSCF v.7.5 starts", "!    total energy = -444.0500000000 Ry",
             "     estimated scf accuracy < " + accuracy + " Ry",
             "     convergence has been achieved in 12 iterations", "     Forces acting on atoms (cartesian axes, Ry/au):", ""]
    lines += [f"atom {i + 1:5d} type 1   force = {force} 0.00000000 0.00000000" for i in range(parsed["nat"])]
    lines += ["", "total   stress  (Ry/bohr**3)                   (kbar)     P=       0.00",
              "0.000001 0.000000 0.000000  0.1471 0.0000 0.0000",
              "0.000000 0.000001 0.000000  0.0000 0.1471 0.0000",
              "0.000000 0.000000 0.000001  0.0000 0.0000 0.1471", ""]
    if ionic:
        lines += ["bfgs converged in 8 scf cycles and 7 bfgs steps", "End of BFGS Geometry Optimization",
                  "Final energy = -444.0500000000 Ry", "Begin final coordinates", "ATOMIC_POSITIONS (angstrom)"]
        lines += ["Al " + " ".join(str(v) for v in row) for row in parsed["positions_A"]]
        lines += ["End final coordinates"]
    return "\n".join(lines + [suffix, "JOB DONE.", ""])


class BulkVacancyQETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = qe.verify_campaign(qe.DEFAULT_CAMPAIGN)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.campaign = self.root / "campaign"
        shutil.copytree(qe.DEFAULT_CAMPAIGN, self.campaign)
        self.run_root = self.root / "runs"
        self.parsed = self.original["cases"]["2V_1NN_D110"]["parsed"]

    def tearDown(self):
        self.temp.cleanup()

    def fake_run(self, *, code=0, body=None):
        fake = self.root / "fake_pw.py"
        fake.write_text("import sys\nprint(" + repr(body or output(self.parsed)) + f")\nsys.exit({code})\n", encoding="utf-8")
        argv = '"' + sys.executable.replace("\\", "/") + '" "' + str(fake).replace("\\", "/") + '"'
        return qe.run(self.campaign, self.run_root, "2V_1NN_D110", execute=True, pw=argv)

    def test_eight_real_cases_verified(self):
        self.assertEqual(set(self.original["cases"]), qe.CASE_IDS)
        self.assertEqual(self.original["cases"]["0V"]["parsed"]["nat"], 108)
        self.assertEqual(self.original["cases"]["1V"]["parsed"]["nat"], 107)

    def test_real_relax_completion_requires_ionic_electronic_and_final_data(self):
        result = qe.parse_output(output(self.parsed), self.parsed)
        self.assertTrue(result["complete"])
        self.assertTrue(result["usable"])
        self.assertTrue(result["ionic_converged"])
        self.assertTrue(result["electronic_converged"])
        self.assertEqual(len(result["forces_eV_A"]), 106)
        self.assertEqual(len(result["final_geometry"]["positions_A"]), 106)
        self.assertAlmostEqual(result["stress"]["kbar"][0][0], 0.1471)

    def test_scf_completion_and_job_done_are_not_relax_completion(self):
        result = qe.parse_output(output(self.parsed, ionic=False), self.parsed)
        self.assertTrue(result["electronic_converged"])
        self.assertFalse(result["complete"])
        self.assertFalse(result["usable"])
        self.assertEqual(result["status"], "ionic_not_converged")
        self.assertEqual(result["energy_Ry"], -444.05)
        self.assertFalse(qe.parse_output("JOB DONE.", self.parsed)["usable"])

    def test_step_time_and_scf_limits_rejected_even_with_job_done(self):
        for warning, status in (("The maximum number of steps has been reached.", "step_limit"),
                                ("Maximum CPU time exceeded", "time_limit"),
                                ("bfgs not converged in 200 steps", "step_limit")):
            result = qe.parse_output(output(self.parsed, suffix=warning), self.parsed)
            self.assertEqual(result["status"], status)
            self.assertFalse(result["usable"])
        self.assertFalse(qe.parse_output(output(self.parsed, accuracy="1.0E-5"), self.parsed)["complete"])
        self.assertFalse(qe.parse_output(output(self.parsed, suffix="convergence NOT achieved"), self.parsed)["complete"])
        self.assertEqual(qe.parse_output(output(self.parsed), self.parsed, stderr="CANCELLED DUE TO TIME LIMIT")["status"], "time_limit")

    def test_force_norm_mismatch_is_review_not_false_parser_failure(self):
        result = qe.parse_output(output(self.parsed, force="0.00005"), self.parsed)
        self.assertTrue(result["complete"])
        self.assertTrue(result["ionic_converged"])
        self.assertFalse(result["usable"])
        self.assertEqual(result["status"], "converged_needs_review")

    def test_real_1v_bfgs_failure_does_not_override_converged_scf(self):
        # Actual QE 7.5 1V stopping messages, 2026-09-25 pw.out SHA256
        # 295a481416faebfc30c0472fbd40c7d1abb2341f3dadba6b49cd770225d2b77b.
        # Surrounding geometry/forces are a small synthetic stub; no production
        # output is embedded in the repository.
        stop = """     Message from routine bfgs:
     history already reset at previous step: exiting
     Energy error            =      4.6E-08 Ry
     Gradient error          =      3.7E-05 Ry/Bohr

     bfgs failed after  18 scf cycles and  17 bfgs steps, convergence not achieved
     (criteria: energy <  1.0E-06 Ry, force <  1.9E-05 Ry/Bohr)"""
        parsed = self.original["cases"]["1V"]["parsed"]
        body = output(parsed, accuracy="3.8E-12", force="0.00003735")
        body = body.replace("convergence has been achieved in 12 iterations",
                            "convergence has been achieved in   9 iterations")
        body = body.replace("bfgs converged in 8 scf cycles and 7 bfgs steps", stop)
        body = body.replace("-444.0500000000", "-448.2953539554")
        result = qe.parse_output(body, parsed)
        self.assertTrue(result["electronic_converged"])
        self.assertFalse(result["ionic_converged"])
        self.assertTrue(result["ionic_optimizer_failed"])
        self.assertEqual(result["ionic_stop_reason"], "bfgs_trust_radius_exhausted")
        self.assertEqual(result["status"], "ionic_optimizer_failed")
        self.assertTrue(result["job_done"])
        self.assertTrue(result["force_norm_target_pass"])
        self.assertFalse(result["time_limit"])
        self.assertFalse(result["step_limit"])
        self.assertFalse(result["complete"])
        self.assertFalse(result["usable"])
        self.assertEqual(result["data_quality_reasons"], [])
        self.assertAlmostEqual(result["energy_Ry"], -448.2953539554)
        self.assertTrue(qe.parse_output(body.replace("\n", "\r\n"), parsed)["electronic_converged"])

    def test_last_standalone_scf_failure_still_overrides_prior_success(self):
        body = output(self.parsed, suffix="     convergence NOT achieved after 100 iterations: stopping")
        result = qe.parse_output(body, self.parsed)
        self.assertFalse(result["electronic_converged"])
        self.assertEqual(result["status"], "electronic_not_converged")
        self.assertFalse(result["usable"])

    def test_wrong_input_pseudo_structure_and_geometry_hashes_fail(self):
        for relative in ("qe/inputs/0V/pw.in", "pseudo/Al.pz-vbc.UPF", "structures/0V/POSCAR", "geometry.json"):
            path = self.campaign / relative
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.assertRaises(qe.CampaignError):
                qe.verify_campaign(self.campaign)
            path.write_bytes(original)

    def test_wrong_structure_even_rehashed_fails_input_geometry_check(self):
        path = self.campaign / "structures/0V/POSCAR"
        text = path.read_text(encoding="utf-8").replace("0.1666666666666667", "0.1766666666666667", 1)
        path.write_text(text, encoding="utf-8")
        manifest_path = self.campaign / "qe_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["cases"]["0V"]["structure_sha256"] = qe.sha_file(path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(qe.CampaignError):
            qe.verify_campaign(self.campaign)

    def test_operational_fields_are_replaced_and_science_is_identical(self):
        template = self.original["cases"]["2V_1NN_D110"]["input_bytes"].decode("utf-8")
        template = template.replace("&CONTROL", "&CONTROL\n wfcdir='/old/unsafe'\n wf_collect=.true.")
        template = template.replace("'from_scratch'", "'restart'").replace("startingpot = 'atomic'", "startingpot = 'file'")
        parsed = qe.parse_input(template)
        rewritten = qe.runtime_input(parsed, "al_test", 1234)
        self.assertNotIn("/old/unsafe", rewritten)
        self.assertNotIn("wf_collect", rewritten)
        self.assertIn("restart_mode = 'from_scratch'", rewritten)
        self.assertIn("startingpot = 'atomic'", rewritten)
        self.assertEqual(parsed["scientific"], qe.parse_input(rewritten)["scientific"])

    def test_refuse_implicit_execution_shell_redirect_and_git_outputs(self):
        with self.assertRaises(qe.CampaignError):
            qe.run(self.campaign, self.run_root, "0V", pw="pw.x")
        for command in (None, "pw.x > stolen.out", "pw.x -in external.in"):
            with self.assertRaises(qe.CampaignError):
                qe.command_argv(command)
        with self.assertRaises(qe.CampaignError):
            qe.run_root_checked(qe.REPO_ROOT / "runtime-test", self.campaign)

    def test_case_lock_is_exclusive(self):
        self.run_root.mkdir()
        with qe.case_lock(self.run_root, "0V"):
            with self.assertRaises(qe.CampaignError):
                with qe.case_lock(self.run_root, "0V"):
                    pass
        self.assertFalse((self.run_root / ".locks" / qe.CAMPAIGN_ID / "0V.lock").exists())

    def test_successful_fake_process_provenance_and_fresh_attempt(self):
        result = self.fake_run()
        self.assertTrue(result["usable"])
        directory = Path(result["attempt"])
        record = json.loads((directory / "attempt.json").read_text(encoding="utf-8"))
        self.assertEqual(record["source_input_sha256"], self.original["cases"]["2V_1NN_D110"]["metadata"]["sha256"])
        self.assertEqual(record["runtime_input_sha256"], qe.sha_file(directory / "pw.in"))
        self.assertTrue((directory / "trajectory.json").is_file())
        self.assertTrue((directory / "final.extxyz").is_file())
        self.assertTrue(qe.inspect_attempt(directory, self.campaign)["usable"])
        second = self.fake_run()
        self.assertNotEqual(second["attempt"], result["attempt"])
        self.assertTrue((directory / "pw.out").is_file())

    def test_failed_process_and_modified_output_cannot_be_accepted(self):
        self.assertFalse(self.fake_run(code=9)["usable"])
        result = self.fake_run()
        path = Path(result["attempt"]) / "pw.out"
        path.write_text(path.read_text(encoding="utf-8") + "modified", encoding="utf-8")
        self.assertFalse(qe.inspect_attempt(path.parent, self.campaign)["usable"])


if __name__ == "__main__":
    unittest.main()
