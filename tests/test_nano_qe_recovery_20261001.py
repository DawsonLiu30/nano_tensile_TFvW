"""No DFT: verify exact physical-input preservation and reject underallocation."""
from pathlib import Path
import importlib.util,json,unittest,tempfile,shutil
R=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('recovery',R/'scripts/run_nano_qe_memory_recovery_20261001.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
C=R/'campaigns/al_nano_qe_memory_recovery_20261001'

class RecoveryTests(unittest.TestCase):
    def test_exact_original_inputs_and_four_case_scope(self):
        new=m.verify(C)
        old=json.loads((R/'campaigns/al_luder_validation_nano_20260930/qe_manifest.json').read_text())
        self.assertEqual(new['parent_manifest_sha256'],m.sha(R/'campaigns/al_luder_validation_nano_20260930/qe_manifest.json'))
        self.assertEqual(new['pseudo_sha256'],old['pseudo_sha256']);self.assertEqual(new['binary_sha256'],old['binary_sha256'])
        for c,original in zip(new['cases'],old['cases'][11:]):
            self.assertEqual(c['input_sha256'],original['input_sha256'])
            self.assertEqual((C/c['input']).read_bytes(),(R/'campaigns/al_luder_validation_nano_20260930'/original['input']).read_bytes())
            self.assertEqual(c['positions_A'],original['positions_A']);self.assertEqual(c['cell_A'],original['cell_A'])
    def test_enforced_limit_overrides_environment_claim(self):
        c=m.verify(C)['cases'][2];env={'SLURM_NTASKS':'29','SLURM_JOB_NUM_NODES':'1','SLURM_MEM_PER_NODE':str(256*1024)}
        with self.assertRaises(ValueError):m.allocation_guard(c,env,64*2**30)
        m.allocation_guard(c,env,256*2**30)
    def test_underrequested_memory_wrong_ranks_or_nodes_rejected(self):
        c=m.verify(C)['cases'][0];good={'SLURM_NTASKS':'29','SLURM_JOB_NUM_NODES':'1','SLURM_MEM_PER_NODE':str(128*1024)}
        for key,value in [('SLURM_MEM_PER_NODE',str(64*1024)),('SLURM_NTASKS','28'),('SLURM_JOB_NUM_NODES','2')]:
            env={**good,key:value}
            with self.assertRaises(ValueError):m.allocation_guard(c,env,256*2**30)
    def test_modified_physical_input_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder)/'campaign';shutil.copytree(C,d)
            c=m.verify(d)['cases'][0];p=d/c['input'];p.chmod(0o644);p.write_text(p.read_text().replace('58.798915481','50.0'))
            with self.assertRaises(ValueError):m.verify(d)
    def test_step_failure_and_oom_do_not_leave_remaining_ranks_running(self):
        cmd=m.launch_command('/test/pw.x')
        for required in ['--kill-on-bad-exit=1','--oom-kill-step=1','--wait=60']:self.assertIn(required,cmd)
        self.assertEqual(cmd[-4:],['29','/test/pw.x','-in','pw.in'])

if __name__=='__main__':unittest.main(verbosity=2)
