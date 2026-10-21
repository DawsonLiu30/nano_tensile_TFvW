from pathlib import Path
import importlib.util,tempfile,unittest,shutil,json
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('restart',ROOT/'scripts/run_nano_qe_restart_20261004.py')
M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
class RestartChecks(unittest.TestCase):
 def test_original_protocol_and_two_cases(self):
  m=M.verify(ROOT/'campaigns/al_nano_qe_restart_20261004');self.assertEqual(len(m['cases']),2)
 def test_reject_changed_physics_even_if_rehashed(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'campaign';shutil.copytree(ROOT/'campaigns/al_nano_qe_restart_20261004',p)
   m=json.loads((p/'qe_manifest.json').read_text());c=m['cases'][0];f=p/c['input']
   text=f.read_text();self.assertIn('degauss = 0.02',text);f.write_text(text.replace('degauss = 0.02','degauss = 0.03'))
   c['input_sha256']=M.sha(f);(p/'qe_manifest.json').write_text(json.dumps(m))
   with self.assertRaises(ValueError):M.verify(p)
 def test_reject_parent_input_tamper(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'campaign';shutil.copytree(ROOT/'campaigns/al_nano_qe_restart_20261004',p)
   m=json.loads((p/'qe_manifest.json').read_text());f=p/m['cases'][0]['parent_input'];f.write_text(f.read_text()+'\n')
   with self.assertRaises(ValueError):M.verify(p)
 def test_memory_guard(self):
  c=M.verify(ROOT/'campaigns/al_nano_qe_restart_20261004')['cases'][0]
  env={'SLURM_NTASKS':'29','SLURM_JOB_NUM_NODES':'1','SLURM_MEM_PER_NODE':str(256*1024)}
  M.GUARD.allocation_guard(c,env,256*2**30)
  with self.assertRaises(ValueError):M.GUARD.allocation_guard(c,env,64*2**30)
 def test_keep_mpi_failure_guards(self):
  cmd=M.GUARD.launch_command('/pw.x')
  for flag in ['--kill-on-bad-exit=1','--oom-kill-step=1','--wait=60']:self.assertIn(flag,cmd)
  self.assertEqual(cmd[-3:],['/pw.x','-in','pw.in'])
if __name__=='__main__':unittest.main()
