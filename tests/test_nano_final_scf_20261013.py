from pathlib import Path
import importlib.util,tempfile,unittest,shutil,json
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('final',ROOT/'scripts/run_nano_final_scf_20261013.py');M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
C=ROOT/'campaigns/al_nano_R20_final_scf_20261013'
class FinalSCFChecks(unittest.TestCase):
 def test_input_and_restart(self):
  m=M.verify(C);self.assertEqual(m['cases'][0]['nat'],175)
  for x in ["startingwfc = 'file'","startingpot = 'file'","restart_mode = 'restart'","calculation = 'scf'"]:self.assertIn(x,(C/'pw.in').read_text())
 def test_reject_rehashed_science_changes(self):
  changes=[('ecutwfc = 58.7989154810','ecutwfc = 40.0'),('degauss = 0.02','degauss = 0.03'),('conv_thr = 1.0d-10','conv_thr = 1.0d-8'),('1 1 3 0 0 0','1 1 1 0 0 0'),("startingwfc = 'file'","startingwfc = 'atomic+random'")]
  for old,new in changes:
   with self.subTest(field=old),tempfile.TemporaryDirectory() as d:
    p=Path(d)/'campaign';shutil.copytree(C,p);m=json.loads((p/'qe_manifest.json').read_text());f=p/'pw.in';f.write_text(f.read_text().replace(old,new));m['cases'][0]['input_sha256']=M.sha(f);(p/'qe_manifest.json').write_text(json.dumps(m))
    with self.assertRaises(ValueError):M.verify(p)
 def test_geometry_tamper_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'campaign';shutil.copytree(C,p);f=p/'final_geometry.card';f.write_text(f.read_text()+'\n')
   with self.assertRaises(ValueError):M.verify(p)
 def test_final_force_gates_are_required(self):
  r={'fmax_eV_A':.00046,'max_force_component_eV_A':.00042,'stress_QE_GPa':[[0]*3,[0]*3,[0,0,.00012]]};M.accept_final_forces(r)
  for key,val in [('fmax_eV_A',.0011),('max_force_component_eV_A',.00051)]:
   with self.assertRaises(ValueError):M.accept_final_forces(r|{key:val})
  with self.assertRaises(ValueError):M.accept_final_forces(r|{'stress_QE_GPa':[[0]*3,[0]*3,[0,0,.051]]})
 def test_resources_and_mpi_failure_guards(self):
  c=M.verify(C)['cases'][0];env={'SLURM_NTASKS':'29','SLURM_JOB_NUM_NODES':'1','SLURM_MEM_PER_NODE':str(256*1024)}
  M.GUARD.allocation_guard(c,env,256*2**30)
  with self.assertRaises(ValueError):M.GUARD.allocation_guard(c,env,64*2**30)
  for f in ['--kill-on-bad-exit=1','--oom-kill-step=1','--wait=60']:self.assertIn(f,M.GUARD.launch_command('/pw.x'))
if __name__=='__main__':unittest.main()
