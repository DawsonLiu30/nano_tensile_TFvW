"""Pure program tests; reuse original fixture and inspector, no DFT/submission."""
from pathlib import Path
import argparse,copy,importlib.util,json,shutil,tempfile,unittest,xml.etree.ElementTree as ET
import run_reviewed_k9 as W

def materialize(folder,v):
 # The inherited fixture is explicitly synthetic. Only input identity/mesh are
 # rewritten to k9; it tests inspection mechanics, never physical convergence.
 a,old,x,r=FIX.materialize(folder,8+v);c=MAN['cases'][v]
 shutil.copyfile(W.HERE/c['input'],a/'pw.in')
 t=ET.parse(x);i=t.getroot().find('input')
 i.find('control_variables/prefix').text=c['prefix'];i.find('k_points_IBZ/monkhorst_pack').set('nk3','9')
 y=a/'scratch'/(c['prefix']+'.save')/'data-file-schema.xml';y.parent.mkdir();t.write(y)
 return a,c,y,M.inspect(a,c,PMAN)

class Tests(unittest.TestCase):
 def test01_all_parent_and_stageproof_hashes_verify(self):
  self.assertEqual(len(MAN['cases']),2);self.assertAlmostEqual(MAN['k5_to_k7_pair_change_eV'],.010049289958260488,places=12)
 def test02_only_two_diagnostics(self):
  for i in [-1,2,6,7,8,9,None]:
   with self.assertRaises(ValueError):W.select(MAN,i)
 def test03_exact_input_equivalence_and_structure(self):
  W.verify_overlay(M,PMAN,MAN,PARENT)
  for c in MAN['cases']:
   self.assertLessEqual(c['nat'],250);self.assertGreaterEqual(min(c['cell_A'][i][i] for i in range(3)),10)
   self.assertEqual(c['resources']['memory_GiB'],256)
 def test04_rehashed_overlay_physics_mutation_rejected(self):
  for field,value in [('nat',94),('resources',dict(nodes=1,ntasks=29,memory_GiB=128,slurm_hours=72)),('nbnd',180),('pbc',[1,1,0])]:
   bad=copy.deepcopy(MAN);bad['cases'][0][field]=value
   with self.assertRaises(ValueError):W.verify_overlay(M,PMAN,bad,PARENT)
 def test05_actualmesh9_synthetic_fixture_accepted(self):
  with tempfile.TemporaryDirectory() as d:
   a,c,x,r=materialize(Path(d),0);self.assertTrue(r['accepted'],r);self.assertEqual(r['status'],'accepted_scf_diagnostic')
 def test06_mesh7_cannot_be_mislabelled9(self):
  with tempfile.TemporaryDirectory() as d:
   a,c,x,r=materialize(Path(d),0);t=ET.parse(x);t.getroot().find('input/k_points_IBZ/monkhorst_pack').set('nk3','7');t.write(x)
   self.assertFalse(M.inspect(a,c,PMAN)['accepted'])
 def test07_prefix_mismatch_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   a,c,x,r=materialize(Path(d),1);t=ET.parse(x);t.getroot().find('input/control_variables/prefix').text='wrong';t.write(x)
   self.assertFalse(M.inspect(a,c,PMAN)['accepted'])
 def test08_unconverged_or_nonzero_exit_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   a,c,x,r=materialize(Path(d),1);self.assertFalse(M.inspect(a,c,PMAN,1)['accepted'])
   t=ET.parse(x);t.getroot().find('output/convergence_info/scf_conv/convergence_achieved').text='false';t.write(x)
   self.assertFalse(M.inspect(a,c,PMAN)['accepted'])
 def test09_changed_fixed_geometry_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   a,c,x,r=materialize(Path(d),0);t=ET.parse(x);el=t.getroot().find('output/atomic_structure/atomic_positions/atom');p=el.text.split();p[0]=str(float(p[0])+.01);el.text=' '.join(p);t.write(x)
   self.assertFalse(M.inspect(a,c,PMAN)['accepted'])
 def test10_cutoff_mismatch_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   a,c,x,r=materialize(Path(d),0);t=ET.parse(x);t.getroot().find('input/basis/ecutwfc').text='20';t.write(x)
   self.assertFalse(M.inspect(a,c,PMAN)['accepted'])
 def test11_high_force_is_only_diagnostic(self):
  with tempfile.TemporaryDirectory() as d:
   a,c,x,r=materialize(Path(d),0);t=ET.parse(x);el=t.getroot().find('output/forces');p=el.text.split();p[0]=str(.03*M.BASE.BOHR/M.BASE.HA);el.text=' '.join(p);t.write(x)
   r=M.inspect(a,c,PMAN);self.assertTrue(r['accepted']);self.assertGreater(r['fmax_eV_A'],.001);self.assertEqual(r['status'],'accepted_scf_diagnostic')
 def test12_allocation_guard(self):
  c=MAN['cases'][0];env={'SLURM_NTASKS':'29','SLURM_JOB_NUM_NODES':'1','SLURM_MEM_PER_NODE':str(256*1024)}
  M.allocation_guard(c,env,256*2**30)
  with self.assertRaises(ValueError):M.allocation_guard(c,{**env,'SLURM_NTASKS':'28'},256*2**30)
  with self.assertRaises(ValueError):M.allocation_guard(c,env,192*2**30)
 def test13_no_new_size_release(self):
  self.assertEqual(MAN['allowed_indices'],[0,1]);self.assertNotIn('gate_release.json',Path(W.__file__).read_text())
  self.assertIn('--kill-on-bad-exit=1',M.launch_command('pw.x'));self.assertIn('--oom-kill-step=1',M.launch_command('pw.x'))
 def test14_parent_guard_still_present(self):
  self.assertIn("require(c['stage']!='conditional_k7'",Path(M.__file__).read_text())

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--parent-snapshot',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
 p.add_argument('--k7-entry',type=Path,required=True);p.add_argument('--k7-evidence-root',type=Path);a=p.parse_args();PARENT=a.parent_snapshot
 M,PMAN,MAN=W.load_and_verify(PARENT,a.root,a.k7_entry,a.k7_evidence_root or a.root)
 spec=importlib.util.spec_from_file_location('frozen_fixture',PARENT/'tests/test_nano_gated.py');FIX=importlib.util.module_from_spec(spec);spec.loader.exec_module(FIX)
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 summary={'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'success':r.wasSuccessful(),
          'scope':'No DFT. Original frozen inspector reused with explicitly synthetic k9 XML fixtures and actual hash/input/stageproof checks.',
          'overlay_manifest_sha256':W.sha(W.HERE/'overlay_manifest.json')}
 (W.HERE/'program_QA.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary));raise SystemExit(0 if r.wasSuccessful() else 1)
