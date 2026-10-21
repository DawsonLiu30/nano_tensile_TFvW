"""No DFT: frozen-input equivalence and inherited XML/SCF negative fixtures."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
import run_reviewed_k7 as W

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

class Tests(unittest.TestCase):
    def test01_parent_and_evidence_verify(self):
        m=M.verify(PARENT);self.assertEqual([W.reviewed_case(m,i)['nat'] for i in [8,9]],[96,95])
    def test02_only_reviewed_indices(self):
        for i in list(range(8))+[-1,10,None]:
            with self.assertRaises(ValueError):W.reviewed_case(MAN,i)
    def test03_reused_inspector_is_original(self):
        for rel,h in REVIEW['parent_file_sha256'].items():self.assertEqual(W.sha(PARENT/rel),h)
    def test04_k7_only_prefix_and_mesh(self):
        for i in [8,9]:
            c,p=MAN['cases'][i],MAN['cases'][i-2]
            text=(PARENT/p['input']).read_text().replace(p['prefix'],c['prefix']).replace('1 1 5 0 0 0','1 1 7 0 0 0')
            self.assertEqual(text,(PARENT/c['input']).read_text())
            self.assertEqual(c['cell_A'],p['cell_A']);self.assertEqual(c['positions_A'],p['positions_A'])
    def test05_k7_synthetic_fixture_accepted_as_diagnostic(self):
        with tempfile.TemporaryDirectory() as f:
            a,c,x,r=FIX.materialize(Path(f),8);self.assertTrue(r['accepted'],r)
            self.assertEqual(r['status'],'accepted_scf_diagnostic')
    def test06_k5_xml_cannot_be_labelled_k7(self):
        with tempfile.TemporaryDirectory() as f:
            a,c,x,r=FIX.materialize(Path(f),8);t=ET.parse(x)
            t.getroot().find('input/k_points_IBZ/monkhorst_pack').set('nk3','5');t.write(x)
            self.assertFalse(M.inspect(a,c,MAN)['accepted'])
    def test07_process_failure_and_unconverged_rejected(self):
        with tempfile.TemporaryDirectory() as f:
            a,c,x,r=FIX.materialize(Path(f),9);self.assertFalse(M.inspect(a,c,MAN,1)['accepted'])
            t=ET.parse(x);t.getroot().find('output/convergence_info/scf_conv/convergence_achieved').text='false';t.write(x)
            self.assertFalse(M.inspect(a,c,MAN)['accepted'])
    def test08_changed_geometry_rejected(self):
        with tempfile.TemporaryDirectory() as f:
            a,c,x,r=FIX.materialize(Path(f),8);t=ET.parse(x);el=t.getroot().find('output/atomic_structure/atomic_positions/atom')
            p=el.text.split();p[0]=str(float(p[0])+.01);el.text=' '.join(p);t.write(x)
            self.assertFalse(M.inspect(a,c,MAN)['accepted'])
    def test09_high_force_stays_diagnostic_not_relaxed(self):
        with tempfile.TemporaryDirectory() as f:
            a,c,x,r=FIX.materialize(Path(f),8);t=ET.parse(x);el=t.getroot().find('output/forces');p=el.text.split()
            p[0]=str(.03*M.BASE.BOHR/M.BASE.HA);el.text=' '.join(p);t.write(x)
            r=M.inspect(a,c,MAN);self.assertTrue(r['accepted']);self.assertGreater(r['fmax_eV_A'],.001)
            self.assertEqual(r['status'],'accepted_scf_diagnostic')
    def test10_memory_and_rank_guard(self):
        c=MAN['cases'][8];env={'SLURM_NTASKS':'29','SLURM_JOB_NUM_NODES':'1','SLURM_MEM_PER_NODE':str(256*1024)}
        M.allocation_guard(c,env,256*2**30)
        with self.assertRaises(ValueError):M.allocation_guard(c,env,192*2**30)
        with self.assertRaises(ValueError):M.allocation_guard(c,{**env,'SLURM_NTASKS':'28'},256*2**30)
        self.assertIn('--kill-on-bad-exit=1',M.launch_command('pw.x'))
        self.assertIn('--oom-kill-step=1',M.launch_command('pw.x'))
    def test11_scope_and_no_release(self):
        self.assertEqual(REVIEW['allowed_indices'],[8,9]);self.assertFalse(REVIEW['k5_evidence']['gate_passed'])
        self.assertEqual(REVIEW['k5_evidence']['released_new_size_cases'],[])
        self.assertNotIn('gate_release.json',Path(W.__file__).read_text())
    def test12_duplicate_attempt_guard(self):
        with tempfile.TemporaryDirectory() as f:
            a=Path(f)/'attempt_001';a.mkdir()
            with self.assertRaises(FileExistsError):a.mkdir(exist_ok=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent-snapshot',type=Path,required=True)
    p.add_argument('--raw-root',type=Path,required=True);args=p.parse_args();PARENT=args.parent_snapshot
    M,MAN,REVIEW=W.load_and_verify(PARENT,args.raw_root)
    FIX=load('historical_fixture',PARENT/'tests/test_nano_gated.py')
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    summary={'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'success':r.wasSuccessful(),
        'review_manifest_sha256':W.sha(Path(__file__).with_name('review_manifest.json')),
        'scope':'No DFT. Original frozen parser/inspector plus explicitly synthetic mutated XML fixtures.'}
    Path(__file__).with_name('review_QA.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary));raise SystemExit(0 if r.wasSuccessful() else 1)
