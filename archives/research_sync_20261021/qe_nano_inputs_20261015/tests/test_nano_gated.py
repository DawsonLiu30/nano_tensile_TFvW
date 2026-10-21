"""No DFT: real k3 parser fixture + explicitly synthetic gate mutations."""
from pathlib import Path
import copy,gzip,hashlib,json,shutil,sys,tempfile,unittest,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import run_nano_gated_20261015 as M
MAN=json.loads((ROOT/'qe_manifest.json').read_text());BL=json.loads((ROOT/'reference/k3_baseline.json').read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def copied(folder):
 d=Path(folder)/'campaign';shutil.copytree(ROOT,d,ignore=shutil.ignore_patterns('__pycache__'));return d
def materialize(root,index,synthetic=True):
 c=copy.deepcopy(MAN['cases'][index]);v=c['vacancy_count'];a=root/c['id']/'attempt_001';a.mkdir(parents=True)
 (a/'pseudo').mkdir();shutil.copyfile(ROOT/MAN['pseudo_file'],a/'pseudo/Al_PAW_PBE.UPF')
 (a/'pw.in').write_bytes((ROOT/c['input']).read_bytes());(a/'pw.out').write_bytes(gzip.decompress((ROOT/f'tests/fixture/R15_{v}V_pw.out.gz').read_bytes()))
 (a/'pw.stderr').write_text('');dump(a/'provenance.json',{'pseudo_sha256':MAN['pseudo_sha256']})
 xml=gzip.decompress((ROOT/f'tests/fixture/R15_{v}V_data-file-schema.xml.gz').read_bytes());t=ET.fromstring(xml)
 if synthetic:
  inp=t.find('input');inp.find('control_variables/calculation').text=c['calculation'];inp.find('control_variables/prefix').text=c['prefix'];inp.find('k_points_IBZ/monkhorst_pack').set('nk3',str(c['kmesh'][2]))
  ac=inp.find('atomic_structure')
  for key,row in zip(['a1','a2','a3'],c['cell_A']):ac.find('cell/'+key).text=' '.join(str(v/M.BASE.BOHR) for v in row)
  for node,row in zip(ac.findall('atomic_positions/atom'),c['positions_A']):node.text=' '.join(str(v/M.BASE.BOHR) for v in row)
 x=a/'scratch'/(c['prefix']+'.save')/'data-file-schema.xml';x.parent.mkdir(parents=True);ET.ElementTree(t).write(x,encoding='utf8')
 r=M.inspect(a,c,MAN);r['returncode']=0;dump(a/'result.json',r)
 return a,c,x,r
def historical_relax(root):
 a=root/'historical_R15_0V';a.mkdir();(a/'pseudo').mkdir();shutil.copyfile(ROOT/MAN['pseudo_file'],a/'pseudo/Al_PAW_PBE.UPF')
 for name in ['pw.in','pw.out']:(a/name).write_bytes(gzip.decompress((ROOT/f'tests/fixture/R15_0V_{name}.gz').read_bytes()))
 (a/'pw.stderr').write_text('');dump(a/'provenance.json',{'pseudo_sha256':MAN['pseudo_sha256']})
 x=a/'scratch/nano_R15_0V.save/data-file-schema.xml';x.parent.mkdir(parents=True);x.write_bytes(gzip.decompress((ROOT/'tests/fixture/R15_0V_data-file-schema.xml.gz').read_bytes()))
 p=M.parse_input((a/'pw.in').read_text());c={'id':'historical_R15_0V','calculation':'vc-relax','kind':'historical_axial_relaxation','prefix':'nano_R15_0V','nat':96,'nbnd':183,'input_sha256':M.sha(a/'pw.in'),'cell_A':p['cell_A'],'positions_A':p['positions_A'],'kmesh':[1,1,3]}
 return a,c,x
class Tests(unittest.TestCase):
 def test01_verify_all_physical_inputs(self):self.assertEqual(len(M.verify(ROOT)['cases']),10)
 def test02_original_inspector_and_memory_guard_are_byte_preserved(self):
  src=Path('C:/OFDFT/nano_tensile_TFvW/scripts')
  if src.exists():
   for name in ['run_luder_qe_20260930.py','run_nano_qe_memory_recovery_20261001.py']:self.assertEqual(M.sha(ROOT/'scripts'/name),M.sha(src/name))
 def test03_rehashed_input_changes_rejected(self):
  changes=[('58.798915481','50.0'),("cell_dofree = 'z'","cell_dofree = 'all'"),('1 1 3 0 0 0','1 1 1 0 0 0'),('forc_conv_thr = 1.94469038114889e-05','forc_conv_thr = 0.001'),('conv_thr = 1e-10','conv_thr = 1e-06'),('smearing = \'mp\'','smearing = \'gaussian\''),('tot_charge = 0.0','tot_charge = 1.0')]
  for old,new in changes:
   with self.subTest(old=old),tempfile.TemporaryDirectory() as f:
    d=copied(f);m=json.loads((d/'qe_manifest.json').read_text());p=d/m['cases'][0]['input'];txt=p.read_text();self.assertIn(old,txt);p.write_text(txt.replace(old,new));m['cases'][0]['input_sha256']=M.sha(p);dump(d/'qe_manifest.json',m)
    with self.assertRaises(ValueError):M.verify(d)
 def test04_rehashed_pseudo_rejected(self):
  with tempfile.TemporaryDirectory() as f:
   d=copied(f);m=json.loads((d/'qe_manifest.json').read_text());p=d/m['pseudo_file'];p.write_bytes(p.read_bytes()+b'changed');m['pseudo_sha256']=M.sha(p);dump(d/'qe_manifest.json',m)
   with self.assertRaises(ValueError):M.verify(d)
 def test05_wrong_pbc_and_memory_rejected(self):
  for field,value in [('pbc',[False,False,True]),('resources',{'nodes':1,'ntasks':29,'memory_GiB':64,'slurm_hours':72})]:
   with tempfile.TemporaryDirectory() as f:
    d=copied(f);m=json.loads((d/'qe_manifest.json').read_text());m['cases'][0][field]=value;dump(d/'qe_manifest.json',m)
    with self.assertRaises(ValueError):M.verify(d)
 def test06_accepted_scf_fixture(self):
  with tempfile.TemporaryDirectory() as f:
   a,c,x,r=materialize(Path(f),6);self.assertTrue(r['accepted'],r)
 def test07_k3_xml_cannot_be_mislabelled_k5(self):
  with tempfile.TemporaryDirectory() as f:
   a,c,x,r=materialize(Path(f),6);t=ET.parse(x);t.getroot().find('input/k_points_IBZ/monkhorst_pack').set('nk3','3');t.write(x)
   self.assertFalse(M.inspect(a,c,MAN)['accepted'])
 def test08_failed_scf_and_process_are_rejected(self):
  with tempfile.TemporaryDirectory() as f:
   a,c,x,r=materialize(Path(f),6);self.assertFalse(M.inspect(a,c,MAN,1)['accepted'])
   t=ET.parse(x);t.getroot().find('output/convergence_info/scf_conv/convergence_achieved').text='false';t.write(x);self.assertFalse(M.inspect(a,c,MAN)['accepted'])
 def test09_fixed_geometry_mismatch_rejected(self):
  with tempfile.TemporaryDirectory() as f:
   a,c,x,r=materialize(Path(f),6);t=ET.parse(x);el=t.getroot().find('output/atomic_structure/atomic_positions/atom');p=el.text.split();p[0]=str(float(p[0])+.01);el.text=' '.join(p);t.write(x);self.assertFalse(M.inspect(a,c,MAN)['accepted'])
 def test10_passing_gate_releases_six_and_hash_guards(self):
  with tempfile.TemporaryDirectory() as f:
   root=Path(f)
   for i in [6,7]:materialize(root,i)
   rel=root/'gate_release.json';g=M.gate(ROOT,root,rel);self.assertTrue(g['passed'],g);M.release_guard(ROOT,root,rel,MAN['cases'][0])
   (root/MAN['cases'][6]['id']/'attempt_001/pw.out').write_text('changed')
   with self.assertRaises(ValueError):M.release_guard(ROOT,root,rel,MAN['cases'][0])
 def test11_each_gate_criterion_fail_closed(self):
  base=[json.loads((ROOT/f'reference/R15_{v}V_k3_result.json').read_text()) for v in [0,1]]
  for field,value in [('accepted',False),('final_scf_accuracy_Ry',2e-10),('fmax_eV_A',.002),('max_force_component_eV_A',.0008),('stress_QE_GPa',[[0,0,0],[0,0,0],[0,0,.1]])]:
   rs=copy.deepcopy(base);rs[1][field]=value;self.assertFalse(M.evaluate_gate(rs,BL)['passed'],field)
  rs=copy.deepcopy(base);rs[1]['energy_eV']+=.006;self.assertFalse(M.evaluate_gate(rs,BL)['passed'])
 def test12_high_force_scf_is_diagnostic_not_relaxed(self):
  with tempfile.TemporaryDirectory() as f:
   root=Path(f);a,c,x,r=materialize(root,6);t=ET.parse(x);e=t.getroot().find('output/forces');v=e.text.split();v[0]=str(.03*M.BASE.BOHR/M.BASE.HA);e.text=' '.join(v);t.write(x)
   r=M.inspect(a,c,MAN);self.assertTrue(r['accepted']);self.assertGreater(r['fmax_eV_A'],.001);r['returncode']=0;dump(a/'result.json',r);materialize(root,7)
   g=M.gate(ROOT,root,root/'gate_release.json');self.assertFalse(g['passed']);self.assertFalse((root/'gate_release.json').exists());self.assertTrue((root/'GATE_FAILED.txt').exists())
 def test13_missing_upstream_output_blocks(self):
  with tempfile.TemporaryDirectory() as f:
   g=M.gate(ROOT,Path(f),Path(f)/'gate_release.json');self.assertFalse(g['passed']);self.assertFalse((Path(f)/'gate_release.json').exists())
 def test14_no_release_no_expensive_run(self):
  with tempfile.TemporaryDirectory() as f:
   with self.assertRaises(ValueError):M.release_guard(ROOT,Path(f),Path(f)/'absent.json',MAN['cases'][0])
 def test15_actual_allocation_and_rank_failures_rejected(self):
  c=MAN['cases'][6];env={'SLURM_NTASKS':'29','SLURM_JOB_NUM_NODES':'1','SLURM_MEM_PER_NODE':str(192*1024)}
  M.allocation_guard(c,env,192*2**30)
  with self.assertRaises(ValueError):M.allocation_guard(c,env,128*2**30)
  for k,v in [('SLURM_NTASKS','28'),('SLURM_JOB_NUM_NODES','2'),('SLURM_MEM_PER_NODE',str(64*1024))]:
   with self.assertRaises(ValueError):M.allocation_guard(c,{**env,k:v},192*2**30)
  self.assertIn('--kill-on-bad-exit=1',M.launch_command('pw.x'));self.assertIn('--oom-kill-step=1',M.launch_command('pw.x'))
 def test16_fixture_compression_hashes(self):
  for x in json.loads((ROOT/'tests/fixture/manifest.json').read_text()):
   p=ROOT/'tests/fixture'/x['file'];self.assertEqual(M.sha(p),x['compressed_sha256']);self.assertEqual(hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest(),x['uncompressed_sha256'])
 def test17_historical_relaxation_is_accepted(self):
  with tempfile.TemporaryDirectory() as f:
   a,c,x=historical_relax(Path(f));r=M.inspect(a,c,MAN);self.assertTrue(r['accepted'],r)
 def test18_relax_force_stress_and_final_scf_markers_enforced(self):
  for mode in ['norm','component','stress','transverse','bfgs','final_scf','timeout']:
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as f:
    a,c,x=historical_relax(Path(f));t=ET.parse(x);o=t.getroot().find('output')
    if mode in ['norm','component']:
     el=o.find('forces');v=el.text.split();v[0]=str((.002 if mode=='norm' else .0008)*M.BASE.BOHR/M.BASE.HA);el.text=' '.join(v)
    elif mode=='stress':
     el=o.find('stress');v=el.text.split();v[8]=str(.1/M.BASE.GPA);el.text=' '.join(v)
    elif mode=='transverse':
     el=o.find('atomic_structure/cell/a1');v=el.text.split();v[0]=str(float(v[0])+.01);el.text=' '.join(v)
    elif mode=='timeout':o.find('convergence_info/scf_conv/convergence_achieved').text='false'
    else:
     q=a/'pw.out';text=q.read_text();text=text.replace('bfgs converged','optimizer marker removed') if mode=='bfgs' else text.replace('Final scf calculation at the relaxed structure','removed final SCF marker');q.write_text(text)
    t.write(x);self.assertFalse(M.inspect(a,c,MAN)['accepted'],mode)
if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);r=unittest.TextTestRunner(verbosity=2).run(suite)
 dump(ROOT/'runner_tests.json',{'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'success':r.wasSuccessful(),'scope':'No DFT; real historical k3 fixtures and explicitly synthetic SCF/gate mutation tests.'})
 sys.exit(0 if r.wasSuccessful() else 1)
