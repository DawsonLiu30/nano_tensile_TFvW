"""Fast synthetic program tests only: never calls DFTpy or the cluster."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator,all_changes
from ase.filters import FrechetCellFilter
from ase.units import Bohr
from staged_optimizer import POLICY, FixedCellFinish
from recover_nano import load_bundle, check_physics


class Harmonic(Calculator):
    implemented_properties=['energy','forces','stress']
    def calculate(self,atoms=None,properties=None,system_changes=all_changes):
        super().calculate(atoms,properties,system_changes)
        x=self.atoms.positions
        self.results={'energy':float(.5*np.sum(x*x)), 'forces':-x.copy(), 'stress':np.zeros(6)}


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def run_harmonic(self,mode,position,steps=600):
        a=Atoms('Al',positions=[[position,0,0]],cell=np.eye(3)*12,pbc=True);a.calc=Harmonic()
        f=FrechetCellFilter(a,mask=[0,0,1,0,0,0],exp_cell_factor=1.)
        ionic={'optimizer':mode,'optimizer_policy':copy.deepcopy(POLICY),'maxstep_A':.01}
        opt=FixedCellFinish(f,str(self.root/'relax.log'),ionic);seen=[]
        opt.attach(lambda:seen.append(opt.nsteps),interval=1)
        result=opt.run(fmax=.001,steps=steps)
        return result,opt,a,seen
    def test_smallstep_converges_and_records_truth(self):
        ok,opt,a,seen=self.run_harmonic('LBFGS_fixed_cell_finish',.008)
        self.assertTrue(ok)
        self.assertLessEqual(np.linalg.norm(a.get_forces(),axis=1).max(),.001)
        self.assertEqual(seen,list(range(opt.nsteps+1)))
        self.assertTrue(all(x['optimizer']=='LBFGS' and x['maxstep_A']==.01 for x in map(json.loads,(self.root/'optimizer_events.jsonl').read_text().splitlines())))
        np.testing.assert_allclose(a.cell.array,np.eye(3)*12,atol=1e-12)
    def test_each_atomic_move_bounded_by_smallstep(self):
        a=Atoms('Al',positions=[[1.,0,0]],cell=np.eye(3)*12,pbc=True);a.calc=Harmonic()
        f=FrechetCellFilter(a,mask=[0,0,1,0,0,0],exp_cell_factor=1.)
        opt=FixedCellFinish(f,str(self.root/'relax.log'),dict(optimizer='LBFGS_fixed_cell_finish',maxstep_A=.01,optimizer_policy=POLICY))
        rows=[];opt.attach(lambda:rows.append(a.positions.copy()),interval=1)
        opt.run(fmax=.001,steps=10)
        self.assertLessEqual(np.max(np.linalg.norm(np.diff(rows,axis=0),axis=2)),.0100000001)
    def test_bounded_unconverged_not_success(self):
        ok,opt,_,_=self.run_harmonic('LBFGS_fixed_cell_finish',1.,steps=1)
        self.assertFalse(ok);self.assertEqual(opt.nsteps,1)
    def test_policy_revision_is_required(self):
        with self.assertRaises(ValueError):FixedCellFinish(None,'x',{'optimizer':'FIRE','optimizer_policy':{}})
    def fixture(self):
        n=76;cell=np.diag([20.,20.,13.4]);grid=[4,4,4];lat=cell/Bohr
        force=np.zeros((n,3));force[0,0]=.005
        meta={'fmax_eV_A':.005,'filter_fmax_eV_A':.005,'ionic_step':1000,'energy_eV':-4760.,'cell_A':cell.tolist(),'stress_GPa':np.zeros((3,3)).tolist()}
        d=dict(metadata_json=json.dumps(meta),positions_A=np.zeros((n,3)),cell_A=cell,numbers=np.full(n,13),
                pbc=np.ones(3,dtype=bool),parent_site_id=np.arange(n),forces_eV_A=force,
                density_electrons_bohr3=np.ones(grid)*3*n/np.linalg.det(lat),lattice_bohr=lat,
                grid=np.array(grid),origin_bohr=np.zeros(3))
        geometry={'n_atoms':n,'cell_A':cell.tolist(),'remaining_parent_ids':list(range(n))}
        result={'ionic_steps':1000,'energy_eV':-4760.,'fmax_eV_A':.005,'cell_A':cell.tolist()}
        return d,geometry,{'grid':grid,'cell':{'stress_max_abs_GPa':.05}},result
    def bundle_check(self,mutator=None):
        d,g,p,r=self.fixture()
        if mutator:mutator(d)
        z=self.root/'checkpoint.npz';np.savez_compressed(z,**d)
        return load_bundle(z,g,p,r)
    def test_coherent_checkpoint_pass(self):
        a,d,m,ne=self.bundle_check();self.assertEqual(len(a),76);self.assertAlmostEqual(ne,228.,places=9)
    def test_grid_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'grid'):self.bundle_check(lambda d:d.update(grid=np.array([5,4,4])))
    def test_lattice_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'lattice'):self.bundle_check(lambda d:d.update(lattice_bohr=d['lattice_bohr']*1.001))
    def test_electron_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'integral'):self.bundle_check(lambda d:d.update(density_electrons_bohr3=d['density_electrons_bohr3']*1.001))
    def test_nonpositive_density_rejected(self):
        def edit(d):d['density_electrons_bohr3'][0,0,0]=0
        with self.assertRaisesRegex(ValueError,'positive'):self.bundle_check(edit)
    def test_wrong_parent_ids_rejected(self):
        with self.assertRaisesRegex(ValueError,'identity'):self.bundle_check(lambda d:d.update(parent_site_id=np.arange(76)[::-1]))
    def test_pbc_rejected(self):
        with self.assertRaisesRegex(ValueError,'PBC'):self.bundle_check(lambda d:d.update(pbc=np.array([1,1,0])))
    def test_force_metadata_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'force/metadata'):self.bundle_check(lambda d:d.update(forces_eV_A=d['forces_eV_A']*2))
    def test_parent_bad_axial_stress_rejected(self):
        def edit(d):
            m=json.loads(d['metadata_json']);m['stress_GPa'][2][2]=.051;d['metadata_json']=json.dumps(m)
        with self.assertRaisesRegex(ValueError,'axial stress'):self.bundle_check(edit)
    def test_original_large_step_cannot_sneak_back(self):
        with self.assertRaises(ValueError):FixedCellFinish(None,'x',dict(optimizer='LBFGS_fixed_cell_finish',optimizer_policy=POLICY,maxstep_A=.15))
    def test_parent_acceptance_unchanged(self):
        spec=importlib.util.spec_from_file_location('base',Path(__file__).with_name('recovery_base.py'))
        b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
        protocol={'ionic':{'fmax_eV_A':.001,'filter_fmax_eV_A':.001},'cell':{'stress_max_abs_GPa':.05}}
        m={'fmax_eV_A':.0009,'filter_fmax_eV_A':.0009,'stress_GPa':np.diag([100,100,.04]).tolist()}
        events=[{'accepted':True}]
        self.assertTrue(b.acceptance(True,m,events,protocol))
        self.assertFalse(b.acceptance(False,m,events,protocol))
        self.assertFalse(b.acceptance(True,m,[{'accepted':False}],protocol))
        for key in ['fmax_eV_A','filter_fmax_eV_A']:
            mm=copy.deepcopy(m);mm[key]=.00101;self.assertFalse(b.acceptance(True,mm,events,protocol))
        mm=copy.deepcopy(m);mm['stress_GPa'][2][2]=.0501;self.assertFalse(b.acceptance(True,mm,events,protocol))
    def test_actual_parent_physics_frozen(self):
        p={'grid':[140,140,96],'xc':'LDA','lambda_tf':.9,'mu_vw':.1,
           'pseudo':{'sha256':'c2fd4300aa39aa2466149a315e1c7cd87aac6615032db5ded8f237e9c40df803'},
           'electronic':dict(method='LBFGS',econv_Ha_per_atom=1e-9,ncheck=3,maxiter=1000,maxls=50,maxfun=100,densityini='heg',reuse=True),
           'ionic':dict(optimizer='LBFGS',fmax_eV_A=.001,filter_fmax_eV_A=.001,max_steps=1000,maxstep_A=.15),
           'cell':dict(mask=[0,0,1,0,0,0],stress_max_abs_GPa=.05)}
        check_physics(p)
        for field,value in [('grid',[140,140,100]),('xc','PBE'),('mu_vw',.2)]:
            bad=copy.deepcopy(p);bad[field]=value
            with self.assertRaises(ValueError):check_physics(bad)
        for group,field,value in [('ionic','fmax_eV_A',.002),('electronic','method','FIRE'),('cell','mask',[1,1,1,0,0,0])]:
            bad=copy.deepcopy(p);bad[group][field]=value
            with self.assertRaises(ValueError):check_physics(bad)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    print(json.dumps({'tests_run':result.testsRun,'successful':result.wasSuccessful(),
                      'errors':len(result.errors),'failures':len(result.failures),
                      'scope':'synthetic optimizer/checkpoint/acceptance tests; no DFT calculation'}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
