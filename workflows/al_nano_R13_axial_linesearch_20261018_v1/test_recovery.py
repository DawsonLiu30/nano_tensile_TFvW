"""Synthetic axial/line-search tests; no DFT or remote access."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator,all_changes
from ase.filters import FrechetCellFilter
from staged_optimizer import AxialLineSearch,POLICY,repeating_cycle
import recovery_base as base

class HarmonicCell(Calculator):
    implemented_properties=['energy','forces','stress']
    def __init__(self,target,Lz,**kwargs):super().__init__(**kwargs);self.target=np.asarray(target);self.Lz=Lz
    def calculate(self,atoms=None,properties=None,system_changes=all_changes):
        super().calculate(atoms,properties,system_changes)
        x=self.atoms.positions;u=x-self.target;strain=np.log(self.atoms.cell[2,2]/self.Lz);K=2.
        stress=np.einsum('ij,ik->jk',u,x)/self.atoms.get_volume();stress[2,2]+=K*strain/self.atoms.get_volume()
        self.results={'energy':float(.5*np.sum(u*u)+.5*K*strain**2),'forces':-u,'stress':stress}

class Tests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def system(self):
        atoms=Atoms('Al2',positions=[[3.01,3,3.01],[6,6.01,6.02]],cell=[12,12,13.4],pbc=True)
        atoms.calc=HarmonicCell([[3,3,3],[6,6,6]],13.45)
        f=FrechetCellFilter(atoms,mask=[0,0,1,0,0,0],scalar_pressure=0,constant_volume=False,exp_cell_factor=2.)
        return atoms,f
    def opt(self,f):return AxialLineSearch(f,str(self.path/'relax.log'),{'optimizer':'BFGSLineSearch_axial','maxstep_A':.005,'optimizer_policy':copy.deepcopy(POLICY)})
    def test_native_axial_filter_compatible_and_descends(self):
        atoms,f=self.system();initial=atoms.get_potential_energy();cell0=atoms.cell.array.copy();o=self.opt(f);checkpoints=[]
        o.attach(lambda:checkpoints.append((o.nsteps,atoms.cell.array.copy())),interval=1)
        ok=o.run(fmax=.001,steps=60)
        self.assertTrue(ok);self.assertLess(atoms.get_potential_energy(),initial)
        self.assertLessEqual(np.linalg.norm(f.get_forces(),axis=1).max(),.001)
        self.assertGreater(abs(atoms.cell[2,2]-cell0[2,2]),1e-5)
        mask=np.ones((3,3),dtype=bool);mask[2,2]=False;np.testing.assert_allclose(atoms.cell.array[mask],cell0[mask],atol=1e-12)
        self.assertEqual([r[0] for r in checkpoints],list(range(o.nsteps+1)))
        rows=[json.loads(x) for x in (self.path/'optimizer_events.jsonl').read_text().splitlines()]
        self.assertIn('max_actual_atomic_displacement_A',rows[-1]);self.assertIn('delta_Lz_A',rows[-1])
        trials=[json.loads(x) for x in (self.path/'ionic_line_search.jsonl').read_text().splitlines()]
        self.assertTrue(all(r['task']=='CONVERGENCE' for r in trials if r['kind']=='line_search_result'))
    def test_filter_gradient_matches_energy_difference(self):
        a,f=self.system();x=f.get_positions().copy();force=f.get_forces().copy();h=1e-6
        for i,j in [(0,2),(len(a)+2,2)]:
            plus=x.copy();minus=x.copy();plus[i,j]+=h;minus[i,j]-=h
            f.set_positions(plus);ep=f.get_potential_energy(force_consistent=False);f.set_positions(minus);em=f.get_potential_energy(force_consistent=False);f.set_positions(x)
            self.assertAlmostEqual(-(ep-em)/(2*h),force[i,j],delta=1e-8)
    def test_warning_not_accepted_even_nonnull_step(self):
        import ase.optimize.bfgslinesearch as m
        class WarningSearch:
            def __init__(self,*a,**kw):self.task='WARNING: ROUNDING ERRORS PREVENT PROGRESS'
            def _line_search(self,*a,**kw):return .1,a[5],a[5],False
        a,f=self.system();o=self.opt(f);seen=[];o.attach(lambda:seen.append(o.nsteps))
        with patch.object(m,'LineSearch',WarningSearch):
            with self.assertRaisesRegex(RuntimeError,'rejected'):o.run(fmax=.001,steps=2)
        self.assertEqual(seen,[0])
    def test_unconverged_FG_not_accepted(self):
        import ase.optimize.bfgslinesearch as m
        class FGSearch:
            def __init__(self,*a,**kw):self.task='FG'
            def _line_search(self,*a,**kw):return .1,a[5],a[5],True
        a,f=self.system();o=self.opt(f)
        with patch.object(m,'LineSearch',FGSearch):
            with self.assertRaisesRegex(RuntimeError,'did not establish'):o.run(fmax=.001,steps=2)
    def test_repeated_four_cycle_detected(self):
        hist=[{'x':np.ones((5,3))*(i%4)*.0001,'energy_eV':-5+(i%4)*1e-5,'filter_fmax_eV_A':.003+(i%4)*.0001} for i in range(13)]
        self.assertEqual(repeating_cycle(hist)['period'],4)
        for i,row in enumerate(hist):row['energy_eV']-=i*.0001
        self.assertIsNone(repeating_cycle(hist))
    def test_cycles_stop_after_checkpoint_observer(self):
        a,f=self.system();o=self.opt(f);seen=[];o.attach(lambda:seen.append(o.nsteps))
        with patch('staged_optimizer.repeating_cycle',return_value={'period':4}):
            with self.assertRaisesRegex(base.BudgetExpired,'cycle'):o.run(fmax=.001,steps=60)
        self.assertEqual(seen,[0])
    def test_limit_and_acceptance_unchanged(self):
        a,f=self.system();o=self.opt(f)
        with self.assertRaises(ValueError):o.run(fmax=.002,steps=60)
        with self.assertRaises(ValueError):o.run(fmax=.001,steps=61)
        p={'ionic':{'fmax_eV_A':.001,'filter_fmax_eV_A':.001},'cell':{'stress_max_abs_GPa':.05}}
        m={'fmax_eV_A':.0009,'filter_fmax_eV_A':.0009,'stress_GPa':np.diag([100,100,.04]).tolist()}
        self.assertTrue(base.acceptance(True,m,[{'accepted':True}],p));m['filter_fmax_eV_A']=.0101;self.assertFalse(base.acceptance(True,m,[{'accepted':True}],p))

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    print(json.dumps({'tests':r.testsRun,'passed':r.testsRun-len(r.failures)-len(r.errors),'failures':len(r.failures),'errors':len(r.errors),'scope':'Synthetic only, no DFT'}));raise SystemExit(0 if r.wasSuccessful() else 1)
