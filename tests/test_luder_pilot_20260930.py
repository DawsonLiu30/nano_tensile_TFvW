"""Geometry, frozen-input, cell-constraint and failure-gate tests (no DFT)."""
from pathlib import Path
import importlib.util, json, sys, unittest
import numpy as np
from ase.io import read
from ase.io.espresso import read_fortran_namelist
from ase.filters import FrechetCellFilter
from ase.calculators.calculator import Calculator, all_changes

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
CAMP=REPO/'campaigns/al_luder_validation_nano_20260930'
def mod(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
qe=mod('qe_runner',REPO/'scripts/run_luder_qe_20260930.py')
of=mod('nano_runner',REPO/'scripts/run_nano_dftpy_pilot_20260930.py')

class StressProbe(Calculator):
    implemented_properties=['energy','forces','stress']
    def calculate(self,atoms=None,properties=None,system_changes=all_changes):
        super().calculate(atoms,properties,system_changes)
        self.results=dict(energy=0.,forces=np.zeros((len(atoms),3)),stress=np.array([1.,2.,3.,.4,.5,.6]))

class PilotTests(unittest.TestCase):
    def test_qe_all_inputs_and_geometry(self):
        m=qe.verify(CAMP);self.assertEqual(len(m['cases']),15)
        for c in m['cases']:
            p=CAMP/c['input'];a=read(p,format='espresso-in')
            with p.open() as f:nl,card=read_fortran_namelist(f)
            np.testing.assert_allclose(a.cell.array,c['cell_A'],atol=1e-10,rtol=0)
            np.testing.assert_allclose(a.positions,c['positions_A'],atol=1e-9,rtol=0)
            self.assertEqual(len(a),c['nat']);self.assertEqual(nl['control']['calculation'],c['calculation'])
            self.assertAlmostEqual(nl['system']['ecutwfc'],58.798915481,places=9)
            self.assertAlmostEqual(nl['system']['ecutrho'],470.3913238484,places=9)
            self.assertEqual(nl['system']['smearing'],'mp');self.assertEqual(nl['system']['degauss'],.02)
            self.assertEqual(nl['electrons']['conv_thr'],1e-10)
            if c['calculation']=='vc-relax':self.assertEqual(nl['cell']['cell_dofree'],'z')
            else:self.assertNotIn('cell',nl)
    def test_bulk_reference_set_complete(self):
        m=qe.verify(CAMP);c=m['cases']
        k4=[v['source_case'] for v in c if v['kind']=='bulk_fixed_geometry_scf' and v['kmesh']==[4]*3]
        self.assertEqual(len(k4),8);self.assertIn('0V',k4);self.assertIn('1V',k4)
        self.assertEqual(sum(x['kmesh']==[3]*3 for x in c),3)
    def test_nano_parents_and_grids(self):
        g=json.loads((CAMP/'nano_geometry.json').read_text())
        for case in of.CASE_IDS:
            check,atoms,snap=of.preflight(CAMP,case)
            self.assertLessEqual(max(check['initial_grid_spacing_A']),.15+1e-12)
            other=g['QE_'+case];mine=g['DFTpy_'+case]
            self.assertEqual(other['remaining_parent_ids'],mine['remaining_parent_ids'])
            p=np.array(other['positions_A']);p[:,:2]-=5;p/=other['a0_A']
            q=np.array(mine['positions_A']);q[:,:2]-=5;q/=mine['a0_A']
            np.testing.assert_allclose(p,q,atol=1e-12)
            config=of.make_config(atoms,CAMP/'pseudo/al.lda.recpot',check['protocol'])
            self.assertEqual(config['GRID']['nr'],check['protocol']['grid'])
            self.assertEqual(config['EXC']['xc'],'LDA')
            self.assertEqual(config['KEDF']['x'],.9)
            self.assertEqual(config['KEDF']['y'],.1)
    def test_filter_holds_vacuum_and_tilts(self):
        checked,a,_=of.preflight(CAMP,'R15_0V');a.calc=StressProbe()
        initial=a.cell.array.copy();f=FrechetCellFilter(a,mask=[0,0,1,0,0,0],exp_cell_factor=float(len(a)))
        forces=f.get_forces()[-3:]
        np.testing.assert_allclose(forces[:2],0.,atol=1e-12)
        self.assertNotEqual(forces[2,2],0.)
        pos=f.get_positions();pos[-1,2]+=.1;f.set_positions(pos)
        mask=np.ones((3,3),bool);mask[2,2]=False
        np.testing.assert_allclose(a.cell.array[mask],initial[mask],atol=1e-12)
        self.assertNotEqual(a.cell[2,2],initial[2,2])
    def test_nano_acceptance_requires_axial_and_electronic_convergence(self):
        c,_,_=of.preflight(CAMP,'R15_0V');m=c['protocol']
        metrics=dict(fmax_eV_A=.0002,filter_fmax_eV_A=.0003,stress_GPa=[[9.,0.,0.],[0.,9.,0.],[0.,0.,.01]])
        self.assertTrue(of.acceptance(True,metrics,[{'accepted':True}],m))
        metrics['stress_GPa'][2][2]=.06
        self.assertFalse(of.acceptance(True,metrics,[{'accepted':True}],m))
        metrics['stress_GPa'][2][2]=.01
        self.assertFalse(of.acceptance(True,metrics,[{'accepted':False}],m))
        self.assertFalse(of.acceptance(False,metrics,[{'accepted':True}],m))
    def test_density_integral_failure(self):
        c,a,_=of.preflight(CAMP,'R15_0V');m=c['protocol']
        event=dict(grid_shape=m['grid'],density_integral_electrons=3*len(a)+1.,converged_flag=0,energy_Ha=-1.)
        with self.assertRaises(RuntimeError):of.validate_electronic_event(event,len(a),m)
    def test_missing_qe_output_is_not_accepted(self):
        c=qe.verify(CAMP)['cases'][0]
        self.assertFalse(qe.inspect(CAMP/'path_does_not_exist',c)['accepted'])

if __name__=='__main__': unittest.main(verbosity=2)
